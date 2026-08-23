"""
v3-b corrector: the v2s recipe (DAgger pools + contraction) with ONE change — the teacher.
  past teacher (v2s):  v_theta(z_t, h_clean_past)                    causal, anchors to reference
  oracle teacher (v3b): v_theta_bidi([clean past, z_t, clean future]) uniform-t, encodes progression
Gate (wan_gate_oracle.py, 24 states x 4 seeds): alpha*_oracle 0.838 (past 0.871), info-content
median 0.91 of the drift gap, systematic at all t; blowups are small-gap denominators, handled by
the loss's own per-sample normalization. Teacher = ADAPTED merged weights in bidi mode (beat
pristine 0.838 vs 0.813) — "same frozen checkpoint, bidirectional mode".
Both loss terms swap teachers (DAgger + contraction); everything else identical to v2s.
Val reports R^2 vs BOTH teachers: R2_or is the train objective; R2_past shows how far the
corrector departs from pure anchoring (may go negative — expected, info ~0.9).
Run: ADAPTED_BASE=wan_cache/adapted_base_4000.pt python -u wan_train_v3b.py
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:native")
import numpy as np
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters
from wan_gate_oracle import predict_future

DEVICE = "cuda"
LOSS_MODE = os.environ.get("LOSS_MODE", "both")
K, W = int(os.environ.get("K", 21)), 9
STEPS = int(os.environ.get("STEPS", 600))
LR = 2e-4
WARMUP = 40
CW_LOSS = float(os.environ.get("CW_LOSS", 0.5))
EVAL_EVERY = 100
OUT = "wan_cache"
POOLS = os.environ.get("POOLS", "pairs_synth_adapt.pt,pairs_synth_dagger_adapt.pt").split(",")
INIT = os.environ.get("INIT", "lora_r_phi_synth_adapt.pt")
CKPT = os.environ.get("CKPT", "lora_r_phi_v3b.pt")
ORACLE_TMIN = float(os.environ.get("ORACLE_TMIN", 0))
"""Teacher is the oracle only at t >= this; past-teacher below. Structure (progression)
is decided at high t, sharpness at low t — gating out the oracle at low t avoids the
conditional-mean blur the pure-oracle v3b showed on eyeball (blur, faint color)."""
FUTURE = os.environ.get("FUTURE", "gt")
"""'gt' = v3-b (reference clip's real future) | 'pred' = v3-c (future AR-predicted from the
clean past, deterministic per state — keeps only the past-predictable share, so the student
has no unpredictable target randomness to average into blur). Gate: info 0.93 vs GT's 0.91."""


def velocity_tf(pipe, cond, history, x0_cur, z_t, t):
    Wh, nf = history.shape[1], z_t.shape[1]
    clean_x = torch.cat([history, x0_cur], 1)
    noisy = torch.cat([history, z_t], 1)
    ts = torch.cat([torch.zeros((1, Wh), device=z_t.device, dtype=torch.float32),
                    torch.full((1, nf), float(t), device=z_t.device, dtype=torch.float32)], 1)
    flow, _ = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts, clean_x=clean_x)
    return flow[:, Wh:]


def velocity_oracle(bidi, cond, gtc, z_t, k, nfb, t, sig, future=None):
    ctx = gtc if future is None else torch.cat([gtc[:, :k + nfb], future], 1)
    ctx_noise = torch.randn(ctx.shape, device=ctx.device, dtype=ctx.dtype)
    z_full = (1 - sig) * ctx + sig * ctx_noise
    z_full = torch.cat([z_full[:, :k], z_t, z_full[:, k + nfb:]], 1)
    ts = torch.full((1, ctx.shape[1]), float(t), device=ctx.device, dtype=torch.float32)
    flow, _ = bidi(noisy_image_or_video=z_full, conditional_dict=cond, timestep=ts)
    return flow[:, k:k + nfb]


def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(True)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    _ab = os.environ.get("ADAPTED_BASE")
    merged = None
    if _ab:
        merged = {k: v.to(torch.bfloat16) for k, v in torch.load(_ab, map_location="cpu")["merged"].items()}
        pipe.generator.model.load_state_dict(merged, strict=False)
        print(f"adapted base loaded: {_ab}", flush=True)
    bidi = WanDiffusionWrapper(is_causal=False).to(dtype=torch.bfloat16).cuda()
    if merged is not None:
        bidi.model.load_state_dict(merged, strict=False)
    bidi.model.requires_grad_(False)
    print("bidi oracle teacher ready (adapted weights)" if merged is not None else "bidi oracle teacher ready (pristine)", flush=True)

    model = pipe.generator.model
    apply_lora(model, rank=16)
    sd = torch.load(os.path.join(OUT, INIT), map_location="cpu")["lora"]
    for i, p in enumerate(lora_parameters(model)):
        p.data.copy_(sd[i].to(p.device, p.dtype))
    model.gradient_checkpointing = True
    print(f"v3b LOSS_MODE={LOSS_MODE} | init from {INIT} | pools {POOLS}", flush=True)
    opt = torch.optim.AdamW(lora_parameters(model), lr=LR)
    START_STEP = 1
    _pp = os.path.join(OUT, CKPT + ".partial")
    if os.environ.get("RESUME", "1") == "1" and os.path.exists(_pp):
        _pd = torch.load(_pp, map_location="cpu")
        for p, w in zip(lora_parameters(model), _pd["lora"]):
            p.data.copy_(w.to(p.device, p.dtype))
        START_STEP = _pd["step"] + 1
        print(f"resumed from partial step {_pd['step']}", flush=True)

    pools = [torch.load(os.path.join(OUT, p), map_location="cpu") for p in POOLS]
    caps = pools[0]["captions"]
    N = pools[0]["gt"].shape[0]
    nval = max(4, N // 8)
    train_ids, val_ids = list(range(N - nval)), list(range(N - nval, N))
    nfb = pipe.num_frame_per_block
    ks = list(range(W, K - 2 * nfb + 1))   # future >= nfb latents so the oracle sees a real future
    sched = pipe._initialize_sample_scheduler(torch.zeros(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
    sigmas, tsteps = sched.sigmas.to(DEVICE).float(), sched.timesteps.to(DEVICE).float()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        cond_cache = {c: pipe.text_encoder(text_prompts=[caps[c]]) for c in range(N)}

    def draw(c):
        pool = pools[int(rng.integers(len(pools)))]
        gtc = pool["gt"][c:c + 1].to(DEVICE).to(torch.bfloat16)
        genc = pool["gen"][c:c + 1].to(DEVICE).to(torch.bfloat16)
        k = int(rng.choice(ks))
        return gtc, genc, k

    def losses(c, grad):
        gtc, genc, k = draw(c)
        idx = int(rng.integers(len(tsteps))); t = tsteps[idx]; sig = sigmas[idx]
        gt_h, gen_h, x0 = gtc[:, k - W:k], genc[:, k - W:k], genc[:, k:k + nfb]
        eps = torch.randn(x0.shape, device=DEVICE, dtype=x0.dtype)
        z_t = (1 - sig) * x0 + sig * eps
        with torch.no_grad():
            set_lora_scale(model, 0.0)
            fut = None
            if FUTURE == "pred":  # v3-c: frozen-base prediction, deterministic per (c, k)
                fut = predict_future(pipe, cond_cache[c], gtc[:, :k], nfb, sigmas, tsteps,
                                     n_lat=K - k - nfb, seed=c * 31 + k)
            v_or = velocity_oracle(bidi, cond_cache[c], gtc, z_t, k, nfb, t, sig, future=fut)
            v_clean = velocity_tf(pipe, cond_cache[c], gt_h, x0, z_t, t)   # past teacher, val-only
            v_base = velocity_tf(pipe, cond_cache[c], gen_h, x0, z_t, t)
        v_tgt = v_or if float(t) >= ORACLE_TMIN else v_clean
        rt = v_tgt - v_base
        rt_past = v_clean - v_base
        set_lora_scale(model, 1.0)
        gctx = torch.enable_grad() if grad else torch.no_grad()
        with gctx:
            v_corr = velocity_tf(pipe, cond_cache[c], gen_h, x0, z_t, t)
            l_dag = ((v_corr - v_tgt) ** 2).sum() / ((rt ** 2).sum() + 1e-8)
            l_past = ((v_corr - v_clean) ** 2).sum() / ((rt_past ** 2).sum() + 1e-8)

            l_con = torch.zeros((), device=DEVICE)
            if LOSS_MODE == "both":
                x0_hat = z_t - sig * v_corr
                k2 = k + nfb
                idx2 = int(rng.integers(len(tsteps))); t2 = tsteps[idx2]; sig2 = sigmas[idx2]
                x0_next = genc[:, k2:k2 + nfb]
                eps2 = torch.randn(x0_next.shape, device=DEVICE, dtype=x0_next.dtype)
                z2 = (1 - sig2) * x0_next + sig2 * eps2
                h2_gen = torch.cat([genc[:, k2 - W:k], x0_hat.to(genc.dtype)], 1)
                with torch.no_grad():
                    set_lora_scale(model, 0.0)
                    if float(t2) >= ORACLE_TMIN:
                        # v3-c: reuse the chunk-k continuation shifted one chunk (its first
                        # chunk overlaps z2's slot; the tail is the k2 future)
                        fut2 = fut[:, nfb:] if fut is not None else None
                        v_tgt2 = velocity_oracle(bidi, cond_cache[c], gtc, z2, k2, nfb, t2, sig2,
                                                 future=fut2)
                    else:
                        v_tgt2 = velocity_tf(pipe, cond_cache[c], gtc[:, k2 - W:k2], x0_next, z2, t2)
                    v_base2 = velocity_tf(pipe, cond_cache[c], genc[:, k2 - W:k2], x0_next, z2, t2)
                set_lora_scale(model, 1.0)
                v_corr2 = velocity_tf(pipe, cond_cache[c], h2_gen, x0_next, z2, t2)
                l_con = ((v_corr2 - v_tgt2) ** 2).sum() / (((v_tgt2 - v_base2) ** 2).sum() + 1e-8)
        return l_dag, l_con, l_past

    @torch.no_grad()
    def val_loss(n=8):
        s_or = s_past = 0.0
        for _ in range(n):
            l_dag, _, l_past = losses(int(rng.choice(val_ids)), grad=False)
            s_or += l_dag.item(); s_past += l_past.item()
        return s_or / n, s_past / n

    for step in range(START_STEP, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        l_dag, l_con, _ = losses(int(rng.choice(train_ids)), grad=True)
        loss = l_dag + CW_LOSS * l_con
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
        opt.step()
        if step % 200 == 0:
            torch.save({"lora": [p.detach().cpu() for p in lora_parameters(model)], "step": step},
                       os.path.join(OUT, CKPT + ".partial"))
        if step % EVAL_EVERY == 0 or step == 1:
            vo, vp = val_loss()
            print(f"step {step:4d} | dag {l_dag.item():.4f} | con {float(l_con):.4f} | "
                  f"val R2_or {1-vo:+.3f} R2_past {1-vp:+.3f}", flush=True)

    torch.save({"lora": {i: p.detach().cpu() for i, p in enumerate(lora_parameters(model))}},
               os.path.join(OUT, CKPT))
    vo, vp = val_loss(16)
    print(f"v3b done | final val R2_or {1-vo:+.3f} R2_past {1-vp:+.3f} | saved {OUT}/{CKPT}", flush=True)


if __name__ == "__main__":
    main()
