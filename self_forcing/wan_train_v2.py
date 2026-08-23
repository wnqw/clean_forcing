"""
v2 corrector training (idea-method.md #6 closed-loop upgrades), initialized from the v1 LoRA:
  DAgger    — train on the AGGREGATED pool: round-0 (base-rollout histories, pairs_k48.pt)
              + round-1 (corrected-rollout histories, pairs_k48_dagger1.pt). Fixes covariate shift.
  contract  — drift-contraction surrogate: commit the corrected one-step x0 as frame k..k+b
              (WITH grad), splice it into the history for block k+b, and penalize the velocity
              gap there vs the clean teacher. Optimizes the accumulation mechanism directly:
                min E_k ‖ v_corr(z', [h, x0_hat]) − v_θ(z', h_clean') ‖²
LOSS_MODE=dagger | both   (contract term weighted CW_LOSS)
Run:  LOSS_MODE=both python -u wan_train_v2.py
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters

DEVICE = "cuda"
LOSS_MODE = os.environ.get("LOSS_MODE", "both")     # dagger | both
K, W = int(os.environ.get("K", 48)), 9
STEPS = int(os.environ.get("STEPS", 600))
LR = 2e-4                                            # continue-training LR (< v1's 5e-4)
WARMUP = 40
CW_LOSS = float(os.environ.get("CW_LOSS", 0.5))     # contraction-term weight
EVAL_EVERY = 100
OUT = "wan_cache"
POOLS = os.environ.get("POOLS", "pairs_k48.pt,pairs_k48_dagger1.pt").split(",")     # DAgger aggregation (round-0 + round-1)
INIT = os.environ.get("INIT", "lora_r_phi_k48.pt")
CKPT = os.environ.get("CKPT", f"lora_r_phi_v2_{LOSS_MODE}.pt")


def velocity_tf(pipe, cond, history, x0_cur, z_t, t):
    Wh, nf = history.shape[1], z_t.shape[1]
    clean_x = torch.cat([history, x0_cur], 1)
    noisy = torch.cat([history, z_t], 1)
    ts = torch.cat([torch.zeros((1, Wh), device=z_t.device, dtype=torch.float32),
                    torch.full((1, nf), float(t), device=z_t.device, dtype=torch.float32)], 1)
    flow, _ = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts, clean_x=clean_x)
    return flow[:, Wh:]


def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(True)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    _ab = os.environ.get("ADAPTED_BASE")
    if _ab:
        _sd = torch.load(_ab, map_location="cpu")["merged"]
        pipe.generator.model.load_state_dict({k: v.to(torch.bfloat16) for k, v in _sd.items()}, strict=False)
        print(f"adapted base loaded: {_ab}", flush=True)
    model = pipe.generator.model
    apply_lora(model, rank=16)
    sd = torch.load(os.path.join(OUT, INIT), map_location="cpu")["lora"]
    for i, p in enumerate(lora_parameters(model)):
        p.data.copy_(sd[i].to(p.device, p.dtype))
    model.gradient_checkpointing = True
    print(f"v2 LOSS_MODE={LOSS_MODE} | init from {INIT} | pools {POOLS}", flush=True)
    opt = torch.optim.AdamW(lora_parameters(model), lr=LR)
    START_STEP = 1
    _pp = os.path.join(OUT, CKPT + ".partial")  # PARTIAL resume after interrupt
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
    ks = list(range(W, K - 2 * nfb + 1))             # need room for the k+nfb contraction block
    sched = pipe._initialize_sample_scheduler(torch.zeros(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
    sigmas, tsteps = sched.sigmas.to(DEVICE).float(), sched.timesteps.to(DEVICE).float()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        cond_cache = {c: pipe.text_encoder(text_prompts=[caps[c]]) for c in range(N)}

    def draw(c):
        pool = pools[int(rng.integers(len(pools)))]  # uniform over aggregated rounds
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
            v_clean = velocity_tf(pipe, cond_cache[c], gt_h, x0, z_t, t)
            v_base = velocity_tf(pipe, cond_cache[c], gen_h, x0, z_t, t)
        rt = v_clean - v_base
        set_lora_scale(model, 1.0)
        gctx = torch.enable_grad() if grad else torch.no_grad()
        with gctx:
            v_corr = velocity_tf(pipe, cond_cache[c], gen_h, x0, z_t, t)
            l_dag = ((v_corr - v_clean) ** 2).sum() / ((rt ** 2).sum() + 1e-8)

            l_con = torch.zeros((), device=DEVICE)
            if LOSS_MODE == "both":
                # commit corrected one-step x0 (grad flows through v_corr) -> next block's history
                x0_hat = z_t - sig * v_corr
                k2 = k + nfb
                idx2 = int(rng.integers(len(tsteps))); t2 = tsteps[idx2]; sig2 = sigmas[idx2]
                x0_next = genc[:, k2:k2 + nfb]
                eps2 = torch.randn(x0_next.shape, device=DEVICE, dtype=x0_next.dtype)
                z2 = (1 - sig2) * x0_next + sig2 * eps2
                h2_gen = torch.cat([genc[:, k2 - W:k], x0_hat.to(genc.dtype)], 1)    # spliced committed frames
                with torch.no_grad():
                    set_lora_scale(model, 0.0)
                    v_clean2 = velocity_tf(pipe, cond_cache[c], gtc[:, k2 - W:k2], x0_next, z2, t2)
                    v_base2 = velocity_tf(pipe, cond_cache[c], genc[:, k2 - W:k2], x0_next, z2, t2)
                set_lora_scale(model, 1.0)
                v_corr2 = velocity_tf(pipe, cond_cache[c], h2_gen, x0_next, z2, t2)
                l_con = ((v_corr2 - v_clean2) ** 2).sum() / (((v_clean2 - v_base2) ** 2).sum() + 1e-8)
        return l_dag, l_con

    # val = dagger loss on held-out clips (R^2 = 1 - loss, same normalization as v1)
    @torch.no_grad()
    def val_loss(n=8):
        s = 0.0
        for _ in range(n):
            l_dag, _ = losses(int(rng.choice(val_ids)), grad=False)
            s += l_dag.item()
        return s / n

    for step in range(START_STEP, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        l_dag, l_con = losses(int(rng.choice(train_ids)), grad=True)
        loss = l_dag + CW_LOSS * l_con
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
        opt.step()
        if step % 200 == 0:
            torch.save({"lora": [p.detach().cpu() for p in lora_parameters(model)], "step": step},
                       os.path.join(OUT, CKPT + ".partial"))
        if step % EVAL_EVERY == 0 or step == 1:
            vl = val_loss()
            print(f"step {step:4d} | dag {l_dag.item():.4f} | con {float(l_con):.4f} | val dag-loss {vl:.4f} (R^2 {1-vl:+.3f})", flush=True)

    torch.save({"lora": {i: p.detach().cpu() for i, p in enumerate(lora_parameters(model))}},
               os.path.join(OUT, CKPT))
    vl = val_loss(16)
    print(f"v2 done | final val dag-loss {vl:.4f} (R^2 {1-vl:+.3f}) | saved {OUT}/{CKPT}", flush=True)


if __name__ == "__main__":
    main()
