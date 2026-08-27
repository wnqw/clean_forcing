"""CF-row v2 corrector: closed-loop upgrades on the CF base, initialized from cf v1.

Exact wan_train_v2.py recipe at the synth K=21 horizon (av2s):
  DAgger      — aggregated pool: round-0 (base-rollout histories) + round-1
                (corrected-rollout histories), uniform.
  contraction — commit corrected one-step x0 (with grad), splice into next block's
                history, penalize velocity gap vs the clean teacher (weight CW_LOSS=0.5).
600 steps, LR 2e-4. Partial resume + durable checkpoints in cf_row/ckpts/.
Run from Self-Forcing repo:  LOSS_MODE=both python -u cf_train_v2.py
"""
import os

import numpy as np
import torch

from cf_common import CKPTS, DEVICE, K, ROW, W, load_cf_pipe

from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters  # noqa: E402

LOSS_MODE = os.environ.get("LOSS_MODE", "both")     # dagger | both
STEPS = int(os.environ.get("STEPS", 600))
LR = 2e-4
WARMUP = 40
CW_LOSS = float(os.environ.get("CW_LOSS", 0.5))
EVAL_EVERY = 100
POOLS = os.environ.get("POOLS", "pairs_synth_cf300.pt,pairs_synth_cf300_dagger1.pt").split(",")
INIT = os.environ.get("INIT", os.path.join(CKPTS, "lora_cf_v1.pt"))
CKPT = os.environ.get("CKPT", f"lora_cf_v2_{LOSS_MODE}.pt")


def velocity_tf(pipe, cond, history, x0_cur, z_t, t):
    Wh, nf = history.shape[1], z_t.shape[1]
    clean_x = torch.cat([history, x0_cur], 1)
    noisy = torch.cat([history, z_t], 1)
    ts = torch.cat([torch.zeros((1, Wh), device=z_t.device, dtype=torch.float32),
                    torch.full((1, nf), float(t), device=z_t.device, dtype=torch.float32)], 1)
    flow, _ = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts, clean_x=clean_x)
    return flow[:, Wh:]


def main():
    torch.set_grad_enabled(True)
    pipe = load_cf_pipe()
    model = pipe.generator.model
    apply_lora(model, rank=16)
    sd = torch.load(INIT, map_location="cpu")["lora"]
    sd = list(sd.values()) if isinstance(sd, dict) else sd
    for p, w in zip(lora_parameters(model), sd):
        p.data.copy_(w.to(p.device, p.dtype))
    model.gradient_checkpointing = True
    print(f"v2 LOSS_MODE={LOSS_MODE} CW={CW_LOSS} | init {INIT} | pools {POOLS}", flush=True)
    opt = torch.optim.AdamW(lora_parameters(model), lr=LR)
    START_STEP = 1
    _pp = os.path.join(CKPTS, CKPT + ".partial")
    if os.environ.get("RESUME", "1") == "1" and os.path.exists(_pp):
        _pd = torch.load(_pp, map_location="cpu")
        for p, w in zip(lora_parameters(model), _pd["lora"]):
            p.data.copy_(w.to(p.device, p.dtype))
        START_STEP = _pd["step"] + 1
        print(f"resumed from partial step {_pd['step']}", flush=True)

    pools = [torch.load(os.path.join(ROW, p), map_location="cpu") for p in POOLS]
    caps = pools[0]["captions"]
    N = pools[0]["gt"].shape[0]
    for pl in pools:
        assert pl["gt"].shape[0] == N
    nval = max(4, N // 8)
    train_ids, val_ids = list(range(N - nval)), list(range(N - nval, N))
    nfb = pipe.num_frame_per_block
    ks = list(range(W, K - 2 * nfb + 1))          # room for the k+nfb contraction block
    sched = pipe._initialize_sample_scheduler(torch.zeros(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
    sigmas, tsteps = sched.sigmas.to(DEVICE).float(), sched.timesteps.to(DEVICE).float()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        cond_cache = {c: pipe.text_encoder(text_prompts=[caps[c]]) for c in range(N)}
    print(f"pairs {N}x{len(pools)} pools (train {len(train_ids)} / val {nval}) | ks {ks[0]}..{ks[-1]}", flush=True)

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
                x0_hat = z_t - sig * v_corr
                k2 = k + nfb
                idx2 = int(rng.integers(len(tsteps))); t2 = tsteps[idx2]; sig2 = sigmas[idx2]
                x0_next = genc[:, k2:k2 + nfb]
                eps2 = torch.randn(x0_next.shape, device=DEVICE, dtype=x0_next.dtype)
                z2 = (1 - sig2) * x0_next + sig2 * eps2
                h2_gen = torch.cat([genc[:, k2 - W:k], x0_hat.to(genc.dtype)], 1)
                with torch.no_grad():
                    set_lora_scale(model, 0.0)
                    v_clean2 = velocity_tf(pipe, cond_cache[c], gtc[:, k2 - W:k2], x0_next, z2, t2)
                    v_base2 = velocity_tf(pipe, cond_cache[c], genc[:, k2 - W:k2], x0_next, z2, t2)
                set_lora_scale(model, 1.0)
                v_corr2 = velocity_tf(pipe, cond_cache[c], h2_gen, x0_next, z2, t2)
                l_con = ((v_corr2 - v_clean2) ** 2).sum() / (((v_clean2 - v_base2) ** 2).sum() + 1e-8)
        return l_dag, l_con

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
                       os.path.join(CKPTS, CKPT + ".partial"))
        if step % EVAL_EVERY == 0 or step == 1:
            vl = val_loss()
            print(f"step {step:4d} | dag {l_dag.item():.4f} | con {float(l_con):.4f} | "
                  f"val dag-loss {vl:.4f} (R^2 {1-vl:+.3f})", flush=True)

    torch.save({"lora": [p.detach().cpu() for p in lora_parameters(model)]},
               os.path.join(CKPTS, CKPT))
    vl = val_loss(16)
    print(f"v2 done | final val dag-loss {vl:.4f} (R^2 {1-vl:+.3f}) | saved {CKPTS}/{CKPT}", flush=True)


if __name__ == "__main__":
    main()
