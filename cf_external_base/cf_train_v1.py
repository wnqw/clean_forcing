"""CF-row v1 corrector: LoRA r16 self-attn q/k/v/o on the CF base, synth pairs (all 300).

Exact wan_train_synth.py recipe (av1s): native-anchored teacher-forcing velocity loss,
drift-gap-normalized; K=21, W=9, 1500 steps, LR 5e-4, warmup 60.
  teacher   = v_theta(z_t, h_clean)      [LoRA scale 0]
  corrected = v_{theta+LoRA}(z_t, h_gen) [LoRA scale 1] -> trained toward teacher
Partial resume + durable step checkpoints in cf_row/ckpts/.
Run from Self-Forcing repo:  python -u cf_train_v1.py
"""
import os

import numpy as np
import torch

from cf_common import CKPTS, DEVICE, K, ROW, W, load_cf_pipe

from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters, num_lora_params  # noqa: E402

STEPS = int(os.environ.get("STEPS", 1500))
LR = 5e-4
WARMUP = 60
EVAL_EVERY = 100
SAVE_EVERY = 200
PAIRS = os.environ.get("PAIRS", os.path.join(ROW, "pairs_synth_cf300.pt"))
CKPT = os.environ.get("CKPT", "lora_cf_v1.pt")


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
    model.gradient_checkpointing = True
    print(f"LoRA params {num_lora_params(model)/1e6:.2f}M | grad-checkpointing on", flush=True)
    opt = torch.optim.AdamW(lora_parameters(model), lr=LR)

    START_STEP = 1
    _pp = os.path.join(CKPTS, CKPT + ".partial")
    if os.environ.get("RESUME", "1") == "1" and os.path.exists(_pp):
        _pd = torch.load(_pp, map_location="cpu")
        for p, w in zip(lora_parameters(model), _pd["lora"]):
            p.data.copy_(w.to(p.device, p.dtype))
        START_STEP = _pd["step"] + 1
        print(f"resumed from partial step {_pd['step']}", flush=True)

    d = torch.load(PAIRS, map_location="cpu")
    gt, gen, caps = d["gt"], d["gen"], d["captions"]
    N = gt.shape[0]
    assert N == 300, f"expected 300 pairs, got {N}"
    nval = max(4, N // 8)
    train_ids, val_ids = list(range(N - nval)), list(range(N - nval, N))
    nfb = pipe.num_frame_per_block
    ks = [k for k in range(W, K - nfb + 1)]
    sched = pipe._initialize_sample_scheduler(torch.zeros(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
    sigmas, tsteps = sched.sigmas.to(DEVICE).float(), sched.timesteps.to(DEVICE).float()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        cond_cache = {c: pipe.text_encoder(text_prompts=[caps[c]]) for c in range(N)}
    print(f"pairs {N} (train {len(train_ids)} / val {nval}) | ks {ks[0]}..{ks[-1]}", flush=True)

    def sample_delta(c, grad):
        gtc = gt[c:c + 1].to(DEVICE).to(torch.bfloat16); genc = gen[c:c + 1].to(DEVICE).to(torch.bfloat16)
        k = int(rng.choice(ks)); idx = int(rng.integers(len(tsteps))); t = tsteps[idx]; sig = sigmas[idx]
        gt_hist, gen_hist, x0 = gtc[:, k - W:k], genc[:, k - W:k], genc[:, k:k + nfb]
        eps = torch.randn(x0.shape, device=DEVICE, dtype=x0.dtype)
        z_t = (1 - sig) * x0 + sig * eps
        with torch.no_grad():
            set_lora_scale(model, 0.0)
            v_clean = velocity_tf(pipe, cond_cache[c], gt_hist, x0, z_t, t)
            r_target = v_clean - velocity_tf(pipe, cond_cache[c], gen_hist, x0, z_t, t)
        set_lora_scale(model, 1.0)
        if grad:
            v_corr = velocity_tf(pipe, cond_cache[c], gen_hist, x0, z_t, t)
        else:
            with torch.no_grad():
                v_corr = velocity_tf(pipe, cond_cache[c], gen_hist, x0, z_t, t)
        return v_corr - v_clean, r_target

    @torch.no_grad()
    def val_r2(n=8):
        num = den = 0.0
        for _ in range(n):
            delta, rt = sample_delta(int(rng.choice(val_ids)), grad=False)
            num += (delta ** 2).sum().item(); den += (rt ** 2).sum().item()
        return 1 - num / (den + 1e-12)

    for step in range(START_STEP, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        delta, rt = sample_delta(int(rng.choice(train_ids)), grad=True)
        loss = (delta ** 2).sum() / ((rt ** 2).sum() + 1e-8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
        opt.step()
        if step % SAVE_EVERY == 0:
            torch.save({"lora": [p.detach().cpu() for p in lora_parameters(model)], "step": step},
                       os.path.join(CKPTS, CKPT + ".partial"))
        if step % EVAL_EVERY == 0 or step == 1:
            tr = 1 - loss.item()
            print(f"step {step:4d} | loss {loss.item():.4f} | train R^2 {tr:+.3f} | val R^2 {val_r2():+.3f}", flush=True)

    torch.save({"lora": [p.detach().cpu() for p in lora_parameters(model)]},
               os.path.join(CKPTS, CKPT))
    print(f"final val R^2 {val_r2(16):+.3f} | saved {CKPTS}/{CKPT}", flush=True)


if __name__ == "__main__":
    main()
