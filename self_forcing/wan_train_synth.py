"""
Minimal first-signal: train the LoRA r_phi on Wan (does backward work + does it close the
gen->clean velocity gap?). Native-anchored, teacher-forcing clean_x path.
  teacher  = v_theta(z_t, h_clean)      [LoRA scale 0]
  corrected= v_{theta+LoRA}(z_t, h_gen) [LoRA scale 1]  -> trained toward teacher
  R^2 = 1 - ||v_corr - v_clean||^2 / ||v_clean - v_gen_base||^2   (fraction of drift gap closed)
Run:  python -u wan_train_lora.py
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters, num_lora_params

DEVICE = "cuda"
K, W = 21, 9
STEPS = int(os.environ.get("STEPS", 1500))
M_INNER = 1
LR = 5e-4
WARMUP = 60
EVAL_EVERY = 100
OUT = "wan_cache"
PAIRS = os.environ.get("PAIRS", "pairs_synth.pt")
CKPT = os.environ.get("CKPT", "lora_r_phi_synth.pt")


def velocity_tf(pipe, cond, history, x0_cur, z_t, t):  # grad-capable teacher-forcing velocity
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
    model.gradient_checkpointing = True   # recompute block activations in backward (else OOM on 18.7k-token seq)
    print(f"LoRA params {num_lora_params(model)/1e6:.2f}M | grad-checkpointing on", flush=True)
    opt = torch.optim.AdamW(lora_parameters(model), lr=LR)
    START_STEP = 1
    _pp = os.path.join(OUT, CKPT + ".partial")  # PARTIAL resume after interrupt
    if os.environ.get("RESUME", "1") == "1" and os.path.exists(_pp):
        _pd = torch.load(_pp, map_location="cpu")
        for p, w in zip(lora_parameters(model), _pd["lora"]):
            p.data.copy_(w.to(p.device, p.dtype))
        START_STEP = _pd["step"] + 1
        print(f"resumed from partial step {_pd['step']}", flush=True)

    d = torch.load(os.path.join(OUT, PAIRS), map_location="cpu")
    gt, gen, caps = d["gt"], d["gen"], d["captions"]
    N = gt.shape[0]
    nval = max(4, N // 8)
    train_ids, val_ids = list(range(N - nval)), list(range(N - nval, N))
    nfb = pipe.num_frame_per_block
    ks = [k for k in range(W, K - nfb + 1)]   # sample the full horizon (9..45); large k = genuinely drifted history
    sched = pipe._initialize_sample_scheduler(torch.zeros(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
    sigmas, tsteps = sched.sigmas.to(DEVICE).float(), sched.timesteps.to(DEVICE).float()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        cond_cache = {c: pipe.text_encoder(text_prompts=[caps[c]]) for c in range(N)}

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
        opt.zero_grad(); loss = 0.0; num = den = 0.0
        for _ in range(M_INNER):
            delta, rt = sample_delta(int(rng.choice(train_ids)), grad=True)
            loss = loss + (delta ** 2).sum() / ((rt ** 2).sum() + 1e-8)
            num += (delta ** 2).sum().item(); den += (rt ** 2).sum().item()
        (loss / M_INNER).backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
        opt.step()
        if step % 200 == 0:
            torch.save({"lora": [p.detach().cpu() for p in lora_parameters(model)], "step": step},
                       os.path.join(OUT, CKPT + ".partial"))
        if step % EVAL_EVERY == 0 or step == 1:
            tr = 1 - num / (den + 1e-12)
            print(f"step {step:4d} | loss {(loss/M_INNER).item():.4f} | train R^2 {tr:+.3f} | val R^2 {val_r2():+.3f}", flush=True)

    torch.save({"lora": {n: p.detach().cpu() for n, p in zip(range(len(lora_parameters(model))), lora_parameters(model))}},
               os.path.join(OUT, CKPT))
    print(f"final val R^2 {val_r2(16):+.3f} | saved {OUT}/{CKPT}", flush=True)


if __name__ == "__main__":
    main()
