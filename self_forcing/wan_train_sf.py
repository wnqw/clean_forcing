"""2nd-host corrector: r_phi_sf on the frozen SF-distilled generator, x0-space counterfactual loss.

  teacher   = x0_theta(z_t, h_clean)        [LoRA scale 0]
  corrected = x0_{theta+LoRA}(z_t, h_gen)   [LoRA scale 1] -> trained toward teacher
  R^2 = 1 - ||x0_corr - x0_clean||^2 / ||x0_clean - x0_gen_base||^2

t is drawn from the 4 distillation timesteps (warped, as at inference); z_t via the host's own
scheduler.add_noise — exactly the renoise op the sampler applies. Same anchoring as v1: shared z_t
from the drifted rollout state, only the history swaps.
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from omegaconf import OmegaConf

from pipeline.causal_inference import CausalInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters, num_lora_params

DEVICE = "cuda"
K, W = 21, 9
STEPS = int(os.environ.get("STEPS", 1500))
LR = 5e-4
WARMUP = 60
EVAL_EVERY = 100
OUT = "wan_cache"
PAIRS = os.environ.get("PAIRS", "pairs_sf.pt").split(",")
INIT = os.environ.get("INIT")  # optional LoRA ckpt to continue from
CKPT = os.environ.get("CKPT", "lora_r_phi_sf.pt")
SF_CKPT = "checkpoints/self_forcing_dmd.pt"


def x0_tf(pipe, cond, history, x0_cur, z_t, t):  # grad-capable teacher-forcing x0 prediction
    Wh, nf = history.shape[1], z_t.shape[1]
    clean_x = torch.cat([history, x0_cur], 1)
    noisy = torch.cat([history, z_t], 1)
    ts = torch.cat([torch.zeros((1, Wh), device=z_t.device, dtype=torch.float32),
                    torch.full((1, nf), float(t), device=z_t.device, dtype=torch.float32)], 1)
    _, pred = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts, clean_x=clean_x)
    return pred[:, Wh:]


def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/self_forcing_dmd.yaml"))
    torch.set_grad_enabled(True)
    pipe = CausalInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    sd = torch.load(SF_CKPT, map_location="cpu")
    pipe.generator.load_state_dict(sd.get("generator", sd.get("generator_ema")))
    pipe = pipe.to(dtype=torch.bfloat16).cuda()
    model = pipe.generator.model
    apply_lora(model, rank=16)
    if INIT:
        _lw = torch.load(INIT, map_location="cpu")["lora"]
        _lw = list(_lw.values()) if isinstance(_lw, dict) else _lw
        for p, w in zip(lora_parameters(model), _lw):
            p.data.copy_(w.to(p.device, p.dtype))
        print(f"continue-training from {INIT}", flush=True)
    model.gradient_checkpointing = True
    print(f"LoRA params {num_lora_params(model)/1e6:.2f}M | 2nd host = SF-distilled", flush=True)
    opt = torch.optim.AdamW(lora_parameters(model), lr=LR)

    pools = [torch.load(os.path.join(OUT, p), map_location="cpu") for p in PAIRS]
    gt = torch.cat([d["gt"] for d in pools])
    gen = torch.cat([d["gen"] for d in pools])
    caps = sum([list(d["captions"]) for d in pools], [])
    N = gt.shape[0]
    nval = max(4, N // 8)
    train_ids, val_ids = list(range(N - nval)), list(range(N - nval, N))
    nfb = pipe.num_frame_per_block
    ks = [k for k in range(W, K - nfb + 1)]
    tlist = pipe.denoising_step_list.to(DEVICE)  # 4 warped distillation timesteps
    rng = np.random.default_rng(0)
    with torch.no_grad():
        cond_cache = {c: pipe.text_encoder(text_prompts=[caps[c]]) for c in range(N)}

    def sample_delta(c, grad):
        gtc = gt[c:c + 1].to(DEVICE).to(torch.bfloat16)
        genc = gen[c:c + 1].to(DEVICE).to(torch.bfloat16)
        k = int(rng.choice(ks))
        t = tlist[int(rng.integers(len(tlist)))]
        gt_hist, gen_hist, x0 = gtc[:, k - W:k], genc[:, k - W:k], genc[:, k:k + nfb]
        eps = torch.randn(x0.shape, device=DEVICE, dtype=x0.dtype)
        z_t = pipe.scheduler.add_noise(
            x0.flatten(0, 1), eps.flatten(0, 1),
            t * torch.ones([x0.shape[0] * x0.shape[1]], device=DEVICE, dtype=torch.long)
        ).unflatten(0, x0.shape[:2])
        with torch.no_grad():
            set_lora_scale(model, 0.0)
            p_clean = x0_tf(pipe, cond_cache[c], gt_hist, x0, z_t, t)
            r_target = p_clean - x0_tf(pipe, cond_cache[c], gen_hist, x0, z_t, t)
        set_lora_scale(model, 1.0)
        if grad:
            p_corr = x0_tf(pipe, cond_cache[c], gen_hist, x0, z_t, t)
        else:
            with torch.no_grad():
                p_corr = x0_tf(pipe, cond_cache[c], gen_hist, x0, z_t, t)
        return p_corr - p_clean, r_target

    @torch.no_grad()
    def val_r2(n=8):
        num = den = 0.0
        for _ in range(n):
            delta, rt = sample_delta(int(rng.choice(val_ids)), grad=False)
            num += (delta ** 2).sum().item()
            den += (rt ** 2).sum().item()
        return 1 - num / (den + 1e-12)

    for step in range(1, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        delta, rt = sample_delta(int(rng.choice(train_ids)), grad=True)
        loss = (delta ** 2).sum() / ((rt ** 2).sum() + 1e-8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f} | val R^2 {val_r2():+.3f}", flush=True)

    torch.save({"lora": [p.detach().cpu() for p in lora_parameters(model)]},
               os.path.join(OUT, CKPT))
    print(f"final val R^2 {val_r2(16):+.3f} | saved {OUT}/{CKPT}", flush=True)


if __name__ == "__main__":
    main()
