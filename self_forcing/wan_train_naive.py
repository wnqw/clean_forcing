"""
Naive-LoRA baseline (AutoRefiner's "+LoRA" adapted): SAME LoRA, SAME GT clips, but trained
with the STANDARD flow-matching loss on clean data — no clean-history counterfactual teacher:
    L = || v_{theta+LoRA}(z_t, h_gt, t) - (eps - x0_gt) ||^2
Isolates the teacher: is the counterfactual what matters, or does ordinary fine-tuning on the
same 40 clean clips already fix drift? Saves corr_naive_k48.pt in the standard lora format.
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
K, W = 48, 9
STEPS = int(os.environ.get("STEPS", 800))
LR = 5e-4
WARMUP = 60
EVAL_EVERY = 100
OUT = "wan_cache"
CKPT = "corr_naive_k48.pt"


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
    model = pipe.generator.model
    apply_lora(model, rank=16)
    set_lora_scale(model, 1.0)
    model.gradient_checkpointing = True
    opt = torch.optim.AdamW(lora_parameters(model), lr=LR)
    print(f"naive-LoRA (standard FM loss on GT) | {STEPS} steps", flush=True)

    d = torch.load(os.path.join(OUT, "pairs_k48.pt"), map_location="cpu")
    gt, caps = d["gt"], d["captions"]
    N = gt.shape[0]
    nval = max(4, N // 8)
    train_ids, val_ids = list(range(N - nval)), list(range(N - nval, N))
    nfb = pipe.num_frame_per_block
    ks = list(range(W, K - nfb + 1))
    sched = pipe._initialize_sample_scheduler(torch.zeros(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
    sigmas, tsteps = sched.sigmas.to(DEVICE).float(), sched.timesteps.to(DEVICE).float()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        cond_cache = {c: pipe.text_encoder(text_prompts=[caps[c]]) for c in range(N)}

    def fm_loss(c, grad):
        gtc = gt[c:c + 1].to(DEVICE).to(torch.bfloat16)
        k = int(rng.choice(ks)); idx = int(rng.integers(len(tsteps))); t = tsteps[idx]; sig = sigmas[idx]
        h, x0 = gtc[:, k - W:k], gtc[:, k:k + nfb]
        eps = torch.randn(x0.shape, device=DEVICE, dtype=x0.dtype)
        z_t = (1 - sig) * x0 + sig * eps
        target = (eps - x0).float()
        gctx = torch.enable_grad() if grad else torch.no_grad()
        with gctx:
            v = velocity_tf(pipe, cond_cache[c], h, x0, z_t, t)
            return ((v.float() - target) ** 2).mean()

    @torch.no_grad()
    def val_loss(n=8):
        return float(np.mean([fm_loss(int(rng.choice(val_ids)), grad=False).item() for _ in range(n)]))

    for step in range(1, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        loss = fm_loss(int(rng.choice(train_ids)), grad=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            print(f"step {step:4d} | fm-loss {loss.item():.4f} | val {val_loss():.4f}", flush=True)

    torch.save({"lora": {i: p.detach().cpu() for i, p in enumerate(lora_parameters(model))}},
               os.path.join(OUT, CKPT))
    print(f"naive done | val {val_loss(16):.4f} | saved {OUT}/{CKPT}", flush=True)


if __name__ == "__main__":
    main()
