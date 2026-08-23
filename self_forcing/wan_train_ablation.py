"""
Axis-1 ablation (parameterization): train the idea-#6 corrector on Wan with 3 different
parameterizations, IDENTICAL data / targets / loss / steps, differing ONLY in form:
  MODE=lora      low-rank deltas on frozen self-attn (5.9M)      -> the proposed form
  MODE=residual  Conv3d+FiLM side-net on frozen base (~1.9M)     -> toy showed weak
  MODE=full      full-finetune of self-attn q/k/v/o (~280M)      -> BAgger-class upper-ish bound
Same native-anchored, teacher-forcing clean_x objective as wan_train_lora_k48.py:
  target v_clean = FROZEN base velocity w/ clean(GT) history;  corrected form fed drifted(gen) history.
  R^2 = 1 - ||v_corr - v_clean||^2 / ||v_clean - v_gen||^2  (fraction of drift gap closed)
Run:  MODE=residual python -u wan_train_ablation.py
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import contextlib
import numpy as np
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters, num_lora_params
from wan.modules.corrector import WanVelocityResidual

DEVICE = "cuda"
MODE = os.environ.get("MODE", "residual")     # lora | residual | full
K, W = 48, 9
STEPS = int(os.environ.get("STEPS", 800))
LR = {"lora": 5e-4, "residual": 2e-4, "full": 1e-5}[MODE]
WARMUP = 60
EVAL_EVERY = 100
OUT = "wan_cache"
PAIRS = "pairs_k48.pt"
CKPT = f"corr_{MODE}_k48.pt"
FULL_TARGETS = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o")


def velocity_tf(pipe, cond, history, x0_cur, z_t, t):  # teacher-forcing velocity on the current block
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
    for p in model.parameters():
        p.requires_grad_(False)

    r_phi = None
    if MODE == "lora":
        apply_lora(model, rank=16)
        model.gradient_checkpointing = True
        trainable = lora_parameters(model)
        frozen_ctx = contextlib.nullcontext                      # scale toggle handles frozen/live
    elif MODE == "residual":
        r_phi = WanVelocityResidual().to(DEVICE)                 # trained in fp32; base stays bf16 (no-grad)
        trainable = list(r_phi.parameters())
        frozen_ctx = contextlib.nullcontext                      # base never modified
    elif MODE == "full":
        full_params = [(n, p) for n, p in model.named_parameters() if any(t in n for t in FULL_TARGETS)]
        for _, p in full_params:
            p.requires_grad_(True)
        model.gradient_checkpointing = True
        trainable = [p for _, p in full_params]
        frozen_snap = {n: p.detach().clone() for n, p in full_params}

        @contextlib.contextmanager
        def _use_frozen():                                       # swap original weights in for the teacher pass
            live = {n: p.detach().clone() for n, p in full_params}
            for n, p in full_params:
                p.data.copy_(frozen_snap[n])
            try:
                yield
            finally:
                for n, p in full_params:
                    p.data.copy_(live[n])
        frozen_ctx = _use_frozen
    else:
        raise ValueError(MODE)

    nparam = sum(p.numel() for p in trainable)
    print(f"MODE={MODE} | trainable {nparam/1e6:.2f}M | LR {LR} | steps {STEPS}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=LR)

    d = torch.load(os.path.join(OUT, PAIRS), map_location="cpu")
    gt, gen, caps = d["gt"], d["gen"], d["captions"]
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

    def set_scale(s):                                            # only lora has a runtime scale
        if MODE == "lora":
            set_lora_scale(model, s)

    def delta_and_target(c, grad):
        gtc = gt[c:c + 1].to(DEVICE).to(torch.bfloat16); genc = gen[c:c + 1].to(DEVICE).to(torch.bfloat16)
        k = int(rng.choice(ks)); idx = int(rng.integers(len(tsteps))); t = tsteps[idx]; sig = sigmas[idx]
        gt_hist, gen_hist, x0 = gtc[:, k - W:k], genc[:, k - W:k], genc[:, k:k + nfb]
        eps = torch.randn(x0.shape, device=DEVICE, dtype=x0.dtype)
        z_t = (1 - sig) * x0 + sig * eps
        with torch.no_grad():                                    # frozen-base teacher + drifted-history baseline
            set_scale(0.0)
            with frozen_ctx():
                v_clean = velocity_tf(pipe, cond_cache[c], gt_hist, x0, z_t, t)
                v_gen = velocity_tf(pipe, cond_cache[c], gen_hist, x0, z_t, t)
        r_target = v_clean - v_gen
        set_scale(1.0)
        gctx = contextlib.nullcontext() if grad else torch.no_grad()
        with gctx:
            if MODE == "residual":
                v_corr = v_gen + r_phi(z_t, torch.full((1, nfb), float(t), device=DEVICE), history=gen_hist)
            else:
                v_corr = velocity_tf(pipe, cond_cache[c], gen_hist, x0, z_t, t)
        return v_corr - v_clean, r_target

    @torch.no_grad()
    def val_r2(n=8):
        num = den = 0.0
        for _ in range(n):
            delta, rt = delta_and_target(int(rng.choice(val_ids)), grad=False)
            num += (delta ** 2).sum().item(); den += (rt ** 2).sum().item()
        return 1 - num / (den + 1e-12)

    for step in range(1, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        delta, rt = delta_and_target(int(rng.choice(train_ids)), grad=True)
        loss = (delta ** 2).sum() / ((rt ** 2).sum() + 1e-8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            tr = 1 - (delta ** 2).sum().item() / ((rt ** 2).sum().item() + 1e-12)
            print(f"step {step:4d} | loss {loss.item():.4f} | train R^2 {tr:+.3f} | val R^2 {val_r2():+.3f}", flush=True)

    if MODE == "lora":
        sd = {"lora": {i: p.detach().cpu() for i, p in enumerate(lora_parameters(model))}}
    elif MODE == "residual":
        sd = {"residual": r_phi.state_dict()}
    else:
        sd = {"full": {n: p.detach().cpu() for n, p in full_params}}
    torch.save(sd, os.path.join(OUT, CKPT))
    print(f"MODE={MODE} final val R^2 {val_r2(16):+.3f} | saved {OUT}/{CKPT}", flush=True)


if __name__ == "__main__":
    main()
