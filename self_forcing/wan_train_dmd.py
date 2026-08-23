"""
Axis-2 baseline: AutoRefiner-style objective (noise + Distribution-Matching Distillation),
SAME LoRA parameterization as ours, but NO clean-history teacher. Isolates the SUPERVISION:
  ours   = velocity toward v_theta(z_t, h_CLEAN)         [counterfactual clean-history teacher]
  DMD    = match the real-data score via a fake-score critic, conditioned only on h_GEN (drifted)
Two adapters on one frozen base ('gen'=refiner G we ship, 'critic'=fake score). Both scores
share clean_x=[h, x0_G] so its leakage cancels in (x0_fake - x0_real) -> clean DMD gradient.
Flow-matching: x_t=(1-sig)x0+sig*eps, v=eps-x0, so x0 = z - sig*v.
Run:  python -u wan_train_dmd.py
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora_dmd import apply_dual_lora, set_active, adapter_parameters, gen_state_dict

DEVICE = "cuda"
K, W = 48, 9
STEPS = int(os.environ.get("STEPS", 600))              # generator updates
CRITIC_WARMUP = int(os.environ.get("CW", 100))         # critic-only steps before joint (so the fake score is meaningful)
UPDATE_RATIO = int(os.environ.get("RATIO", 5))         # critic updates per generator update (DMD2-style; stabilizes)
LR_G, LR_C = 5e-5, 5e-4                                 # two-timescale: critic much faster than generator
WARMUP = 60
EVAL_EVERY = 100
OUT = "wan_cache"
PAIRS = "pairs_k48.pt"
CKPT = "corr_dmd_k48.pt"


def velocity_tf(pipe, cond, history, x0_ctx, z_t, t):   # teacher-forcing velocity on current block
    Wh, nf = history.shape[1], z_t.shape[1]
    clean_x = torch.cat([history, x0_ctx], 1)
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
    apply_dual_lora(model, rank=16)
    model.gradient_checkpointing = True
    g_params, c_params = adapter_parameters(model, "gen"), adapter_parameters(model, "critic")
    print(f"DMD | gen {sum(p.numel() for p in g_params)/1e6:.2f}M | critic {sum(p.numel() for p in c_params)/1e6:.2f}M", flush=True)
    opt_g = torch.optim.AdamW(g_params, lr=LR_G)
    opt_c = torch.optim.AdamW(c_params, lr=LR_C)

    d = torch.load(os.path.join(OUT, PAIRS), map_location="cpu")
    gt, gen, caps = d["gt"], d["gen"], d["captions"]
    N = gt.shape[0]
    nval = max(4, N // 8)
    train_ids, val_ids = list(range(N - nval)), list(range(N - nval, N))
    nfb = pipe.num_frame_per_block
    ks = list(range(W, K - nfb + 1))
    sched = pipe._initialize_sample_scheduler(torch.zeros(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
    sigmas, tsteps = sched.sigmas.to(DEVICE).float(), sched.timesteps.to(DEVICE).float()
    nT = len(tsteps)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        cond_cache = {c: pipe.text_encoder(text_prompts=[caps[c]]) for c in range(N)}

    def sample_ctx(c):
        genc = gen[c:c + 1].to(DEVICE).to(torch.bfloat16)
        k = int(rng.choice(ks))
        return cond_cache[c], genc[:, k - W:k], genc[:, k:k + nfb]      # cond, h_gen, x0_gen(drifted block to refine)

    def generate(cond, h, x0_gen):                                       # G: SDEdit-style one-step refine of drifted block
        idx = int(rng.integers(nT // 4, nT // 2)); t = tsteps[idx]; sig = sigmas[idx]
        eps = torch.randn(x0_gen.shape, device=DEVICE, dtype=x0_gen.dtype)
        z = (1 - sig) * x0_gen + sig * eps
        set_active(model, "gen")
        v = velocity_tf(pipe, cond, h, x0_gen, z, t)
        return z - sig * v                                               # x0_G

    def score_x0(which, cond, h, x0_ctx, x_t, t, sig):                   # predicted x0 from a score net
        set_active(model, which)                                         # None(base=real) | 'critic'(fake)
        v = velocity_tf(pipe, cond, h, x0_ctx, x_t, t)
        return x_t - sig * v

    def critic_step(cond, h, x0_G):                                      # train fake score to denoise G's samples
        x0_G = x0_G.detach()
        idx = int(rng.integers(nT)); t = tsteps[idx]; sig = sigmas[idx]
        eps = torch.randn(x0_G.shape, device=DEVICE, dtype=x0_G.dtype)
        x_t = (1 - sig) * x0_G + sig * eps
        set_active(model, "critic")
        v = velocity_tf(pipe, cond, h, x0_G, x_t, t)
        loss = F.mse_loss(v.float(), (eps - x0_G).float())
        opt_c.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(c_params, 1.0); opt_c.step()
        return loss.item()

    def gen_step(cond, h, x0_gen):                                       # DMD update: push x0_G toward real manifold
        x0_G = generate(cond, h, x0_gen)
        idx = int(rng.integers(nT)); t = tsteps[idx]; sig = sigmas[idx]
        eps = torch.randn(x0_G.shape, device=DEVICE, dtype=x0_G.dtype)
        x_t = (1 - sig) * x0_G.detach() + sig * eps
        with torch.no_grad():                                            # shared clean_x -> leakage cancels in the diff
            x0_real = score_x0(None, cond, h, x0_G.detach(), x_t, t, sig)
            x0_fake = score_x0("critic", cond, h, x0_G.detach(), x_t, t, sig)
        grad = (x0_fake - x0_real).float()
        norm = (x0_real.float() - x0_G.detach().float()).abs().mean() + 1e-8
        target = (x0_G.float() - grad / norm).detach()
        loss = 0.5 * F.mse_loss(x0_G.float(), target)
        opt_g.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(g_params, 1.0); opt_g.step()
        return loss.item()

    @torch.no_grad()
    def val_dmd_gap(n=8):     # sanity proxy: does the refiner move drifted samples toward the real manifold?
        s = 0.0
        for _ in range(n):
            cond, h, x0_gen = sample_ctx(int(rng.choice(val_ids)))
            x0_G = generate(cond, h, x0_gen)
            idx = int(rng.integers(nT)); t = tsteps[idx]; sig = sigmas[idx]
            eps = torch.randn(x0_G.shape, device=DEVICE, dtype=x0_G.dtype)
            x_t = (1 - sig) * x0_G + sig * eps
            x0_real = score_x0(None, cond, h, x0_G, x_t, t, sig)
            s += (x0_G.float() - x0_real.float()).pow(2).mean().item()   # residual to real manifold (lower=better)
        return s / n

    best_gap = float("inf")
    # critic-only warmup so the fake score tracks G before the adversarial game starts
    for _ in range(CRITIC_WARMUP):
        cond, h, x0_gen = sample_ctx(int(rng.choice(train_ids)))
        with torch.no_grad():
            x0_G = generate(cond, h, x0_gen)
        critic_step(cond, h, x0_G)

    for step in range(1, STEPS + 1):
        for pg in opt_g.param_groups:
            pg["lr"] = LR_G * min(1.0, step / WARMUP)
        for _ in range(UPDATE_RATIO):                              # keep the critic ahead of the generator
            cond, h, x0_gen = sample_ctx(int(rng.choice(train_ids)))
            with torch.no_grad():
                x0_G = generate(cond, h, x0_gen)
            lc = critic_step(cond, h, x0_G)
        cond, h, x0_gen = sample_ctx(int(rng.choice(train_ids)))
        lg = gen_step(cond, h, x0_gen)
        if step % EVAL_EVERY == 0 or step == 1:
            gap = val_dmd_gap()
            tag = ""
            if gap < best_gap:                                 # keep the most stable checkpoint (DMD diverges late)
                best_gap = gap
                torch.save({"lora": gen_state_dict(model)}, os.path.join(OUT, CKPT))
                tag = " <- saved best"
            print(f"step {step:4d} | critic {lc:.4f} | gen {lg:.4f} | val manifold-gap {gap:.4f}{tag}", flush=True)

    print(f"DMD done | best val manifold-gap {best_gap:.4f} | saved {OUT}/{CKPT}", flush=True)


if __name__ == "__main__":
    main()
