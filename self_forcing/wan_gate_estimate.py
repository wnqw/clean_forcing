"""
Drift-SNR gate estimation (idea-method.md #1): alpha*(t) = ||bias||^2 / (||bias||^2 + var),
where Delta_m = v_theta(z_t^m, h_clean) - v_theta(z_t^m, h_gen) over M noise seeds at fixed
(clip,k,t). bias = reproducible drift (correct it), var = seed diversity (preserve it).
Finite-sample estimator: alpha_hat = max(0, ||bias||^2 - var/M) / ||bias||^2.
Averaged over samples per schedule timestep -> lookup table wan_cache/gate_alpha.pt.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline

DEVICE = "cuda"
K, W = 48, 9
M = 4                        # noise seeds per (clip,k,t)
NSAMPLES = 4                 # (clip,k) samples per t
OUT = "wan_cache"


def velocity_tf(pipe, cond, history, x0_cur, z_t, t):
    Wh, nf = history.shape[1], z_t.shape[1]
    clean_x = torch.cat([history, x0_cur], 1)
    noisy = torch.cat([history, z_t], 1)
    ts = torch.cat([torch.zeros((1, Wh), device=z_t.device, dtype=torch.float32),
                    torch.full((1, nf), float(t), device=z_t.device, dtype=torch.float32)], 1)
    flow, _ = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts, clean_x=clean_x)
    return flow[:, Wh:]


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None

    d = torch.load(os.path.join(OUT, "pairs_k48.pt"), map_location="cpu")
    gt, gen, caps = d["gt"], d["gen"], d["captions"]
    N = gt.shape[0]
    train_ids = list(range(N - 5))                # gate estimated on train clips only
    nfb = pipe.num_frame_per_block
    sched = pipe._initialize_sample_scheduler(torch.zeros(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
    sigmas, tsteps = sched.sigmas.to(DEVICE).float(), sched.timesteps.to(DEVICE).float()
    rng = np.random.default_rng(0)
    cond_cache = {}

    alphas = np.zeros(len(tsteps))
    for ti in range(len(tsteps)):
        t, sig = tsteps[ti], sigmas[ti]
        a_s = []
        for _ in range(NSAMPLES):
            c = int(rng.choice(train_ids))
            if c not in cond_cache:
                cond_cache[c] = pipe.text_encoder(text_prompts=[caps[c]])
            k = int(rng.integers(W, K - nfb + 1))
            gtc = gt[c:c + 1].to(DEVICE).to(torch.bfloat16); genc = gen[c:c + 1].to(DEVICE).to(torch.bfloat16)
            gt_h, gen_h, x0 = gtc[:, k - W:k], genc[:, k - W:k], genc[:, k:k + nfb]
            deltas = []
            for m in range(M):
                eps = torch.randn(x0.shape, device=DEVICE, dtype=x0.dtype)
                z_t = (1 - sig) * x0 + sig * eps
                dv = velocity_tf(pipe, cond_cache[c], gt_h, x0, z_t, t) - \
                     velocity_tf(pipe, cond_cache[c], gen_h, x0, z_t, t)
                deltas.append(dv.float())
            D = torch.stack(deltas)                       # (M, ...)
            bias = D.mean(0)
            var = (D - bias).pow(2).sum(dim=tuple(range(1, D.ndim))).mean().item()
            b2 = bias.pow(2).sum().item()
            a_s.append(max(0.0, b2 - var / M) / (b2 + 1e-12))
        alphas[ti] = float(np.mean(a_s))
        print(f"t={t.item():7.1f} (sigma {sig.item():.3f}) | alpha* = {alphas[ti]:.3f}", flush=True)

    torch.save({"timesteps": tsteps.cpu(), "alpha": torch.tensor(alphas)}, os.path.join(OUT, "gate_alpha.pt"))
    print(f"\nalpha(t) table: {np.round(alphas, 3).tolist()}")
    print(f"saved {OUT}/gate_alpha.pt")


if __name__ == "__main__":
    main()
