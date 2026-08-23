"""
r_target signal screen on the Wan2.1-1.3B DF base (step 1, GT-free PROXY).

Faithful r_target needs a clean-history teacher (GT video) — none shipped at Wan res.
This screens the *necessary condition*: does the base velocity depend on history, and is
that dependence high-t concentrated? Uses drift-like history corruption (per-channel
scale+shift in latent space ~ the low-freq color/exposure drift the toy showed):

   r_proxy(t,k) = || v_theta(z_t^k, h_gen, t) - v_theta(z_t^k, corrupt(h_gen), t) ||

Near-zero everywhere -> velocity ignores history -> corrector can't help (STOP).
Substantial + high-t -> promising -> build the faithful GT-history version next.

Builds the reusable KV-cache velocity extractor `wan_velocity(...)`.
Run from repo root:  python -u wan_r_target_check.py
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline

DEVICE = "cuda"
K = 21            # self-rollout horizon (>21 trips a KV-cache eviction bug; 21 is verified-safe)
W = 9             # history window fed to the velocity call
N_SEED = 3        # corruption seeds per (t,k)
FSL = 1560        # frame_seq_length
OUT = "outputs_wan_rtarget"


@torch.no_grad()
def wan_velocity(pipe, cond, history, x0_cur, z_t, t):
    """v_theta for the current chunk given clean `history`, via the model's teacher-forcing path
    (clean_x context + noised current, no KV-cache — the trained mechanism, robust).
    history:(1,W,16,60,104) clean ; x0_cur:(1,nf,...) clean current ; z_t:(1,nf,...) noised -> (1,nf,...) velocity."""
    W = history.shape[1]
    nf = z_t.shape[1]
    clean_x = torch.cat([history, x0_cur], dim=1)                 # (1, W+nf, ...) full clean sequence
    noisy = torch.cat([history, z_t], dim=1)                      # (1, W+nf, ...) history@t=0, current noised
    ts = torch.cat([torch.zeros((1, W), device=z_t.device, dtype=torch.float32),
                    torch.full((1, nf), float(t), device=z_t.device, dtype=torch.float32)], dim=1)
    flow, _ = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts, clean_x=clean_x)
    return flow[:, W:]                                            # velocity for the current chunk


def drift_corrupt(h, strength=0.3, gen=None):
    # low-freq per-channel scale+shift ~ color/exposure drift
    s = h.std()
    scale = 1 + strength * torch.randn(1, 1, h.shape[2], 1, 1, device=h.device, dtype=h.dtype, generator=gen)
    shift = strength * s * torch.randn(1, 1, h.shape[2], 1, 1, device=h.device, dtype=h.dtype, generator=gen)
    return h * scale + shift


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    prompt = ["A serene mountain lake at sunrise, mist rising over still water."]
    cond = pipe.text_encoder(text_prompts=prompt)

    print("self-rollout for generated history ...", flush=True)
    torch.manual_seed(0)
    noise = torch.randn(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
    _, gen_lat = pipe.inference(noise=noise, text_prompts=prompt, return_latents=True)
    gen_lat = gen_lat[:, :K]
    sched = pipe._initialize_sample_scheduler(noise)
    sigmas = sched.sigmas.to(DEVICE).float()             # (n+1,)
    tsteps = sched.timesteps.to(DEVICE).float()           # (n,)
    pipe.corrector = None

    nfb = pipe.num_frame_per_block
    ks = [k for k in [9, 12, 18] if k >= W and k + nfb <= K]  # k>=W -> constant seq len (W+nf) -> stable TF mask
    tidx = np.linspace(0, len(tsteps) - 1, 6).astype(int)
    e_norm = np.zeros((len(tidx), len(ks)))
    rel = np.zeros((len(tidx), len(ks)))
    print(f"grid ks={ks} t={[int(tsteps[i]) for i in tidx]}", flush=True)
    for ki, k in enumerate(ks):
        hist = gen_lat[:, max(0, k - W):k]
        x0 = gen_lat[:, k:k + nfb]
        for ti, idx in enumerate(tidx):
            t = tsteps[idx]; sig = sigmas[idx]
            dn, vn = 0.0, 0.0
            for s in range(N_SEED):
                g = torch.Generator(device=DEVICE).manual_seed(int(100 * k + idx + s))
                eps = torch.randn(x0.shape, generator=g, device=DEVICE, dtype=x0.dtype)
                z_t = (1 - sig) * x0 + sig * eps
                v_gen = wan_velocity(pipe, cond, hist, x0, z_t, t)
                v_cor = wan_velocity(pipe, cond, drift_corrupt(hist, gen=g), x0, z_t, t)
                d = (v_gen - v_cor).float()
                dn += d.flatten().norm().item()
                vn += v_gen.float().flatten().norm().item()
            e_norm[ti, ki] = dn / N_SEED
            rel[ti, ki] = (dn / N_SEED) / (vn / N_SEED + 1e-8)
        print(f"k={k}: rel(t-low..high)={[round(rel[ti,ki],3) for ti in range(len(tidx))]}", flush=True)

    np.savez(os.path.join(OUT, "wan_rtarget.npz"), e_norm=e_norm, rel=rel, ks=ks, tsteps=[int(tsteps[i]) for i in tidx])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, M, ttl in [(ax[0], e_norm, "||r_proxy||"), (ax[1], rel, "rel ||r_proxy||/||v_gen||")]:
        im = a.imshow(M, aspect="auto", origin="lower", cmap="viridis")
        a.set_xticks(range(len(ks))); a.set_xticklabels(ks); a.set_xlabel("frame k")
        a.set_yticks(range(len(tidx))); a.set_yticklabels([int(tsteps[i]) for i in tidx]); a.set_ylabel("timestep t")
        a.set_title(ttl); fig.colorbar(im, ax=a)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "wan_rtarget.png"), dpi=120)

    hi = rel[:2].mean(); lo = rel[-2:].mean()   # rows ordered high-noise (t~999) -> low-noise (t~208)
    print("\n================ Wan r_target PROXY ================")
    print(f"mean rel ||r_proxy||/||v_gen|| = {rel.mean():.3f} | high-noise {hi:.3f} / low-noise {lo:.3f}")
    print(f"saved {OUT}/wan_rtarget.png")
    if rel.mean() < 0.01:
        print("RESULT: velocity ~insensitive to history -> corrector unlikely to help. Investigate.")
    else:
        print("RESULT: velocity IS history-sensitive" + (" and high-noise concentrated." if hi > lo else "."),
              "\nNOTE: proxy (synthetic corruption). Faithful go/no-go needs GT-video clean-history teacher.")


if __name__ == "__main__":
    main()
