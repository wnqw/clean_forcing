"""
Faithful r_target on Wan2.1-1.3B (step 1, the real go/no-go).

Same as the toy r_target check, on the real base:
  - h_clean = GT video history (Disney clip frames -> Wan VAE latents)
  - h_gen   = self-rollout seeded from the clip's GT context + caption (drifts from GT)
  - r_target(t,k) = || v_theta(z_t^k, h_clean, t) - v_theta(z_t^k, h_gen, t) ||,  z_t anchored on GT[k]
Both velocities via the teacher-forcing clean_x path (verified). Averaged over clips.

Near-zero -> no real drift signal. High-t concentrated + grows with k -> green-light corrector.
Run:  python -u wan_r_target_faithful.py
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import glob
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan_r_target_check import wan_velocity

DEVICE = "cuda"
K = 21          # latent frames (81 pixel frames); >21 trips KV-cache eviction
W = 9           # history window (k>=W -> constant teacher-forcing seq len)
NCTX = 3        # GT latent frames used to seed the self-rollout
NCLIPS = 3
NSEED = 2
OUT = "outputs_wan_rtarget"


def load_clip_latent(vae, path, n_pix=81):
    r = iio.get_reader(path)
    frames = [fr for i, fr in enumerate(r) if i < n_pix]
    r.close()
    arr = torch.from_numpy(np.stack(frames)).float() / 127.5 - 1
    arr = F.interpolate(arr.permute(0, 3, 1, 2), size=(480, 832), mode="bilinear", align_corners=False)
    pix = arr.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return vae.encode_to_latent(pix.to(next(vae.parameters()).dtype))


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    nfb = pipe.num_frame_per_block

    vids = [l.strip() for l in open("data_gt/disney/videos.txt") if l.strip()]
    caps = [l.strip() for l in open("data_gt/disney/prompt.txt") if l.strip()]
    sched = pipe._initialize_sample_scheduler(torch.zeros(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
    sigmas = sched.sigmas.to(DEVICE).float()
    tsteps = sched.timesteps.to(DEVICE).float()
    tidx = np.linspace(0, len(tsteps) - 1, 6).astype(int)
    ks = [k for k in [9, 12, 18] if k >= W and k + nfb <= K]
    e_norm = np.zeros((len(tidx), len(ks)))
    rel = np.zeros((len(tidx), len(ks)))
    nseen = 0

    for c in range(NCLIPS):
        path = os.path.join("data_gt/disney", vids[c])
        gt_lat = load_clip_latent(pipe.vae, path)[:, :K]
        if gt_lat.shape[1] < K:
            continue
        cond = pipe.text_encoder(text_prompts=[caps[c]])
        # GT-seeded self-rollout -> drifted history
        torch.manual_seed(c)
        noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
        _, gen_lat = pipe.inference(noise=noise, text_prompts=[caps[c]],
                                    initial_latent=gt_lat[:, :NCTX].to(torch.bfloat16), return_latents=True)
        gen_lat = gen_lat[:, :K]
        print(f"clip {c}: gt {tuple(gt_lat.shape)} gen {tuple(gen_lat.shape)}", flush=True)
        for ki, k in enumerate(ks):
            gt_hist = gt_lat[:, k - W:k].to(torch.bfloat16)
            gen_hist = gen_lat[:, k - W:k].to(torch.bfloat16)
            x0 = gt_lat[:, k:k + nfb].to(torch.bfloat16)
            for ti, idx in enumerate(tidx):
                t = tsteps[idx]; sig = sigmas[idx]
                for s in range(NSEED):
                    g = torch.Generator(device=DEVICE).manual_seed(int(1000 * c + 100 * k + idx + s))
                    eps = torch.randn(x0.shape, generator=g, device=DEVICE, dtype=x0.dtype)
                    z_t = (1 - sig) * x0 + sig * eps
                    v_clean = wan_velocity(pipe, cond, gt_hist, x0, z_t, t)
                    v_gen = wan_velocity(pipe, cond, gen_hist, x0, z_t, t)
                    r = (v_clean - v_gen).float()
                    e_norm[ti, ki] += r.flatten().norm().item()
                    rel[ti, ki] += r.flatten().norm().item() / (v_gen.float().flatten().norm().item() + 1e-8)
        nseen += 1
    e_norm /= (nseen * NSEED); rel /= (nseen * NSEED)

    np.savez(os.path.join(OUT, "wan_rtarget_faithful.npz"), e_norm=e_norm, rel=rel, ks=ks,
             tsteps=[int(tsteps[i]) for i in tidx])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, M, ttl in [(ax[0], e_norm, "E||r_target||"), (ax[1], rel, "rel ||r_target||/||v_gen||")]:
        im = a.imshow(M, aspect="auto", origin="lower", cmap="viridis")
        a.set_xticks(range(len(ks))); a.set_xticklabels(ks); a.set_xlabel("frame k")
        a.set_yticks(range(len(tidx))); a.set_yticklabels([int(tsteps[i]) for i in tidx]); a.set_ylabel("timestep t")
        a.set_title(ttl); fig.colorbar(im, ax=a)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "wan_rtarget_faithful.png"), dpi=120)

    hi = rel[:2].mean(); lo = rel[-2:].mean()      # rows high-noise (t~999) -> low-noise (t~208)
    kgrow = float(np.corrcoef(ks, e_norm.mean(0))[0, 1]) if len(ks) > 1 else float("nan")
    print(f"\n================ Wan FAITHFUL r_target ({nseen} clips) ================")
    print(f"mean rel ||r_target||/||v_gen|| = {rel.mean():.3f} | high-noise {hi:.3f} / low-noise {lo:.3f} | corr(||r||,k)={kgrow:+.2f}")
    print(f"saved {OUT}/wan_rtarget_faithful.png")
    if rel.mean() < 0.02:
        print("RESULT: ~no real drift signal on Wan -> reconsider before building the corrector.")
    else:
        print("RESULT: real drift signal PRESENT" + (" + high-noise concentrated" if hi > lo else "")
              + (" + grows with k" if kgrow > 0 else "") + " -> GREEN-LIGHT LoRA corrector on Wan.")


if __name__ == "__main__":
    main()
