"""
History Guidance baseline (DFoT fractional HG, BAgger w=1.2) on the in-domain 15-rollout
protocol: v = v(z|h_sigma-noised) + w*(v(z|h_clean) - v(z|h_noised)), dual KV-cache branches,
4 forwards/step. Training-free. Reference: baseline 0.491/29.6, v1 0.166/36.9, v2 0.131/60.5.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import pyiqa
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline

DEVICE = "cuda"
K, NCTX, LATE = int(os.environ.get("K", 48)), 3, 32
NHELD = int(os.environ.get("NHELD", 5))
SEEDS = list(range(int(os.environ.get("NSEEDS", 3))))
HG_W = float(os.environ.get("HG_W", 1.2))
HG_SIGMA = float(os.environ.get("HG_SIGMA", 0.5))
OUT = "wan_cache"


def saturation(pix):
    x = ((pix.float() + 1) / 2).clamp(0, 1)
    cmax, cmin = x.amax(1), x.amin(1)
    return ((cmax - cmin) / (cmax + 1e-6)).mean(dim=(1, 2)).cpu().numpy()


@torch.no_grad()
def musiq_pf(metric, pix):
    x = ((pix.float() + 1) / 2).clamp(0, 1)
    return torch.cat([metric(x[i:i + 16]).flatten() for i in range(0, x.shape[0], 16)]).cpu().numpy()


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    musiq = pyiqa.create_metric("musiq", device=DEVICE)
    d = torch.load(os.path.join(OUT, "pairs_k48.pt"), map_location="cpu")
    gt, caps = d["gt"], d["captions"]
    held = list(range(gt.shape[0] - NHELD, gt.shape[0]))
    P_CTX, P_CEIL = (NCTX - 1) * 4 + 1, (21 - 1) * 4 + 1

    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    pipe.hg_scale, pipe.hg_sigma = HG_W, HG_SIGMA      # set BEFORE first inference (cache alloc)
    print(f"HG w={HG_W} sigma={HG_SIGMA} | {NHELD} clips x {len(SEEDS)} seeds | K={K}", flush=True)

    sat_d, late = [], []
    for c in held:
        seed_lat = gt[c:c + 1, :NCTX].to(DEVICE).to(torch.bfloat16)
        for s in SEEDS:
            torch.manual_seed(1000 * s + c)
            noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
            _, lat = pipe.inference(noise=noise, text_prompts=[caps[c]], initial_latent=seed_lat, return_latents=True)
            pix = pipe.vae.decode_to_pixel(lat[:, :K])[0]
            sat = saturation(pix); ref = sat[P_CTX:P_CTX + 8].mean()
            sat_d.append(float(np.abs(sat - ref)[P_CEIL:].mean()))
            late.append(float(musiq_pf(musiq, pix)[-LATE:].mean()))
        print(f"clip {c} done", flush=True)

    np.savez(os.path.join(OUT, "hg_eval.npz"), sat=np.array(sat_d), late=np.array(late))
    print(f"\nHG(w={HG_W}) | sat_drift {np.mean(sat_d):.3f}±{np.std(sat_d):.3f} | MUSIQ_late {np.mean(late):.2f}±{np.std(late):.2f}")
    print("(baseline 0.491±0.169 / 29.6±5.6; v1 0.166 / 36.9; v2 0.131 / 60.5)")


if __name__ == "__main__":
    main()
