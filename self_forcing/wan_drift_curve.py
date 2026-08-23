"""
Long-horizon corrected-rollout drift curve (the money plot). K=48 (>> old 21-frame ceiling).
Per-frame |Δsat from start| averaged over held-out clips, baseline (LoRA scale 0) vs corrected.
Saves wan_cache/drift_curve.png + prints the per-frame table. Question: does r_phi FLATTEN
the drift runaway (saturation -> clipping) over a real horizon, not just shave a mild 21-frame one?
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters

DEVICE = "cuda"
K, NCTX, NCLIPS = 48, 3, 4
SCALES = [0.0, 1.0, 0.5]     # 0 == baseline
OUT = "wan_cache"
PAIRS = "pairs_k48.pt"
CKPT = "lora_r_phi_k48.pt"   # scaled: 40 clips / 800 steps / full-horizon k


def saturation(pix):  # pix (F,3,H,W) in [-1,1] -> mean HSV-sat per frame (F,)
    x = ((pix.float() + 1) / 2).clamp(0, 1)
    cmax, cmin = x.amax(1), x.amin(1)
    return ((cmax - cmin) / (cmax + 1e-6)).mean(dim=(1, 2)).cpu().numpy()


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    model = pipe.generator.model
    apply_lora(model, rank=16)
    sd = torch.load(os.path.join(OUT, CKPT), map_location="cpu")["lora"]
    for i, p in enumerate(lora_parameters(model)):
        p.data.copy_(sd[i].to(p.device, p.dtype))
    print(f"local_attn_size={model.local_attn_size} | loaded LoRA ({len(sd)} tensors) | K={K}", flush=True)

    d = torch.load(os.path.join(OUT, PAIRS), map_location="cpu")
    gt, caps = d["gt"], d["captions"]
    val_ids = list(range(gt.shape[0] - NCLIPS, gt.shape[0]))   # held out from training

    # decode_to_pixel temporally upsamples: latent f -> pixel (f-1)*4+1 (f>0), latent 0 -> pixel 0.
    lat2pix = lambda f: 0 if f == 0 else (f - 1) * 4 + 1
    P_CTX, P_CEIL = lat2pix(NCTX), lat2pix(21)   # context end / old ceiling, in pixel frames

    # per-pixel-frame |Δsat| averaged over clips (sized lazily from first decode)
    curves = {s: None for s in SCALES}
    for c in val_ids:
        seed = gt[c:c + 1, :NCTX].to(DEVICE).to(torch.bfloat16)   # pairs.pt holds K=21 GT; only NCTX used
        for s in SCALES:
            set_lora_scale(model, s)
            torch.manual_seed(c)
            noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
            _, lat = pipe.inference(noise=noise, text_prompts=[caps[c]],
                                    initial_latent=seed, return_latents=True)
            pix = pipe.vae.decode_to_pixel(lat[:, :K])[0]
            sat = saturation(pix)
            ref = sat[P_CTX:P_CTX + 8].mean()
            dv = np.abs(sat - ref)
            curves[s] = dv if curves[s] is None else curves[s] + dv
        print(f"clip {c} done", flush=True)
    for s in SCALES:
        curves[s] /= len(val_ids)

    NF = len(curves[0.0])
    frames = np.arange(NF)
    plt.figure(figsize=(8, 5))
    labels = {0.0: "baseline (α=0)", 1.0: "corrected α=1", 0.5: "corrected α=0.5"}
    for s in SCALES:
        plt.plot(frames[P_CTX:], curves[s][P_CTX:], lw=1.5, label=labels[s])
    plt.axvline(P_CEIL, ls="--", c="gray", lw=1, label="old K≤21 ceiling")
    plt.xlabel("pixel frame"); plt.ylabel("|Δsaturation from start| (drift)")
    plt.title(f"Long-horizon drift: baseline vs corrector (K={K} latent, {len(val_ids)} clips)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    png = os.path.join(OUT, "drift_curve.png")
    plt.savefig(png, dpi=130)
    print(f"\nsaved {png}  ({NF} pixel frames)", flush=True)

    print("\npixframe |  base   α=1    α=0.5")
    for f in range(P_CTX, NF, 12):
        print(f"  {f:4d}   | {curves[0.0][f]:.3f}  {curves[1.0][f]:.3f}  {curves[0.5][f]:.3f}")
    lo = P_CEIL
    print(f"\n=========== beyond-old-ceiling drift (pixel frames {lo}-{NF}) ===========")
    base = curves[0.0][lo:].mean()
    for s in SCALES:
        m = curves[s][lo:].mean()
        red = "" if s == 0.0 else f"  | reduction vs baseline: {100*(1-m/(base+1e-9)):+.0f}%"
        print(f"scale {s}: mean|Δsat| = {m:.4f}{red}")


if __name__ == "__main__":
    main()
