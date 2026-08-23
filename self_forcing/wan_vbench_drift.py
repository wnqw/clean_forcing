"""
VBench-style ΔDriftQuality: re-score the long-horizon rollout with MUSIQ (the exact
no-reference image-quality model VBench's 'imaging_quality' dimension uses), instead of the
saturation proxy. Drift = quality lost from an early window to a late window:
    ΔDriftQuality = mean MUSIQ(early frames) - mean MUSIQ(late frames)   (>0 = quality degraded)
Lower is better. Rolling Forcing (2509.25161) reports this kind of drift for SelfForcing/CausVid.
Compares baseline vs ours (velocity+clean-teacher LoRA) over held-out clips at K=48.
Saves wan_cache/vbench_drift.png (per-frame MUSIQ curve) + table.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import pyiqa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters

DEVICE = "cuda"
K, NCTX, NCLIPS = 48, 3, 4
EARLY, LATE = 32, 32                 # pixel-frame window sizes for the drift delta
OUT = "wan_cache"
PAIRS = "pairs_k48.pt"
CONFIGS = {"baseline": None, "ours (velocity+clean-teacher)": "lora_r_phi_k48.pt"}


@torch.no_grad()
def musiq_per_frame(metric, pix):    # pix (F,3,H,W) in [-1,1] -> MUSIQ score per frame (F,)
    x = ((pix.float() + 1) / 2).clamp(0, 1)
    scores = []
    for i in range(0, x.shape[0], 16):
        scores.append(metric(x[i:i + 16]).flatten())
    return torch.cat(scores).cpu().numpy()


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    musiq = pyiqa.create_metric("musiq", device=DEVICE)
    d = torch.load(os.path.join(OUT, PAIRS), map_location="cpu")
    gt, caps = d["gt"], d["captions"]
    val_ids = list(range(gt.shape[0] - NCLIPS, gt.shape[0]))
    P_CTX = (NCTX - 1) * 4 + 1

    curves = {}
    for name, ckpt in CONFIGS.items():
        pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
        pipe.corrector = None
        model = pipe.generator.model
        if ckpt is not None:
            apply_lora(model, rank=16)
            sd = torch.load(os.path.join(OUT, ckpt), map_location="cpu")["lora"]
            for i, p in enumerate(lora_parameters(model)):
                p.data.copy_(sd[i].to(p.device, p.dtype))
            set_lora_scale(model, 1.0)
        curve = None
        for c in val_ids:
            seed = gt[c:c + 1, :NCTX].to(DEVICE).to(torch.bfloat16)
            torch.manual_seed(c)
            noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
            _, lat = pipe.inference(noise=noise, text_prompts=[caps[c]], initial_latent=seed, return_latents=True)
            mq = musiq_per_frame(musiq, pipe.vae.decode_to_pixel(lat[:, :K])[0])
            curve = mq if curve is None else curve + mq
        curves[name] = curve / len(val_ids)
        e, l = curves[name][P_CTX:P_CTX + EARLY].mean(), curves[name][-LATE:].mean()
        print(f"[{name:32s}] MUSIQ early {e:.2f} late {l:.2f} | ΔDriftQuality {e - l:+.3f}", flush=True)
        del pipe; torch.cuda.empty_cache()

    NF = len(curves["baseline"]); frames = np.arange(NF)
    plt.figure(figsize=(8, 5))
    for name in CONFIGS:
        plt.plot(frames[P_CTX:], curves[name][P_CTX:], lw=1.5, label=name)
    plt.axvline((21 - 1) * 4 + 1, ls="--", c="gray", lw=1, label="old K≤21 ceiling")
    plt.xlabel("pixel frame"); plt.ylabel("MUSIQ imaging quality (higher=better)")
    plt.title(f"VBench imaging-quality over horizon (K={K}, {NCLIPS} clips)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "vbench_drift.png"), dpi=130)

    print("\n============ VBench ΔDriftQuality (MUSIQ early - late; lower=less drift) ============")
    base = None
    for name in CONFIGS:
        e = curves[name][P_CTX:P_CTX + EARLY].mean(); l = curves[name][-LATE:].mean(); dq = e - l
        if base is None:
            base = dq
        red = "" if name == "baseline" else f" | drift reduction vs baseline: {100*(1-dq/(base+1e-9)):+.0f}%"
        print(f"{name:34s} | early {e:.2f} | late {l:.2f} | ΔDriftQuality {dq:+.3f}{red}")
    print("\nsaved wan_cache/vbench_drift.png")


if __name__ == "__main__":
    main()
