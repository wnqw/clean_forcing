"""
Axis-2 eval (objective, same LoRA parameterization): long-horizon corrected-rollout drift for
  ours = velocity + clean-history teacher   (lora_r_phi_k48.pt)
  dmd  = AutoRefiner-style noise + DMD, no clean teacher   (corr_dmd_k48.pt)
vs baseline. Isolates the SUPERVISION. Metric: mean|Δsat| over pixel frames 81-189, 4 held-out clips.
Saves wan_cache/axis2_drift.png + table. Fresh base per config.
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
OUT = "wan_cache"
PAIRS = "pairs_k48.pt"
CONFIGS = {"baseline": None, "ours (velocity+clean-teacher)": "lora_r_phi_k48.pt",
           "dmd (AutoRefiner noise+DMD)": "corr_dmd_k48.pt"}


def saturation(pix):
    x = ((pix.float() + 1) / 2).clamp(0, 1)
    cmax, cmin = x.amax(1), x.amin(1)
    return ((cmax - cmin) / (cmax + 1e-6)).mean(dim=(1, 2)).cpu().numpy()


@torch.no_grad()
def rollout_curve(pipe, val_ids, gt, caps):
    curve = None
    for c in val_ids:
        seed = gt[c:c + 1, :NCTX].to(DEVICE).to(torch.bfloat16)
        torch.manual_seed(c)
        noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
        _, lat = pipe.inference(noise=noise, text_prompts=[caps[c]], initial_latent=seed, return_latents=True)
        sat = saturation(pipe.vae.decode_to_pixel(lat[:, :K])[0])
        ref = sat[(NCTX - 1) * 4 + 1:(NCTX - 1) * 4 + 9].mean()
        dv = np.abs(sat - ref)
        curve = dv if curve is None else curve + dv
    return curve / len(val_ids)


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    d = torch.load(os.path.join(OUT, PAIRS), map_location="cpu")
    gt, caps = d["gt"], d["captions"]
    val_ids = list(range(gt.shape[0] - NCLIPS, gt.shape[0]))
    P_CTX, P_CEIL = (NCTX - 1) * 4 + 1, (21 - 1) * 4 + 1

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
        curves[name] = rollout_curve(pipe, val_ids, gt, caps)
        print(f"[{name:32s}] mean|Δsat| (pix {P_CEIL}-{len(curves[name])}) = {curves[name][P_CEIL:].mean():.4f}", flush=True)
        del pipe; torch.cuda.empty_cache()

    NF = len(curves["baseline"]); frames = np.arange(NF)
    plt.figure(figsize=(8, 5))
    for name in CONFIGS:
        plt.plot(frames[P_CTX:], curves[name][P_CTX:], lw=1.5, label=name)
    plt.axvline(P_CEIL, ls="--", c="gray", lw=1, label="old K≤21 ceiling")
    plt.xlabel("pixel frame"); plt.ylabel("|Δsaturation from start| (drift)")
    plt.title(f"Axis-2 objective ablation (same LoRA form, K={K}, {NCLIPS} clips)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "axis2_drift.png"), dpi=130)

    base = curves["baseline"][P_CEIL:].mean()
    print("\n============ Axis-2 objective ablation (drift beyond ceiling) ============")
    for name in CONFIGS:
        m = curves[name][P_CEIL:].mean()
        red = "-" if name == "baseline" else f"{100*(1-m/(base+1e-9)):+.0f}%"
        print(f"{name:34s} | mean|Δsat| {m:.4f} | {red}")
    print("\nsaved wan_cache/axis2_drift.png")


if __name__ == "__main__":
    main()
