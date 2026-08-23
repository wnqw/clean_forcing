"""
Axis-1 ablation eval: long-horizon corrected-rollout drift (beyond old K<=21 ceiling) for
each parameterization of the SAME corrector objective — baseline vs LoRA vs residual side-net
vs full-finetune. Answers: is LoRA specifically the right form, or would any adapter do?
Metric: mean |Δsat from start| over pixel frames 81-189 (the new long horizon), 4 held-out clips.
Saves wan_cache/ablation_drift.png + prints the table. Fresh base per config (no cross-contamination).
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
from wan.modules.corrector import WanVelocityResidual

DEVICE = "cuda"
K, NCTX, NCLIPS, W = 48, 3, 4, 9
OUT = "wan_cache"
PAIRS = "pairs_k48.pt"
CONFIGS = ["baseline", "lora", "residual", "full"]
CKPTS = {"lora": "lora_r_phi_k48.pt", "residual": "corr_residual_k48.pt", "full": "corr_full_k48.pt"}
FULL_TARGETS = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o")


def saturation(pix):
    x = ((pix.float() + 1) / 2).clamp(0, 1)
    cmax, cmin = x.amax(1), x.amin(1)
    return ((cmax - cmin) / (cmax + 1e-6)).mean(dim=(1, 2)).cpu().numpy()


def build_pipe(cfg):
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    return pipe


def configure(pipe, mode):
    model = pipe.generator.model
    if mode == "baseline":
        return
    if mode == "lora":
        apply_lora(model, rank=16)
        sd = torch.load(os.path.join(OUT, CKPTS["lora"]), map_location="cpu")["lora"]
        for i, p in enumerate(lora_parameters(model)):
            p.data.copy_(sd[i].to(p.device, p.dtype))
        set_lora_scale(model, 1.0)
    elif mode == "residual":
        r_phi = WanVelocityResidual().to(DEVICE)
        r_phi.load_state_dict(torch.load(os.path.join(OUT, CKPTS["residual"]), map_location="cpu")["residual"])
        pipe.attach_corrector(r_phi, alpha=1.0)
        pipe.corrector_hist = W                       # match training history window
    elif mode == "full":
        sd = torch.load(os.path.join(OUT, CKPTS["full"]), map_location="cpu")["full"]
        named = dict(model.named_parameters())
        for n, t in sd.items():
            named[n].data.copy_(t.to(named[n].device, named[n].dtype))


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
    for mode in CONFIGS:
        pipe = build_pipe(cfg)                        # fresh base each config
        configure(pipe, mode)
        curves[mode] = rollout_curve(pipe, val_ids, gt, caps)
        m = curves[mode][P_CEIL:].mean()
        print(f"[{mode:9s}] mean|Δsat| (pix {P_CEIL}-{len(curves[mode])}) = {m:.4f}", flush=True)
        del pipe
        torch.cuda.empty_cache()

    NF = len(curves["baseline"])
    frames = np.arange(NF)
    plt.figure(figsize=(8, 5))
    for mode in CONFIGS:
        plt.plot(frames[P_CTX:], curves[mode][P_CTX:], lw=1.5, label=mode)
    plt.axvline(P_CEIL, ls="--", c="gray", lw=1, label="old K≤21 ceiling")
    plt.xlabel("pixel frame"); plt.ylabel("|Δsaturation from start| (drift)")
    plt.title(f"Axis-1 parameterization ablation (K={K}, {NCLIPS} held-out clips)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "ablation_drift.png"), dpi=130)

    base = curves["baseline"][P_CEIL:].mean()
    print("\n============ Axis-1 parameterization ablation (drift beyond ceiling) ============")
    print(f"{'config':10s} | mean|Δsat| | reduction vs baseline | params")
    pcount = {"baseline": "-", "lora": "5.9M", "residual": "1.9M", "full": "283M"}
    for mode in CONFIGS:
        m = curves[mode][P_CEIL:].mean()
        red = "-" if mode == "baseline" else f"{100*(1-m/(base+1e-9)):+.0f}%"
        print(f"{mode:10s} |   {m:.4f}   |        {red:>5s}         | {pcount[mode]}")
    print("\nsaved wan_cache/ablation_drift.png")


if __name__ == "__main__":
    main()
