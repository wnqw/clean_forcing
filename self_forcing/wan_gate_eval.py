"""
Drift-SNR gate ablation: v2 LoRA with alpha(t) gate vs flat alpha=1, same codepath
(flat = all-ones table through the same per-timestep lora-scale hook). 15-rollout protocol
(5 held-out clips x 3 seeds), sat drift + MUSIQ_late. Baseline row rerun for reference.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import pyiqa
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, lora_parameters

DEVICE = "cuda"
K, NCTX, NHELD, LATE = 48, 3, 5, 32
SEEDS = [0, 1, 2]
OUT = "wan_cache"
CKPT = "lora_r_phi_v2_both.pt"


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

    gate = torch.load(os.path.join(OUT, "gate_alpha.pt"), map_location="cpu")
    flat = {"timesteps": gate["timesteps"], "alpha": torch.ones_like(gate["alpha"])}
    print("alpha(t):", np.round(gate["alpha"].numpy(), 3).tolist(), flush=True)

    CONFIGS = {"v2_flat": flat, "v2_gated": gate}
    res = {m: {cn: [] for cn in CONFIGS} for m in ("sat", "late")}
    for cn, table in CONFIGS.items():
        pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
        pipe.corrector = None
        model = pipe.generator.model
        apply_lora(model, rank=16)
        sd = torch.load(os.path.join(OUT, CKPT), map_location="cpu")["lora"]
        for i, p in enumerate(lora_parameters(model)):
            p.data.copy_(sd[i].to(p.device, p.dtype))
        pipe.lora_gate = table
        for c in held:
            seed_lat = gt[c:c + 1, :NCTX].to(DEVICE).to(torch.bfloat16)
            for s in SEEDS:
                torch.manual_seed(1000 * s + c)
                noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
                _, lat = pipe.inference(noise=noise, text_prompts=[caps[c]], initial_latent=seed_lat, return_latents=True)
                pix = pipe.vae.decode_to_pixel(lat[:, :K])[0]
                sat = saturation(pix); ref = sat[P_CTX:P_CTX + 8].mean()
                res["sat"][cn].append(float(np.abs(sat - ref)[P_CEIL:].mean()))
                res["late"][cn].append(float(musiq_pf(musiq, pix)[-LATE:].mean()))
        print(f"[{cn}] done", flush=True)
        del pipe; torch.cuda.empty_cache()

    np.savez(os.path.join(OUT, "gate_eval.npz"), **{f"{m}_{cn}": np.array(res[m][cn]) for m in res for cn in CONFIGS})
    print(f"\n===== Drift-SNR gate ablation (v2, {NHELD} clips × {len(SEEDS)} seeds) =====")
    print("(reference: baseline sat 0.491±0.169, MUSIQ_late 29.6±5.6 from multiseed.npz)")
    for cn in CONFIGS:
        sm, ss = np.mean(res["sat"][cn]), np.std(res["sat"][cn])
        lm, ls = np.mean(res["late"][cn]), np.std(res["late"][cn])
        print(f"{cn:9s} | sat_drift {sm:.3f}±{ss:.3f} | MUSIQ_late {lm:.2f}±{ls:.2f}")


if __name__ == "__main__":
    main()
