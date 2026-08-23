"""
v2 eval: multi-seed closed-loop drift for baseline / v1 LoRA / v2 DAgger-only / v2 DAgger+contraction.
Same protocol as wan_multiseed.py (5 held-out clips × 3 seeds, K=48, sat drift + MUSIQ).
Answers: does closed-loop training (DAgger, contraction) improve on the one-step v1 teacher?
Saves wan_cache/v2_eval.npz + mean±std table.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import pyiqa
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters

DEVICE = "cuda"
K, NCTX, NHELD, EARLY, LATE = 48, 3, 5, 32, 32
SEEDS = [0, 1, 2]
OUT = "wan_cache"
CONFIGS = {"baseline": None, "v1": "lora_r_phi_k48.pt",
           "v2_dagger": "lora_r_phi_v2_dagger.pt", "v2_both": "lora_r_phi_v2_both.pt"}


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

    res = {m: {cn: [] for cn in CONFIGS} for m in ("sat", "late")}
    for cn, ckpt in CONFIGS.items():
        pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
        pipe.corrector = None
        model = pipe.generator.model
        if ckpt is not None:
            apply_lora(model, rank=16)
            sd = torch.load(os.path.join(OUT, ckpt), map_location="cpu")["lora"]
            for i, p in enumerate(lora_parameters(model)):
                p.data.copy_(sd[i].to(p.device, p.dtype))
            set_lora_scale(model, 1.0)
        for c in held:
            seed_lat = gt[c:c + 1, :NCTX].to(DEVICE).to(torch.bfloat16)
            for s in SEEDS:
                torch.manual_seed(1000 * s + c)
                noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
                _, lat = pipe.inference(noise=noise, text_prompts=[caps[c]], initial_latent=seed_lat, return_latents=True)
                pix = pipe.vae.decode_to_pixel(lat[:, :K])[0]
                sat = saturation(pix); ref = sat[P_CTX:P_CTX + 8].mean()
                mq = musiq_pf(musiq, pix)
                res["sat"][cn].append(float(np.abs(sat - ref)[P_CEIL:].mean()))
                res["late"][cn].append(float(mq[-LATE:].mean()))
        print(f"[{cn}] done", flush=True)
        del pipe; torch.cuda.empty_cache()

    np.savez(os.path.join(OUT, "v2_eval.npz"), **{f"{m}_{cn}": np.array(res[m][cn]) for m in res for cn in CONFIGS})
    b = np.mean(res["sat"]["baseline"])
    print(f"\n===== v2 vs v1 vs baseline ({NHELD} clips × {len(SEEDS)} seeds) =====")
    print(f"{'config':10s} | sat_drift (↓)       red% | MUSIQ_late (↑)")
    for cn in CONFIGS:
        sm, ss = np.mean(res["sat"][cn]), np.std(res["sat"][cn])
        lm, ls = np.mean(res["late"][cn]), np.std(res["late"][cn])
        r = "-" if cn == "baseline" else f"{100*(1-sm/(b+1e-9)):+.0f}%"
        print(f"{cn:10s} | {sm:.3f}±{ss:.3f}   {r:>5s} | {lm:.2f}±{ls:.2f}")
    print("\nsaved wan_cache/v2_eval.npz")


if __name__ == "__main__":
    main()
