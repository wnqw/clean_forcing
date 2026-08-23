"""
Multi-seed hardening: mean±std over (held-out clips × seeds) for every config, both metrics.
Configs: baseline / ours(lora) / dmd / full / residual.  Metrics per K=48 rollout:
  - sat_drift  = mean|Δsat from start| over pixel frames beyond the old ceiling (81-189)
  - vbench_dq  = MUSIQ ΔDriftQuality (mean early - mean late; >0 = quality decayed)
  - musiq_late = absolute MUSIQ over the last window (higher=better)
Held-out clips = the 5 the correctors never trained on (train was clips 0..34). Fresh base per config.
Saves wan_cache/multiseed.npz + prints the mean±std table.
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
from wan.modules.corrector import WanVelocityResidual

DEVICE = "cuda"
K, NCTX, NHELD, EARLY, LATE, W = 48, 3, 5, 32, 32, 9
SEEDS = [0, 1, 2]
OUT = "wan_cache"
FULL_TARGETS = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o")
CONFIGS = ["baseline", "ours", "dmd", "full", "residual"]
CKPTS = {"ours": "lora_r_phi_k48.pt", "dmd": "corr_dmd_k48.pt",
         "full": "corr_full_k48.pt", "residual": "corr_residual_k48.pt"}


def saturation(pix):
    x = ((pix.float() + 1) / 2).clamp(0, 1)
    cmax, cmin = x.amax(1), x.amin(1)
    return ((cmax - cmin) / (cmax + 1e-6)).mean(dim=(1, 2)).cpu().numpy()


@torch.no_grad()
def musiq_pf(metric, pix):
    x = ((pix.float() + 1) / 2).clamp(0, 1)
    return torch.cat([metric(x[i:i + 16]).flatten() for i in range(0, x.shape[0], 16)]).cpu().numpy()


def configure(pipe, cfg_name):
    model = pipe.generator.model
    if cfg_name in ("ours", "dmd"):
        apply_lora(model, rank=16)
        sd = torch.load(os.path.join(OUT, CKPTS[cfg_name]), map_location="cpu")["lora"]
        for i, p in enumerate(lora_parameters(model)):
            p.data.copy_(sd[i].to(p.device, p.dtype))
        set_lora_scale(model, 1.0)
        return None
    if cfg_name == "full":
        sd = torch.load(os.path.join(OUT, CKPTS["full"]), map_location="cpu")["full"]
        named = dict(model.named_parameters())
        for n, t in sd.items():
            named[n].data.copy_(t.to(named[n].device, named[n].dtype))
        return None
    if cfg_name == "residual":
        r = WanVelocityResidual().to(DEVICE)
        r.load_state_dict(torch.load(os.path.join(OUT, CKPTS["residual"]), map_location="cpu")["residual"])
        pipe.attach_corrector(r, alpha=1.0); pipe.corrector_hist = W
        return r
    return None


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    musiq = pyiqa.create_metric("musiq", device=DEVICE)
    d = torch.load(os.path.join(OUT, "pairs_k48.pt"), map_location="cpu")
    gt, caps = d["gt"], d["captions"]
    held = list(range(gt.shape[0] - NHELD, gt.shape[0]))
    P_CTX, P_CEIL = (NCTX - 1) * 4 + 1, (21 - 1) * 4 + 1

    res = {m: {cn: [] for cn in CONFIGS} for m in ("sat", "dq", "late")}
    for cn in CONFIGS:
        pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
        pipe.corrector = None
        configure(pipe, cn)
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
                res["dq"][cn].append(float(mq[P_CTX:P_CTX + EARLY].mean() - mq[-LATE:].mean()))
                res["late"][cn].append(float(mq[-LATE:].mean()))
        print(f"[{cn}] done ({len(res['sat'][cn])} rollouts)", flush=True)
        del pipe; torch.cuda.empty_cache()

    np.savez(os.path.join(OUT, "multiseed.npz"), **{f"{m}_{cn}": np.array(res[m][cn]) for m in res for cn in CONFIGS})

    def ms(a):
        a = np.array(a); return a.mean(), a.std()
    b_sat, _ = ms(res["sat"]["baseline"]); b_dq, _ = ms(res["dq"]["baseline"])
    print(f"\n===== multi-seed mean±std ({NHELD} held-out clips × {len(SEEDS)} seeds = {NHELD*len(SEEDS)} rollouts/config) =====")
    print(f"{'config':10s} | sat_drift (↓)      red% | ΔDriftQuality (↓)   red% | MUSIQ_late (↑)")
    for cn in CONFIGS:
        sm, ss = ms(res["sat"][cn]); dm, ds = ms(res["dq"][cn]); lm, ls = ms(res["late"][cn])
        sr = "-" if cn == "baseline" else f"{100*(1-sm/(b_sat+1e-9)):+.0f}%"
        dr = "-" if cn == "baseline" else f"{100*(1-dm/(b_dq+1e-9)):+.0f}%"
        print(f"{cn:10s} | {sm:.3f}±{ss:.3f}  {sr:>5s} | {dm:+.2f}±{ds:.2f}  {dr:>5s} | {lm:.2f}±{ls:.2f}")
    print("\nsaved wan_cache/multiseed.npz")


if __name__ == "__main__":
    main()
