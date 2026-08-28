"""
Rank-sweep / diversity eval on the ADAPTED host — exact reconstruction of the paper's
in-domain 2x2 protocol (tab:2x2 adapted row, source wan_cache/adapt2x2_eval.npz):
  5 held-out Disney clips (35-39 of pairs_k48.pt) x 3 seeds (torch.manual_seed(1000*s+c)),
  K=48, NCTX=3 seed latents, sat_drift = mean|sat - ref(frames 9..16)| over pixel frames 81+,
  MUSIQ_late = mean MUSIQ over the last 32 pixel frames.
Also dumps 8 evenly-spaced extension-window pixel frames per rollout (uint8) for the
inter-seed diversity analysis (Task 2).

Env:
  CONFIGS  comma list of name=ckpt:rank entries; "base" = no corrector.
           e.g. CONFIGS="base,r16=lora_r_phi_k48_adapt.pt:16,v2=lora_r_phi_v2_both_adapt.pt:16"
  OUT_JSON results json (per-rollout metrics appended per config; resumable)
  FRAMES_DIR directory for the diversity frame dumps
Run from Self-Forcing: python -u wan_rank_sweep_eval.py
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time

import numpy as np
import pyiqa
import torch
from omegaconf import OmegaConf

from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters

DEVICE = "cuda"
K, NCTX, NHELD, LATE, W = 48, 3, 5, 32, 9
SEEDS = [0, 1, 2]
CACHE = "wan_cache"
ADAPTED = os.path.join(CACHE, "adapted_base_4000.pt")
OUT_JSON = os.environ["OUT_JSON"]
FRAMES_DIR = os.environ["FRAMES_DIR"]


def parse_configs():
    out = []
    for entry in os.environ["CONFIGS"].split(","):
        entry = entry.strip()
        if entry == "base":
            out.append(("base", None, 0))
        else:
            name, spec = entry.split("=")
            ckpt, rank = spec.rsplit(":", 1)
            out.append((name, ckpt, int(rank)))
    return out


def saturation(pix):
    x = ((pix.float() + 1) / 2).clamp(0, 1)
    cmax, cmin = x.amax(1), x.amin(1)
    return ((cmax - cmin) / (cmax + 1e-6)).mean(dim=(1, 2)).cpu().numpy()


@torch.no_grad()
def musiq_pf(metric, pix):
    x = ((pix.float() + 1) / 2).clamp(0, 1)
    return torch.cat([metric(x[i:i + 16]).flatten() for i in range(0, x.shape[0], 16)]).cpu().numpy()


def build_pipe(cfg, ckpt, rank):
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    sd = torch.load(ADAPTED, map_location="cpu")["merged"]
    pipe.generator.model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
    if ckpt is not None:
        model = pipe.generator.model
        apply_lora(model, rank=rank)
        lora_sd = torch.load(os.path.join(CACHE, ckpt), map_location="cpu")["lora"]
        params = lora_parameters(model)
        assert len(params) == len(lora_sd), f"{ckpt}: {len(lora_sd)} tensors vs {len(params)} lora params"
        for i, p in enumerate(params):
            assert p.shape == lora_sd[i].shape, f"{ckpt} rank mismatch at {i}: {lora_sd[i].shape} vs {p.shape}"
            p.data.copy_(lora_sd[i].to(p.device, p.dtype))
        set_lora_scale(model, 1.0)
    return pipe


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    os.makedirs(FRAMES_DIR, exist_ok=True)
    musiq = pyiqa.create_metric("musiq", device=DEVICE)
    d = torch.load(os.path.join(CACHE, "pairs_k48.pt"), map_location="cpu")
    gt, caps = d["gt"], d["captions"]
    held = list(range(gt.shape[0] - NHELD, gt.shape[0]))
    P_CTX, P_CEIL = (NCTX - 1) * 4 + 1, (21 - 1) * 4 + 1

    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    for cn, ckpt, rank in parse_configs():
        if cn in results:
            print(f"[{cn}] already in {OUT_JSON}, skipping", flush=True)
            continue
        t0 = time.time()
        pipe = build_pipe(cfg, ckpt, rank)
        sat_l, late_l, rollouts = [], [], []
        frames = {}
        for c in held:
            seed_lat = gt[c:c + 1, :NCTX].to(DEVICE).to(torch.bfloat16)
            for s in SEEDS:
                torch.manual_seed(1000 * s + c)
                noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
                _, lat = pipe.inference(noise=noise, text_prompts=[caps[c]],
                                        initial_latent=seed_lat, return_latents=True)
                pix = pipe.vae.decode_to_pixel(lat[:, :K])[0]
                sat = saturation(pix); ref = sat[P_CTX:P_CTX + 8].mean()
                mq = musiq_pf(musiq, pix)
                sat_l.append(float(np.abs(sat - ref)[P_CEIL:].mean()))
                late_l.append(float(mq[-LATE:].mean()))
                rollouts.append({"clip": c, "seed": s})
                idx = np.linspace(P_CTX, pix.shape[0] - 1, 8).round().astype(int)
                fr = ((pix[idx].float() + 1) / 2).clamp(0, 1).mul(255).byte().cpu()
                frames[f"c{c}_s{s}"] = fr
                print(f"[{cn}] clip {c} seed {s} | sat {sat_l[-1]:.3f} late {late_l[-1]:.1f}", flush=True)
        torch.save(frames, os.path.join(FRAMES_DIR, f"{cn}_frames.pt"))
        sat_a, late_a = np.array(sat_l), np.array(late_l)
        results[cn] = {"ckpt": ckpt, "rank": rank, "rollouts": rollouts,
                       "sat": sat_l, "late": late_l,
                       "sat_mean": float(sat_a.mean()), "sat_std": float(sat_a.std()),
                       "late_mean": float(late_a.mean()), "late_std": float(late_a.std()),
                       "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(results, open(OUT_JSON, "w"), indent=1)
        print(f"RESULT [{cn}] sat {sat_a.mean():.3f}±{sat_a.std():.3f} | "
              f"late {late_a.mean():.1f}±{late_a.std():.1f} | {(time.time()-t0)/60:.0f} min", flush=True)
        del pipe
        torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
