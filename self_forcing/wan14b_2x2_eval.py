"""Stage 5 of the 14B scale-up: in-domain 2x2 — mirror of the paper's Table-2 protocol
(wan_v2_eval.py): 5 held-out clips x 3 seeds, paired (same clip context + noise seeds across
arms), K=48 seeded rollouts, sat-drift (mean |sat - ref| past the 21-latent ceiling, ref =
first 8 post-context frames) + MUSIQ_late (last 32 frames).

Arms = {unadapted, adapted} x {none, v1, v2} on the 14B. Held-out = last 5 clips of the pairs
pool (inside the val split, never trained on). KILL BAR (pre-stated): best corrector on the
ADAPTED host cuts sat-drift >= 40% vs adapted-no-corrector.
Output: wan_cache/wan14b_2x2.json + .npz (per-rollout arrays). Resumable per arm.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time
import numpy as np
import torch
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters
from wan14b_common import CausalDiffusionInferencePipeline14B, load_cfg, write_timing

DEVICE = "cuda"
K, NCTX, NHELD, LATE = 48, 3, 5, 32
SEEDS = [0, 1, 2]
OUT = "wan_cache"
PAIRS = os.environ.get("PAIRS", "wan14b_pairs_synth.pt")
ADAPTED = os.environ.get("ADAPTED_BASE")            # deploy-selected merged ckpt
V1, V2 = "wan14b_lora_v1.pt", os.environ.get("V2_NAME", "wan14b_lora_v2.pt")
ARMS = [("unadapted", None, None), ("unadapted", None, V1), ("unadapted", None, V2),
        ("adapted", "AB", None), ("adapted", "AB", V1), ("adapted", "AB", V2)]
OUT_JSON = os.path.join(OUT, "wan14b_2x2.json")
BAR_CUT = 0.40


def saturation(pix01):
    cmax, cmin = pix01.amax(1), pix01.amin(1)
    return ((cmax - cmin) / (cmax + 1e-6)).mean(dim=(1, 2)).cpu().numpy()


@torch.no_grad()
def main():
    import pyiqa
    assert ADAPTED, "set ADAPTED_BASE"
    cfg = load_cfg()
    torch.set_grad_enabled(False)
    musiq = pyiqa.create_metric("musiq", device=DEVICE)
    d = torch.load(os.path.join(OUT, PAIRS), map_location="cpu")
    gt, caps = d["gt"], d["captions"]
    held = list(range(gt.shape[0] - NHELD, gt.shape[0]))
    P_CTX, P_CEIL = (NCTX - 1) * 4 + 1, (21 - 1) * 4 + 1

    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    t0 = time.time()
    for base, ab, cor in ARMS:
        name = f"{base}+{'none' if cor is None else cor.split('_')[-1].replace('.pt', '')}"
        if name in res and len(res[name]["sat"]) == NHELD * len(SEEDS):
            print(f"[{name}] cached, skip", flush=True)
            continue
        pipe = CausalDiffusionInferencePipeline14B(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
        pipe.corrector = None
        model = pipe.generator.model
        if ab == "AB":
            sd = torch.load(ADAPTED, map_location="cpu")["merged"]
            model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
        if cor is not None:
            apply_lora(model, rank=16)
            sd = torch.load(os.path.join(OUT, cor), map_location="cpu")["lora"]
            sd = list(sd.values()) if isinstance(sd, dict) else sd
            for i, p in enumerate(lora_parameters(model)):
                p.data.copy_(sd[i].to(p.device, p.dtype))
            set_lora_scale(model, 1.0)
        sat_l, late_l = [], []
        for c in held:
            seed_lat = gt[c:c + 1, :NCTX].to(DEVICE).to(torch.bfloat16)
            for s in SEEDS:
                torch.manual_seed(1000 * s + c)
                noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
                video, lat = pipe.inference(noise=noise, text_prompts=[caps[c]],
                                            initial_latent=seed_lat, return_latents=True)
                pix01 = video[0].float().clamp(0, 1)                       # (189,3,H,W)
                sat = saturation(pix01)
                ref = sat[P_CTX:P_CTX + 8].mean()
                mq = torch.cat([musiq(pix01[i:i + 16]).flatten()
                                for i in range(0, pix01.shape[0], 16)]).cpu().numpy()
                sat_l.append(float(np.abs(sat - ref)[P_CEIL:].mean()))
                late_l.append(float(mq[-LATE:].mean()))
        res[name] = {"sat": sat_l, "late": late_l,
                     "sat_mean": float(np.mean(sat_l)), "sat_std": float(np.std(sat_l)),
                     "late_mean": float(np.mean(late_l)), "late_std": float(np.std(late_l))}
        json.dump(res, open(OUT_JSON, "w"), indent=1)
        print(f"[{name}] sat {res[name]['sat_mean']:.3f}±{res[name]['sat_std']:.3f} "
              f"| MUSIQ_late {res[name]['late_mean']:.1f}±{res[name]['late_std']:.1f}", flush=True)
        del pipe
        torch.cuda.empty_cache()

    b = res["adapted+none"]["sat_mean"]
    cuts = {n: 1 - res[n]["sat_mean"] / (b + 1e-9) for n in res if n.startswith("adapted+") and n != "adapted+none"}
    best = max(cuts, key=cuts.get)
    verdict = "PASS" if cuts[best] >= BAR_CUT else "FAIL"
    res["_kill_bar"] = {"rule": f"best corrector cut on adapted host >= {BAR_CUT:.0%}",
                        "cuts": {n: round(c, 4) for n, c in cuts.items()},
                        "best": best, "best_cut": round(cuts[best], 4), "verdict": verdict}
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    np.savez(os.path.join(OUT, "wan14b_2x2.npz"),
             **{f"{m}_{n}": np.array(res[n][m]) for n in res if not n.startswith("_") for m in ("sat", "late")})
    write_timing("2x2", time.time() - t0, len(ARMS) * NHELD * len(SEEDS))
    print(f"\n===== 14B in-domain 2x2 ({NHELD} clips x {len(SEEDS)} seeds) =====")
    for n in [a for a in res if not a.startswith("_")]:
        r = res[n]
        print(f"{n:16s} | sat {r['sat_mean']:.3f}±{r['sat_std']:.3f} | late {r['late_mean']:.1f}±{r['late_std']:.1f}")
    print(f"KILL BAR: {verdict} (best cut {cuts[best]:+.0%} by {best})")


if __name__ == "__main__":
    main()
