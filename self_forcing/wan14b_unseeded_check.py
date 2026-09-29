"""Stage-2 KILL BAR check + deploy-select: un-seeded causal T2V rollouts on the 14B.

Arms: unadapted base + each wan14b_adapted_base_{2000,4000,6000}.pt. Per arm: NPROMPT un-seeded
K=21 rollouts (the trained window; the bar is "no collapse within the 21-latent window").
Automated collapse metrics per video (strips saved for visual inspection):
  sat_ramp  = mean sat(last 8 frames) - mean sat(first 8)   (unadapted 14B collapse: +0.28 mean)
  lowstd    = frac of frames with pixel std < 0.05           (flat-color collapse detector)
  musiq_late= mean MUSIQ over last 16 frames                 (quality; deploy-select tie-break)
PRE-STATED: an arm PASSES if mean |sat_ramp| < 0.10 AND lowstd == 0 on all its rollouts.
Deploy-select among passing adapted ckpts: highest musiq_late (paper precedent picked 4K).
Output: wan_cache/wan14b_unseeded.json + outputs_wan14b/unseeded_{arm}_p{i}_strip.png.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time
import numpy as np
import torch
from wan14b_common import (CausalDiffusionInferencePipeline14B, load_cfg, ref_prompt_pool,
                           save_strip, mean_saturation, write_timing)

DEVICE = "cuda"
K = 21
NPROMPT = int(os.environ.get("NPROMPT", 4))
ARMS = os.environ.get("ARMS", "unadapted,2000,4000,6000").split(",")
OUT_JSON = "wan_cache/wan14b_unseeded.json"
STRIPS = "outputs_wan14b"


@torch.no_grad()
def main():
    import pyiqa
    os.makedirs(STRIPS, exist_ok=True)
    cfg = load_cfg()
    torch.set_grad_enabled(False)
    musiq = pyiqa.create_metric("musiq", device=DEVICE)
    # 4 training-pool prompts (indices 100.. to avoid the ref-strip subset) — in-domain style check
    prompts = ref_prompt_pool(104)[100:100 + NPROMPT]

    results = {}
    t0 = time.time()
    for arm in ARMS:
        pipe = CausalDiffusionInferencePipeline14B(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
        pipe.corrector = None
        if arm != "unadapted":
            sd = torch.load(f"wan_cache/wan14b_adapted_base_{arm}.pt", map_location="cpu")["merged"]
            missing = pipe.generator.model.load_state_dict(
                {k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
            print(f"[{arm}] adapted base loaded ({len(sd)} tensors)", flush=True)
        rows = []
        for i, prompt in enumerate(prompts):
            torch.manual_seed(7000 + i)
            noise = torch.randn(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
            video, _ = pipe.inference(noise=noise, text_prompts=[prompt], return_latents=True)
            v = video[0]                                    # (81,3,H,W) in [0,1]
            sat = mean_saturation(v)
            ramp = float(sat[-8:].mean() - sat[:8].mean())
            std = v.float().std(dim=(1, 2, 3))
            lowstd = float((std < 0.05).float().mean())
            x = v.float().clamp(0, 1)
            mq = torch.cat([musiq(x[j:j + 16]).flatten() for j in range(0, x.shape[0], 16)])
            mlate = float(mq[-16:].mean())
            save_strip(v, os.path.join(STRIPS, f"unseeded_{arm}_p{i}_strip.png"), n=10)
            rows.append({"prompt_i": i, "sat_ramp": round(ramp, 4), "lowstd_frac": round(lowstd, 4),
                         "musiq_late": round(mlate, 2), "sat_first8": round(float(sat[:8].mean()), 4),
                         "sat_last8": round(float(sat[-8:].mean()), 4)})
            print(f"[{arm}] p{i}: sat_ramp {ramp:+.3f} lowstd {lowstd:.2f} musiq_late {mlate:.1f}", flush=True)
        mean_abs_ramp = float(np.mean([abs(r["sat_ramp"]) for r in rows]))
        any_lowstd = any(r["lowstd_frac"] > 0 for r in rows)
        passed = mean_abs_ramp < 0.10 and not any_lowstd
        results[arm] = {"rows": rows, "mean_abs_sat_ramp": round(mean_abs_ramp, 4),
                        "any_lowstd": any_lowstd,
                        "musiq_late_mean": round(float(np.mean([r["musiq_late"] for r in rows])), 2),
                        "pass": passed}
        print(f"[{arm}] mean|ramp| {mean_abs_ramp:.3f} | lowstd {any_lowstd} | "
              f"musiq_late {results[arm]['musiq_late_mean']:.1f} | {'PASS' if passed else 'FAIL'}", flush=True)
        del pipe
        torch.cuda.empty_cache()

    passing = [a for a in ARMS if a != "unadapted" and results[a]["pass"]]
    select = max(passing, key=lambda a: results[a]["musiq_late_mean"]) if passing else None
    results["_kill_bar"] = {"rule": "mean|sat_ramp|<0.10 AND no lowstd frames, any adapted ckpt",
                            "passing_ckpts": passing, "deploy_select": select,
                            "verdict": "PASS" if passing else "FAIL"}
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=1)
    write_timing("unseeded", time.time() - t0, len(ARMS) * NPROMPT)
    print(f"\nKILL BAR: {results['_kill_bar']['verdict']} | deploy-select: {select} | saved {OUT_JSON}")


if __name__ == "__main__":
    main()
