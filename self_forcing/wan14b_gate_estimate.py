"""Stage 3 of the 14B viability gate: Drift-SNR gate on the UNADAPTED Wan2.1-T2V-14B.

REPLICATES wan_gate_estimate.py's estimator (same velocity_tf teacher-forcing probe, same
finite-sample alpha*), on the zero-real pairs from stages 1-2:
  Delta_m = v_theta(z_t^m, h_clean) - v_theta(z_t^m, h_gen)  over M=4 noise seeds at fixed
  (clip,k,t); z_t^m = (1-sigma) x0 + sigma eps_m with x0 = gen[k:k+3] (drifted-rollout state,
  matched across branches -- only history differs).
  alpha*_hat = max(0, ||bias||^2 - var/M) / ||bias||^2          (systematic fraction)
  rel_v      = ||Delta|| / ||v_theta(z_t, h_clean)|| per seed   (relative drift-gap size;
               the /||v_gen|| variant of wan_r_target_faithful.py is recorded alongside)

Grid: all 20 solver timesteps (shift 8) x depths k in {9,12,15,18} x NSAMP=2 clips x M=4 seeds
(clean refs are 21 latents, so k <= 21 - nfb - like the 1.3B synthetic/zero-real k-range).

PRE-STATED aggregation (fixed before results): "pooled high-t alpha*" = mean over grid samples
with t >= 900 (~top half of the shift-8 schedule; the 1.3B drift gap concentrates there);
t >= 800 also recorded. rel_v bar checked on the full-grid pooled mean (all t, all k).
Bars: alpha*_high >= 0.8 AND rel_v >= 0.15 -> PASS; alpha* in [0.4,0.8) or rel_v in
[0.05,0.15) -> MARGINAL; below -> FAIL.

Output: wan_cache/wan14b_gate.pt (per-t table, mirrors gate_alpha.pt) +
wan_cache/wan14b_gate.json (all records + aggregates + verdict). Run from repo root.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time
import numpy as np
import torch
from wan14b_common import CausalDiffusionInferencePipeline14B, load_cfg, write_timing

DEVICE = "cuda"
W = 9                       # history window fed to the velocity probe (as wan_gate_estimate.py)
M = 4                       # noise seeds per (clip,k,t)
NSAMP = 2                   # clips sampled per (t,k) cell
KS = [9, 12, 15, 18]        # rollout depths (k+3 <= 21 = clean-ref horizon)
PAIRS = "wan_cache/wan14b_pairs.pt"
OUT_PT = "wan_cache/wan14b_gate.pt"
OUT_JSON = "wan_cache/wan14b_gate.json"

BAR = {"alpha_pass": 0.8, "alpha_marginal": 0.4, "rel_pass": 0.15, "rel_marginal": 0.05,
       "high_t": 900}


def velocity_tf(pipe, cond, history, x0_cur, z_t, t):
    """Verbatim from wan_gate_estimate.py: teacher-forcing velocity of the current chunk."""
    Wh, nf = history.shape[1], z_t.shape[1]
    clean_x = torch.cat([history, x0_cur], 1)
    noisy = torch.cat([history, z_t], 1)
    ts = torch.cat([torch.zeros((1, Wh), device=z_t.device, dtype=torch.float32),
                    torch.full((1, nf), float(t), device=z_t.device, dtype=torch.float32)], 1)
    flow, _ = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts, clean_x=clean_x)
    return flow[:, Wh:]


def band(x, lo, hi):
    return 2 if x >= hi else (1 if x >= lo else 0)


@torch.no_grad()
def main():
    cfg = load_cfg()
    torch.set_grad_enabled(False)
    t_load = time.time()
    pipe = CausalDiffusionInferencePipeline14B(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    load_s = time.time() - t_load
    print(f"14B pipeline up in {load_s:.0f}s", flush=True)

    d = torch.load(PAIRS, map_location="cpu")
    gt, gen, caps = d["gt"], d["gen"], d["captions"]
    N = gt.shape[0]
    nfb = pipe.num_frame_per_block
    assert max(KS) + nfb <= gt.shape[1], "depth exceeds clean-ref horizon"
    sched = pipe._initialize_sample_scheduler(torch.zeros(1, 48, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
    sigmas, tsteps = sched.sigmas.to(DEVICE).float(), sched.timesteps.to(DEVICE).float()
    rng = np.random.default_rng(0)
    cond_cache = {}
    records = []

    t0 = time.time()
    for ti in range(len(tsteps)):
        t, sig = tsteps[ti], sigmas[ti]
        for k in KS:
            for _ in range(NSAMP):
                c = int(rng.choice(N))
                if c not in cond_cache:
                    cond_cache[c] = pipe.text_encoder(text_prompts=[caps[c]])
                gtc = gt[c:c + 1].to(DEVICE).to(torch.bfloat16)
                genc = gen[c:c + 1].to(DEVICE).to(torch.bfloat16)
                gt_h, gen_h, x0 = gtc[:, k - W:k], genc[:, k - W:k], genc[:, k:k + nfb]
                deltas, rel_c, rel_g = [], [], []
                for m in range(M):
                    eps = torch.randn(x0.shape, device=DEVICE, dtype=x0.dtype)
                    z_t = (1 - sig) * x0 + sig * eps
                    v_clean = velocity_tf(pipe, cond_cache[c], gt_h, x0, z_t, t)
                    v_gen = velocity_tf(pipe, cond_cache[c], gen_h, x0, z_t, t)
                    dv = (v_clean - v_gen).float()
                    deltas.append(dv)
                    rel_c.append((dv.norm() / (v_clean.float().norm() + 1e-8)).item())
                    rel_g.append((dv.norm() / (v_gen.float().norm() + 1e-8)).item())
                D = torch.stack(deltas)
                bias = D.mean(0)
                var = (D - bias).pow(2).sum(dim=tuple(range(1, D.ndim))).mean().item()
                b2 = bias.pow(2).sum().item()
                alpha = max(0.0, b2 - var / M) / (b2 + 1e-12)
                records.append({"t": float(t.item()), "sigma": float(sig.item()), "k": k, "clip": c,
                                "alpha": alpha, "bias2": b2, "var": var,
                                "rel_v_clean": rel_c, "rel_v_gen": rel_g})
        rt = [r for r in records if r["t"] == float(t.item())]
        print(f"t={t.item():7.1f} (sigma {sig.item():.3f}) | alpha* = "
              f"{np.mean([r['alpha'] for r in rt]):.3f} | rel_v = "
              f"{np.mean([np.mean(r['rel_v_clean']) for r in rt]):.3f}", flush=True)
    wall_s = time.time() - t0

    def agg(sel, key):
        rs = [r for r in records if sel(r)]
        if key == "alpha":
            return float(np.mean([r["alpha"] for r in rs]))
        return float(np.mean([np.mean(r[key]) for r in rs]))

    tvals = [float(t.item()) for t in tsteps]
    alpha_t = [agg(lambda r, tv=tv: r["t"] == tv, "alpha") for tv in tvals]
    rel_t = [agg(lambda r, tv=tv: r["t"] == tv, "rel_v_clean") for tv in tvals]
    alpha_k = {k: agg(lambda r, k=k: r["k"] == k, "alpha") for k in KS}
    rel_k = {k: agg(lambda r, k=k: r["k"] == k, "rel_v_clean") for k in KS}
    pooled = {
        "alpha_pooled": agg(lambda r: True, "alpha"),
        "alpha_pooled_high_t900": agg(lambda r: r["t"] >= 900, "alpha"),
        "alpha_pooled_high_t800": agg(lambda r: r["t"] >= 800, "alpha"),
        "rel_v_pooled": agg(lambda r: True, "rel_v_clean"),
        "rel_v_pooled_high_t900": agg(lambda r: r["t"] >= 900, "rel_v_clean"),
        "rel_v_gen_pooled": agg(lambda r: True, "rel_v_gen"),
    }
    verdict_code = min(band(pooled["alpha_pooled_high_t900"], BAR["alpha_marginal"], BAR["alpha_pass"]),
                       band(pooled["rel_v_pooled"], BAR["rel_marginal"], BAR["rel_pass"]))
    verdict = {2: "PASS", 1: "MARGINAL", 0: "FAIL"}[verdict_code]

    torch.save({"timesteps": tsteps.cpu(), "alpha": torch.tensor(alpha_t)}, OUT_PT)
    out = {"protocol": {"model": "Wan2.1-T2V-14B unadapted, block-causal", "W": W, "M": M,
                        "NSAMP": NSAMP, "KS": KS, "n_clips": N, "solver_steps": len(tvals),
                        "shift": float(cfg.timestep_shift), "bars": BAR},
           "timesteps": tvals, "alpha_per_t": alpha_t, "rel_v_per_t": rel_t,
           "alpha_per_k": alpha_k, "rel_v_per_k": rel_k, **pooled,
           "verdict": verdict, "wall_s": round(wall_s, 1),
           "model_load_s": round(load_s, 1), "records": records}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    write_timing("gate", wall_s, len(records), {"model_load_s": round(load_s, 1),
                                                "tf_calls": len(records) * M * 2})

    print(f"\nalpha*(t): {np.round(alpha_t, 3).tolist()}")
    print(f"rel_v(t):  {np.round(rel_t, 3).tolist()}")
    print(f"alpha*(k): { {k: round(v, 3) for k, v in alpha_k.items()} }")
    print(f"rel_v(k):  { {k: round(v, 3) for k, v in rel_k.items()} }")
    print({k: round(v, 4) for k, v in pooled.items()})
    print(f"VERDICT: {verdict} (bars: alpha_high>= {BAR['alpha_pass']} & rel_v >= {BAR['rel_pass']} -> PASS)")
    print(f"saved {OUT_PT} + {OUT_JSON}")


if __name__ == "__main__":
    main()
