"""Reliability-shrunken gate analysis (retrospective, no rollouts).

Hypothesis: the deploy gain the sweeps converged to is predicted by
    gain(t) = alpha*(t) * rho(t)
where alpha*(t) = ||bias||^2/(||bias||^2+var) is the stored Drift-SNR gate
(wan_cache/gate_alpha.pt) and rho(t) is the per-solver-timestep validation R^2
of the deployed corrector LoRA (fraction of the drift gap it closes).

rho(t) is measured with EXACTLY the wan_train_v2.py validation quantity
(dagger loss on held-out clips, drift-gap normalization, native anchoring),
just accumulated per timestep bin instead of pooled:
    rho(t) = 1 - sum ||v_corr - v_clean||^2 / sum ||v_clean - v_base||^2

Usage (from Self-Forcing/, conda env df-gb300, via safe_run.sh):
  LORA=wan_cache/lora_r_phi_v2s_adapt.pt POOLS=pairs_synth_adapt.pt,pairs_synth_dagger_adapt.pt \
  K=21 TAG=av2s SPT=8 python -u gate_reliability.py

Writes JSON results to OUT_JSON (default: alongside this script's scratch dir).
Read-only w.r.t. wan_cache: loads pairs/LoRA/gate, writes nothing there.
"""
import json
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys

SF = "./self_forcing"
sys.path.insert(0, SF)
os.chdir(SF)

import numpy as np
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters

DEVICE = "cuda"
W = 9
K = int(os.environ.get("K", 48))
SPT = int(os.environ.get("SPT", 8))          # samples per timestep bin
TAG = os.environ.get("TAG", "corr")
LORA = os.environ["LORA"]
POOLS = os.environ["POOLS"].split(",")
ADAPTED = os.environ.get("ADAPTED_BASE", "wan_cache/adapted_base_4000.pt")
OUT_JSON = os.environ.get("OUT_JSON", f"/tmp/gate_reliability_{TAG}.json")
CACHE = "wan_cache"


def velocity_tf(pipe, cond, history, x0_cur, z_t, t):  # identical to wan_train_v2.velocity_tf
    Wh, nf = history.shape[1], z_t.shape[1]
    clean_x = torch.cat([history, x0_cur], 1)
    noisy = torch.cat([history, z_t], 1)
    ts = torch.cat([torch.zeros((1, Wh), device=z_t.device, dtype=torch.float32),
                    torch.full((1, nf), float(t), device=z_t.device, dtype=torch.float32)], 1)
    flow, _ = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts, clean_x=clean_x)
    return flow[:, Wh:]


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    sd = torch.load(ADAPTED, map_location="cpu")["merged"]
    pipe.generator.model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
    model = pipe.generator.model
    apply_lora(model, rank=16)
    lw = torch.load(LORA, map_location="cpu")["lora"]
    lw = list(lw.values()) if isinstance(lw, dict) else lw
    for p, w in zip(lora_parameters(model), lw):
        p.data.copy_(w.to(p.device, p.dtype))
    print(f"[{TAG}] LoRA {LORA} on {ADAPTED} | pools {POOLS} | K={K} SPT={SPT}", flush=True)

    pools = [torch.load(os.path.join(CACHE, p), map_location="cpu") for p in POOLS]
    caps = pools[0]["captions"]
    N = pools[0]["gt"].shape[0]
    nval = max(4, N // 8)                                 # wan_train_v2 split
    val_ids = list(range(N - nval, N))
    nfb = pipe.num_frame_per_block
    ks = list(range(W, K - 2 * nfb + 1))                  # wan_train_v2 k-range
    sched = pipe._initialize_sample_scheduler(torch.zeros(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
    sigmas, tsteps = sched.sigmas.to(DEVICE).float(), sched.timesteps.to(DEVICE).float()
    rng = np.random.default_rng(0)
    cond_cache = {}

    nT = len(tsteps)
    num = np.zeros(nT)
    den = np.zeros(nT)
    for ti in range(nT):
        t, sig = tsteps[ti], sigmas[ti]
        for _ in range(SPT):
            pool = pools[int(rng.integers(len(pools)))]
            c = int(rng.choice(val_ids))
            if c not in cond_cache:
                cond_cache[c] = pipe.text_encoder(text_prompts=[caps[c]])
            k = int(rng.choice(ks))
            gtc = pool["gt"][c:c + 1].to(DEVICE).to(torch.bfloat16)
            genc = pool["gen"][c:c + 1].to(DEVICE).to(torch.bfloat16)
            gt_h, gen_h, x0 = gtc[:, k - W:k], genc[:, k - W:k], genc[:, k:k + nfb]
            eps = torch.randn(x0.shape, device=DEVICE, dtype=x0.dtype)
            z_t = (1 - sig) * x0 + sig * eps
            set_lora_scale(model, 0.0)
            v_clean = velocity_tf(pipe, cond_cache[c], gt_h, x0, z_t, t)
            v_base = velocity_tf(pipe, cond_cache[c], gen_h, x0, z_t, t)
            set_lora_scale(model, 1.0)
            v_corr = velocity_tf(pipe, cond_cache[c], gen_h, x0, z_t, t)
            num[ti] += (v_corr - v_clean).float().pow(2).sum().item()
            den[ti] += (v_clean - v_base).float().pow(2).sum().item()
        rho_ti = 1 - num[ti] / (den[ti] + 1e-12)
        print(f"[{TAG}] t={t.item():7.1f} (sigma {sig.item():.3f}) | rho = {rho_ti:+.3f}", flush=True)

    rho = 1 - num / (den + 1e-12)
    rho_overall = 1 - num.sum() / (den.sum() + 1e-12)
    gate = torch.load(os.path.join(CACHE, "gate_alpha.pt"), map_location="cpu")
    g_ts, alpha = gate["timesteps"].numpy(), gate["alpha"].numpy()
    # gate table was estimated on the K=48 schedule; align by timestep value (same 20-step
    # schedule for K=21 vs K=48 unless shift differs -> nearest-neighbor match, report error)
    ts_np = tsteps.cpu().numpy()
    idx = np.array([int(np.argmin(np.abs(g_ts - t))) for t in ts_np])
    align_err = float(np.max(np.abs(g_ts[idx] - ts_np)))
    gain = alpha[idx] * rho

    print(f"\n[{TAG}] overall val R^2 (v2 normalization) = {rho_overall:+.4f}")
    print(f"[{TAG}] gate align max |dt| = {align_err:.2f}")
    print(f"[{TAG}] {'t':>7} {'sigma':>6} {'alpha*':>7} {'rho':>7} {'gain=a*rho':>10}")
    for ti in range(nT):
        print(f"[{TAG}] {ts_np[ti]:7.1f} {sigmas[ti].item():6.3f} {alpha[idx[ti]]:7.3f} {rho[ti]:+7.3f} {gain[ti]:10.3f}")
    w = den / den.sum()                                   # drift-gap-energy weighting
    print(f"[{TAG}] mean gain (unweighted) = {float(gain.mean()):.3f} | "
          f"gap-energy-weighted = {float((gain * w).sum()):.3f}")

    out = {"tag": TAG, "lora": LORA, "pools": POOLS, "K": K, "spt": SPT,
           "timesteps": ts_np.tolist(), "sigmas": sigmas.cpu().numpy().tolist(),
           "alpha": alpha[idx].tolist(), "rho": rho.tolist(), "gain": gain.tolist(),
           "num": num.tolist(), "den": den.tolist(),
           "rho_overall": float(rho_overall), "gain_mean": float(gain.mean()),
           "gain_gap_weighted": float((gain * w).sum()), "gate_align_max_dt": align_err}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[{TAG}] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
