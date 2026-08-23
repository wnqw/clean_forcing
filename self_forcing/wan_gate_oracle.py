"""
v3-b oracle-teacher gate (stage 0, no training).
Question: does a future-informed target — the same checkpoint run BIDIRECTIONALLY over
[clean past, z_t chunk, clean future] — carry learnable signal beyond the past-only teacher?

Formulation (in-distribution for the bidi model): noise the clean context frames to the SAME
uniform t as the chunk and read the velocity at the chunk frames. Vanilla WanModel supports only
a scalar timestep, so the mixed-t variant (t=0 context) is not expressible; uniform-t is exactly
the bidi training regime and is how bidi teachers are used in distillation.

Per state (c, k, t), per seed s (fresh z_t noise AND fresh context noise — the corrector can see
z_t but never the context noise, so oracle variance from it is correctly charged):
  v_gen^s    = causal TF velocity, gen history        (what the host does)
  v_clean^s  = causal TF velocity, clean history      (past-only teacher = current target)
  v_orA^s    = bidi velocity, ADAPTED merged weights  (oracle A)
  v_orP^s    = bidi velocity, PRISTINE base weights   (oracle B)

GATE metrics (per state, then aggregated):
  alpha_past   = ||mean_s(v_clean - v_gen)||^2 / mean_s ||v_clean - v_gen||^2   (reference)
  alpha_orA/P  = same with v_or - v_gen              (systematic fraction of oracle gap)
  info_A/P     = ||mean_s v_or - mean_s v_clean|| / ||mean_s(v_clean - v_gen)||
                 (~0: oracle adds nothing; ~0.3-1: new signal; >>1: unstable/off-scale)
Run: cd Self-Forcing && ADAPTED_BASE=wan_cache/adapted_base_4000.pt PAIRS=pairs_synth_adapt.pt \
     python -u wan_gate_oracle.py
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:native")
import numpy as np
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper

DEVICE = "cuda"
K, W = 21, 9
NSTATES = int(os.environ.get("NSTATES", 24))
NSEEDS = int(os.environ.get("NSEEDS", 4))
PAIRS = os.environ.get("PAIRS", "pairs_synth_adapt.pt")
ADAPTED = os.environ.get("ADAPTED_BASE", "wan_cache/adapted_base_4000.pt")
OUT = os.environ.get("OUT", "wan_cache/gate_oracle_v3b.npz")


def velocity_tf(pipe, cond, history, x0_cur, z_t, t):  # causal teacher-forcing (as in wan_train_synth)
    Wh, nf = history.shape[1], z_t.shape[1]
    clean_x = torch.cat([history, x0_cur], 1)
    noisy = torch.cat([history, z_t], 1)
    ts = torch.cat([torch.zeros((1, Wh), device=z_t.device, dtype=torch.float32),
                    torch.full((1, nf), float(t), device=z_t.device, dtype=torch.float32)], 1)
    flow, _ = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts, clean_x=clean_x)
    return flow[:, Wh:]


@torch.no_grad()
def predict_future(pipe, cond, hist, nfb, sigmas, tsteps, n_lat, n_steps=8, seed=0):
    """v3-c continuation: AR Euler sampling of n_lat latents from clean history via the
    teacher-forced causal path. Every call keeps the trainer's 12-frame window (the causal
    model caches its TF block mask for one shape). x0_cur slot is masked from the chunk's
    own prediction, so zeros are safe there."""
    g = torch.Generator(device=hist.device).manual_seed(seed)
    ids = np.linspace(0, len(tsteps) - 1, n_steps).round().astype(int)
    h, out = hist[:, -9:], []
    for _ in range(0, n_lat, nfb):
        z = torch.randn(1, nfb, 16, 60, 104, device=hist.device, dtype=torch.float32, generator=g)
        for a, idx in enumerate(ids):
            v = velocity_tf(pipe, cond, h, torch.zeros_like(z).to(h.dtype), z.to(h.dtype), tsteps[idx]).float()
            sig_next = float(sigmas[ids[a + 1]]) if a + 1 < len(ids) else 0.0
            z = z + (sig_next - float(sigmas[idx])) * v
        out.append(z.to(h.dtype))
        h = torch.cat([h, out[-1]], 1)[:, -9:]
    return torch.cat(out, 1)[:, :n_lat]


def velocity_bidi(bidi, cond, gtc, z_t, k, nfb, t, sig, gen_ctx, past=None, future=None):
    """Bidi velocity at the chunk slot. Context = gtc (clean) everywhere, except the
    past is `past` when given (bridge teacher: drifted past + clean future) and the
    frames after the chunk are `future` when given (v3-c: predicted future)."""
    ctx_noise = torch.randn(gtc.shape, device=gtc.device, dtype=gtc.dtype, generator=gen_ctx)
    ctx = gtc if past is None else torch.cat([past, gtc[:, k:]], 1)
    if future is not None:
        ctx = torch.cat([ctx[:, :k + nfb], future], 1)
    z_full = (1 - sig) * ctx + sig * ctx_noise
    z_full = torch.cat([z_full[:, :k], z_t, z_full[:, k + nfb:]], 1)
    ts = torch.full((1, K), float(t), device=gtc.device, dtype=torch.float32)
    flow, _ = bidi(noisy_image_or_video=z_full, conditional_dict=cond, timestep=ts)
    return flow[:, k:k + nfb]


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    merged = {k: v.to(torch.bfloat16) for k, v in torch.load(ADAPTED, map_location="cpu")["merged"].items()}
    pipe.generator.model.load_state_dict(merged, strict=False)
    print(f"causal host loaded with {ADAPTED}", flush=True)

    bidi_pris = WanDiffusionWrapper(is_causal=False).to(dtype=torch.bfloat16).cuda()
    bidi_adap = WanDiffusionWrapper(is_causal=False).to(dtype=torch.bfloat16).cuda()
    missing, unexpected = bidi_adap.model.load_state_dict(merged, strict=False)
    print(f"bidi-adapted load: {len(missing)} missing / {len(unexpected)} unexpected keys", flush=True)

    d = torch.load(os.path.join("wan_cache", PAIRS), map_location="cpu")
    gt, gen, caps = d["gt"], d["gen"], d["captions"]
    N = gt.shape[0]
    nfb = pipe.num_frame_per_block
    ks = list(range(W, K - 2 * nfb + 1))  # keep >= nfb future latents so the oracle has a real future
    sched = pipe._initialize_sample_scheduler(torch.zeros(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
    sigmas, tsteps = sched.sigmas.to(DEVICE).float(), sched.timesteps.to(DEVICE).float()
    rng = np.random.default_rng(0)
    gen_ctx = torch.Generator(device=DEVICE)

    rows = []
    for si in range(NSTATES):
        c = int(rng.integers(N)); k = int(rng.choice(ks))
        idx = int(rng.integers(len(tsteps))); t = tsteps[idx]; sig = sigmas[idx]
        cond = pipe.text_encoder(text_prompts=[caps[c]])
        gtc = gt[c:c + 1].to(DEVICE).to(torch.bfloat16)
        genc = gen[c:c + 1].to(DEVICE).to(torch.bfloat16)
        gt_hist, gen_hist, x0 = gtc[:, k - W:k], genc[:, k - W:k], genc[:, k:k + nfb]
        # v3-c future: deterministic per state (a function of the clean past, not a fresh
        # sample per seed — that determinism is the point of the predicted-future teacher)
        fut_pred = predict_future(pipe, cond, gtc[:, :k], nfb, sigmas, tsteps,
                                  n_lat=K - k - nfb, seed=si)
        vg, vc, vA, vP, vB, vC = [], [], [], [], [], []
        for s in range(NSEEDS):
            gen_ctx.manual_seed(10_000 * si + s)
            eps = torch.randn(x0.shape, device=DEVICE, dtype=x0.dtype, generator=gen_ctx)
            z_t = (1 - sig) * x0 + sig * eps
            vg.append(velocity_tf(pipe, cond, gen_hist, x0, z_t, t).float())
            vc.append(velocity_tf(pipe, cond, gt_hist, x0, z_t, t).float())
            vA.append(velocity_bidi(bidi_adap, cond, gtc, z_t, k, nfb, t, sig, gen_ctx).float())
            vP.append(velocity_bidi(bidi_pris, cond, gtc, z_t, k, nfb, t, sig, gen_ctx).float())
            vB.append(velocity_bidi(bidi_adap, cond, gtc, z_t, k, nfb, t, sig, gen_ctx,
                                    past=genc[:, :k]).float())  # bridge: drifted past + clean future
            vC.append(velocity_bidi(bidi_adap, cond, gtc, z_t, k, nfb, t, sig, gen_ctx,
                                    future=fut_pred).float())   # v3-c: clean past + predicted future
        vg, vc, vA, vP, vB, vC = [torch.cat(v) for v in (vg, vc, vA, vP, vB, vC)]

        def alpha(target):
            gap = target - vg
            return (gap.mean(0) ** 2).sum().item() / ((gap ** 2).mean(0).sum().item() + 1e-12)

        gap_mean = (vc - vg).mean(0)
        gnorm = gap_mean.norm().item() + 1e-12
        row = dict(c=c, k=k, t=float(t),
                   a_past=alpha(vc), a_orA=alpha(vA), a_orP=alpha(vP), a_orB=alpha(vB), a_orC=alpha(vC),
                   info_A=(vA.mean(0) - vc.mean(0)).norm().item() / gnorm,
                   info_P=(vP.mean(0) - vc.mean(0)).norm().item() / gnorm,
                   info_B=(vB.mean(0) - vg.mean(0)).norm().item() / gnorm,  # bridge signal vs what host does
                   info_C=(vC.mean(0) - vc.mean(0)).norm().item() / gnorm,  # v3-c new info vs past teacher
                   gap=gnorm)
        rows.append(row)
        print(f"state {si:2d} c={c:3d} k={k:2d} t={float(t):6.1f} | a_past {row['a_past']:.3f} "
              f"a_orA {row['a_orA']:.3f} a_orP {row['a_orP']:.3f} a_orB {row['a_orB']:.3f} "
              f"a_orC {row['a_orC']:.3f} | info_A {row['info_A']:.3f} info_P {row['info_P']:.3f} "
              f"info_B {row['info_B']:.3f} info_C {row['info_C']:.3f}", flush=True)
        np.savez(OUT, **{key: np.array([r[key] for r in rows]) for key in rows[0]})  # save-first discipline

    arr = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    lo, hi = np.percentile(arr["t"], [33, 66])
    for name in ("a_past", "a_orA", "a_orP", "a_orB", "a_orC", "info_A", "info_P", "info_B", "info_C"):
        v = arr[name]
        by_t = [v[arr["t"] <= lo].mean(), v[(arr["t"] > lo) & (arr["t"] <= hi)].mean(), v[arr["t"] > hi].mean()]
        print(f"GATE {name}: mean {v.mean():.3f} median {np.median(v):.3f} | "
              f"by-t-tercile lo/mid/hi {by_t[0]:.3f}/{by_t[1]:.3f}/{by_t[2]:.3f}", flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
