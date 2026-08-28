"""
Open-loop validation R^2 per LoRA-rank checkpoint on the ADAPTED host — the trainer's val
protocol (wan_train_lora_k48.py) with a fixed, shared state list so ranks are compared on
identical states:
  pairs_k48_adapt.pt, val clips 35-39, W=9, k in 9..45, 20-step scheduler timesteps,
  z_t = (1-sigma)x0 + sigma*eps with per-state fixed eps,
  R^2 = 1 - sum||v_corr - v_clean||^2 / sum||v_clean - v_base||^2  (teacher passes at scale 0).

Env: CKPTS comma list of name=ckpt:rank; NSTATES (default 64); OUT_JSON.
Run from Self-Forcing: python -u wan_rank_valr2.py
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters, num_lora_params

DEVICE = "cuda"
K, W = 48, 9
CACHE = "wan_cache"
ADAPTED = os.path.join(CACHE, "adapted_base_4000.pt")
PAIRS = os.path.join(CACHE, "pairs_k48_adapt.pt")
NSTATES = int(os.environ.get("NSTATES", "64"))
OUT_JSON = os.environ["OUT_JSON"]


def velocity_tf(pipe, cond, history, x0_cur, z_t, t):
    Wh, nf = history.shape[1], z_t.shape[1]
    clean_x = torch.cat([history, x0_cur], 1)
    noisy = torch.cat([history, z_t], 1)
    ts = torch.cat([torch.zeros((1, Wh), device=z_t.device, dtype=torch.float32),
                    torch.full((1, nf), float(t), device=z_t.device, dtype=torch.float32)], 1)
    flow, _ = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts, clean_x=clean_x)
    return flow[:, Wh:]


def build_pipe(cfg, ckpt, rank):
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    sd = torch.load(ADAPTED, map_location="cpu")["merged"]
    pipe.generator.model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
    model = pipe.generator.model
    apply_lora(model, rank=rank)
    lora_sd = torch.load(os.path.join(CACHE, ckpt), map_location="cpu")["lora"]
    params = lora_parameters(model)
    assert len(params) == len(lora_sd), f"{ckpt}: {len(lora_sd)} tensors vs {len(params)} params"
    for i, p in enumerate(params):
        assert p.shape == lora_sd[i].shape, f"{ckpt} shape mismatch at {i}"
        p.data.copy_(lora_sd[i].to(p.device, p.dtype))
    return pipe


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    d = torch.load(PAIRS, map_location="cpu")
    gt, gen, caps = d["gt"], d["gen"], d["captions"]
    N = gt.shape[0]
    nval = max(4, N // 8)
    val_ids = list(range(N - nval, N))

    entries = []
    for e in os.environ["CKPTS"].split(","):
        name, spec = e.strip().split("=")
        ckpt, rank = spec.rsplit(":", 1)
        entries.append((name, ckpt, int(rank)))

    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    state_list = None
    for name, ckpt, rank in entries:
        if name in results:
            print(f"[{name}] already done, skipping", flush=True)
            continue
        t0 = time.time()
        pipe = build_pipe(cfg, ckpt, rank)
        model = pipe.generator.model
        print(f"[{name}] {ckpt} rank {rank} | {num_lora_params(model)/1e6:.2f}M lora params", flush=True)
        nfb = pipe.num_frame_per_block
        ks = list(range(W, K - nfb + 1))
        sched = pipe._initialize_sample_scheduler(torch.zeros(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
        sigmas, tsteps = sched.sigmas.to(DEVICE).float(), sched.timesteps.to(DEVICE).float()
        if state_list is None:  # identical states for every checkpoint
            rng = np.random.default_rng(123)
            state_list = [(int(rng.choice(val_ids)), int(rng.choice(ks)), int(rng.integers(len(tsteps))))
                          for _ in range(NSTATES)]
        cond_cache = {c: pipe.text_encoder(text_prompts=[caps[c]]) for c in set(s[0] for s in state_list)}

        num = den = 0.0
        for i, (c, k, ti) in enumerate(state_list):
            t, sig = tsteps[ti], sigmas[ti]
            gtc = gt[c:c + 1].to(DEVICE).to(torch.bfloat16)
            genc = gen[c:c + 1].to(DEVICE).to(torch.bfloat16)
            gt_hist, gen_hist, x0 = gtc[:, k - W:k], genc[:, k - W:k], genc[:, k:k + nfb]
            torch.manual_seed(9000 + i)
            eps = torch.randn(x0.shape, device=DEVICE, dtype=x0.dtype)
            z_t = (1 - sig) * x0 + sig * eps
            set_lora_scale(model, 0.0)
            v_clean = velocity_tf(pipe, cond_cache[c], gt_hist, x0, z_t, t)
            v_base = velocity_tf(pipe, cond_cache[c], gen_hist, x0, z_t, t)
            set_lora_scale(model, 1.0)
            v_corr = velocity_tf(pipe, cond_cache[c], gen_hist, x0, z_t, t)
            num += ((v_corr - v_clean) ** 2).sum().item()
            den += ((v_clean - v_base) ** 2).sum().item()
        r2 = 1 - num / (den + 1e-12)
        results[name] = {"ckpt": ckpt, "rank": rank, "val_r2": round(r2, 4), "nstates": NSTATES,
                         "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(results, open(OUT_JSON, "w"), indent=1)
        print(f"RESULT [{name}] val R^2 {r2:+.4f} over {NSTATES} states | {(time.time()-t0)/60:.0f} min", flush=True)
        del pipe
        torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
