"""Stage 2 of the 14B viability gate: drifted rollouts on the UNADAPTED 14B run block-causally.

Causal AR K=48-latent rollouts (rolling 21-latent KV window, 3-latent chunks, 20-step UniPC,
CFG 6 / shift 8) seeded from each clean ref's first NCTX=3 latents + same caption -- the
wan_build_pairs_synth.py protocol with wan_build_pairs.py's K=48 horizon. Saves drift frame
strips + the saturation drift proxy (1.3B drift = low-freq color/saturation).
Resumable. Output: wan_cache/wan14b_pairs.pt {gt (N,21), gen (N,48), captions, sat_gen}
+ outputs_wan14b/rollout_clip{c}_strip.png. Run from repo root.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import numpy as np
import torch
from wan14b_common import (CausalDiffusionInferencePipeline14B, load_cfg, save_strip,
                           mean_saturation, write_timing)

DEVICE = "cuda"
K = int(os.environ.get("K", 48))
NCTX = 3
REFS = "wan_cache/wan14b_refs.pt"
OUT = "wan_cache/wan14b_pairs.pt"
STRIPS = "outputs_wan14b"


@torch.no_grad()
def main():
    os.makedirs(STRIPS, exist_ok=True)
    cfg = load_cfg()
    torch.set_grad_enabled(False)
    t_load = time.time()
    pipe = CausalDiffusionInferencePipeline14B(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    load_s = time.time() - t_load
    print(f"14B causal pipeline up in {load_s:.0f}s | {pipe.num_transformer_blocks} blocks, "
          f"window {pipe.local_attn_size}, nfb {pipe.num_frame_per_block}, CFG {cfg.guidance_scale}", flush=True)

    d = torch.load(REFS, map_location="cpu")
    gt_all, caps = d["gt"], d["captions"]
    N = gt_all.shape[0]

    gens, sats, done = [], [], 0
    if os.path.exists(OUT):
        prev = torch.load(OUT, map_location="cpu")
        gens = list(prev["gen"].unbind(0))
        sats = [s for s in prev["sat_gen"]]
        done = len(gens)
        print(f"resuming from {done}", flush=True)

    t0, per = time.time(), []
    for c in range(done, N):
        tc = time.time()
        gt = gt_all[c:c + 1].to(DEVICE).to(torch.bfloat16)
        torch.manual_seed(c)
        noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
        video, gen = pipe.inference(noise=noise, text_prompts=[caps[c]],
                                    initial_latent=gt[:, :NCTX], return_latents=True)
        gens.append(gen[0, :K].half().cpu())
        sat = mean_saturation(video[0])
        sats.append(sat)
        save_strip(video[0], os.path.join(STRIPS, f"rollout_clip{c}_strip.png"), n=10)
        torch.save({"gt": gt_all, "gen": torch.stack(gens), "captions": caps,
                    "sat_gen": np.stack(sats)}, OUT)
        per.append(round(time.time() - tc, 1))
        print(f"rollout {c + 1}/{N} in {per[-1]}s | sat first8 {sat[:8].mean():.3f} "
              f"-> last8 {sat[-8:].mean():.3f}", flush=True)

    write_timing("rollouts", time.time() - t0, N - done,
                 {"model_load_s": round(load_s, 1), "per_clip_s": per, "K": K})
    print(f"done: {len(gens)} drifted rollouts -> {OUT}")


if __name__ == "__main__":
    main()
