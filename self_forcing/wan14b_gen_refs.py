"""Stage 1 of the 14B viability gate: clean references (h_clean source of the zero-real recipe).

8 single-shot BIDIRECTIONAL 21-latent (5 s) clips from the MovieGen training-prompt pool
(wan_gen_synthetic.py pool logic + seeds; prompts disjoint from the finals 128, asserted).
30-step UniPC teacher sampling (shift 8 hardcoded in the bidirectional pipeline), CFG 6.
Resumable. Output: wan_cache/wan14b_refs.pt {gt (N,21,16,60,104) fp16, captions, saturation}
+ outputs_wan14b/ref_clip{c}_strip.png. Run from repo root.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import numpy as np
import torch
from pipeline.bidirectional_diffusion_inference import BidirectionalDiffusionInferencePipeline
from wan14b_common import load_cfg, ref_prompt_pool, save_strip, mean_saturation, write_timing

DEVICE = "cuda"
NCLIPS = int(os.environ.get("NCLIPS", 8))
STEPS = int(os.environ.get("STEPS", 30))
OUT = "wan_cache/wan14b_refs.pt"
STRIPS = "outputs_wan14b"


@torch.no_grad()
def main():
    os.makedirs(STRIPS, exist_ok=True)
    cfg = load_cfg()
    torch.set_grad_enabled(False)
    t_load = time.time()
    pipe = BidirectionalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.sampling_steps = STEPS
    load_s = time.time() - t_load
    print(f"14B bidirectional pipeline up in {load_s:.0f}s | CFG {cfg.guidance_scale} steps {STEPS}", flush=True)
    prompts = ref_prompt_pool(NCLIPS)

    lats, caps, sats = [], [], []
    done = 0
    if os.path.exists(OUT):
        d = torch.load(OUT, map_location="cpu")
        lats, caps = list(d["gt"].unbind(0)), list(d["captions"])
        sats = [s for s in d["saturation"]]
        done = len(caps)
        print(f"resuming from {done} clips", flush=True)

    t0, per = time.time(), []
    for i in range(done, NCLIPS):
        tc = time.time()
        torch.manual_seed(100000 + i)
        noise = torch.randn(1, 21, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
        video, lat = pipe.inference(noise=noise, text_prompts=[prompts[i]], return_latents=True)
        lats.append(lat.squeeze(0).half().cpu())
        caps.append(prompts[i])
        sats.append(mean_saturation(video[0]))
        if i < 16 or i % 50 == 0:
            save_strip(video[0], os.path.join(STRIPS, f"ref_clip{i}_strip.png"))
        if (i + 1) % 10 == 0 or i == NCLIPS - 1:
            torch.save({"gt": torch.stack(lats), "captions": caps, "saturation": np.stack(sats)}, OUT)
        per.append(round(time.time() - tc, 1))
        print(f"ref clip {i + 1}/{NCLIPS} in {per[-1]}s | mean sat {sats[-1].mean():.3f}", flush=True)

    write_timing("refs", time.time() - t0, NCLIPS - done,
                 {"model_load_s": round(load_s, 1), "per_clip_s": per, "steps": STEPS})
    print(f"done: {len(caps)} clean refs -> {OUT}")


if __name__ == "__main__":
    main()
