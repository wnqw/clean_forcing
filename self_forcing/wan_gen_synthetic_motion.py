"""Motion-rich clean references (progression-bias fix, Stage 2): same recipe as wan_gen_synthetic.py
but every prompt gets an explicit camera-motion clause, so "clean" includes scene progression.
Output: wan_cache/synth_clips_motion.pt (same prompts/order as synth_clips.pt for pairing).
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from omegaconf import OmegaConf

from pipeline.bidirectional_diffusion_inference import BidirectionalDiffusionInferencePipeline

DEVICE = "cuda"
NCLIPS = int(os.environ.get("NCLIPS", 300))
STEPS = int(os.environ.get("STEPS", 30))
OUT = "wan_cache/synth_clips_motion.pt"

# Motion-DIVERSE, not motion-biased: the corrector must inherit no systematic camera prior —
# camera behavior belongs to the prompt. ~40% keep natural phrasing ("" entries), the rest
# spread over camera types including explicit static.
MOTION_CLAUSES = [
    "", "", "", "",                                                        # natural (no clause)
    " The camera steadily tracks forward through the scene.",
    " A smooth dolly shot, the camera moving continuously ahead.",
    " The camera pans slowly across the scene, revealing new details.",
    " A tracking shot following the subject as it moves through the environment.",
    " A slow orbiting arc shot circling the subject.",
    " A static tripod shot with a perfectly still camera.",
]


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    pipe = BidirectionalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.sampling_steps = STEPS

    allp = [l.strip() for l in open("prompts/MovieGenVideoBench_extended.txt") if l.strip()]
    eval_base = [l.strip() for l in open("prompts/MovieGenVideoBench.txt") if l.strip()]
    eval_idx = set(range(0, len(eval_base), max(1, len(eval_base) // 16)))
    base_prompts = [p for i, p in enumerate(allp) if i not in eval_idx][:NCLIPS]
    prompts = [p + MOTION_CLAUSES[i % len(MOTION_CLAUSES)] for i, p in enumerate(base_prompts)]

    lats, caps = [], []
    done = 0
    if os.path.exists(OUT):
        d = torch.load(OUT, map_location="cpu")
        lats, caps = list(d["gt"].unbind(0)), list(d["captions"])
        done = len(caps)
        print(f"resuming from {done} clips", flush=True)

    for i in range(done, len(prompts)):
        torch.manual_seed(200000 + i)
        noise = torch.randn(1, 21, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
        _, lat = pipe.inference(noise=noise, text_prompts=[prompts[i]], return_latents=True)
        lats.append(lat.squeeze(0).half().cpu())
        caps.append(prompts[i])
        if (i + 1) % 20 == 0 or i == len(prompts) - 1:
            torch.save({"gt": torch.stack(lats), "captions": caps}, OUT)
            print(f"{i + 1}/{len(prompts)} clips saved", flush=True)

    print(f"done: {len(caps)} motion-rich clips -> {OUT}")


if __name__ == "__main__":
    main()
