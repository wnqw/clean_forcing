"""
Prompts-only training data (AutoRefiner data regime): the BIDIRECTIONAL Wan generates 5s/21-latent
clips (single-shot full attention = non-drifted by construction) from diverse MovieGen prompts.
These synthetic clips become h_clean for the corrector loss — zero real videos.
Skips the 16 eval-prompt indices used by wan_bagger_eval (no leakage). Saves incrementally.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from omegaconf import OmegaConf
from pipeline.bidirectional_diffusion_inference import BidirectionalDiffusionInferencePipeline

DEVICE = "cuda"
NCLIPS = int(os.environ.get("NCLIPS", 300))
STEPS = int(os.environ.get("STEPS", 30))       # UniPC steps for the teacher clips
OUT = "wan_cache/synth_clips.pt"


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    pipe = BidirectionalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.sampling_steps = STEPS

    allp = [l.strip() for l in open("prompts/MovieGenVideoBench_extended.txt") if l.strip()]
    eval_base = [l.strip() for l in open("prompts/MovieGenVideoBench.txt") if l.strip()]
    eval_idx = set(range(0, len(eval_base), max(1, len(eval_base) // 16)))   # wan_bagger_eval's stride
    prompts = [p for i, p in enumerate(allp) if i not in eval_idx][:NCLIPS]

    lats, caps = [], []
    done = 0
    if os.path.exists(OUT):                                  # resume
        d = torch.load(OUT, map_location="cpu")
        lats, caps = list(d["gt"].unbind(0)), list(d["captions"])
        done = len(caps)
        print(f"resuming from {done} clips", flush=True)

    for i in range(done, len(prompts)):
        torch.manual_seed(100000 + i)
        noise = torch.randn(1, 21, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
        _, lat = pipe.inference(noise=noise, text_prompts=[prompts[i]], return_latents=True)
        lats.append(lat.squeeze(0).half().cpu()); caps.append(prompts[i])
        if (i + 1) % 20 == 0 or i == len(prompts) - 1:
            torch.save({"gt": torch.stack(lats), "captions": caps}, OUT)
            print(f"{i + 1}/{len(prompts)} clips saved", flush=True)

    print(f"done: {len(caps)} synthetic clips -> {OUT}")


if __name__ == "__main__":
    main()
