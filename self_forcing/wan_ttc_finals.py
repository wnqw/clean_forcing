"""Pathwise TTC finals row (arXiv 2602.05871): adapted base + on-path correction at levels {500, 250}
(their chosen config), 128 finals prompts, 50s un-seeded T2V. Resumable; batch-4; seed = prompt index.
Tag: attc. Score afterwards with scripts/score_finals_row.py --tag attc.
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import imageio.v2 as imageio
import torch
from omegaconf import OmegaConf

from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline

DEVICE = "cuda"
KLAT = 201
BATCH = 4
D = "wan_cache/finals128"
ADAPTED = "wan_cache/adapted_base_4000.pt"
TTC_STEPS = [int(s) for s in os.environ.get("TTC_STEPS", "500,250").split(",")]
TAG = os.environ.get("TAG", "attc")


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    prompts = [ln.strip() for ln in open(f"{D}/prompts_used.txt") if ln.strip()]
    assert len(prompts) == 128

    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    sd = torch.load(ADAPTED, map_location="cpu")["merged"]
    pipe.generator.model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
    pipe.ttc_steps = TTC_STEPS
    print(f"adapted base + pathwise TTC @ {TTC_STEPS} tag={TAG}", flush=True)

    for b0 in range(0, 128, BATCH):
        idxs = [i for i in range(b0, min(b0 + BATCH, 128))
                if not os.path.exists(f"{D}/{TAG}_p{i:03d}.mp4")]
        if not idxs:
            continue
        pad = [idxs[-1]] * (BATCH - len(idxs))  # kv caches are allocated at BATCH; pad partial batches
        noise = []
        for i in idxs + pad:
            torch.manual_seed(i)
            noise.append(torch.randn(1, KLAT, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
        noise = torch.cat(noise)
        video = pipe.inference(noise=noise, text_prompts=[prompts[i] for i in idxs + pad])
        for bi, i in enumerate(idxs):
            fr = (video[bi].permute(0, 2, 3, 1).float() * 255).byte().cpu().numpy()
            tmp = f"{D}/{TAG}_p{i:03d}.mp4.tmp.mp4"
            imageio.mimsave(tmp, fr, fps=16, quality=8)
            os.rename(tmp, f"{D}/{TAG}_p{i:03d}.mp4")
        print(f"DONE batch {b0 // BATCH + 1}/32 (prompts {idxs})", flush=True)
    print(f"{TAG} finals complete", flush=True)


if __name__ == "__main__":
    main()
