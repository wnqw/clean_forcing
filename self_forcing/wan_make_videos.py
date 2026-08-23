"""
Qualitative side-by-side videos: baseline (left) vs ours (right), same clip + seed, K=48.
Writes wan_cache/videos/cmp_clip{c}.mp4 (labels burned in). This is the qualitative drift
figure — baseline should visibly degrade/oversaturate over the horizon while ours stays clean.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters

DEVICE = "cuda"
K, NCTX, NCLIPS, FPS = 48, 3, 4, 16
OUT, VID = "wan_cache", "wan_cache/videos"
CKPT = "lora_r_phi_k48.pt"


def to_uint8(pix):                       # (F,3,H,W) [-1,1] -> (F,H,W,3) uint8
    x = ((pix.float().permute(0, 2, 3, 1) + 1) / 2).clamp(0, 1) * 255
    return x.round().byte().cpu().numpy()


@torch.no_grad()
def rollout(pipe, seed_lat, cap, c):
    torch.manual_seed(c)
    noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
    _, lat = pipe.inference(noise=noise, text_prompts=[cap], initial_latent=seed_lat, return_latents=True)
    return to_uint8(pipe.vae.decode_to_pixel(lat[:, :K])[0])


def label(frame, text):                  # burn a caption top-left
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(img)


@torch.no_grad()
def main():
    os.makedirs(VID, exist_ok=True)
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    d = torch.load(os.path.join(OUT, "pairs_k48.pt"), map_location="cpu")
    gt, caps = d["gt"], d["captions"]
    val_ids = list(range(gt.shape[0] - NCLIPS, gt.shape[0]))

    # baseline pass (no LoRA) for all clips first, then ours (LoRA modifies weights)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    base = {c: rollout(pipe, gt[c:c + 1, :NCTX].to(DEVICE).to(torch.bfloat16), caps[c], c) for c in val_ids}
    print("baseline rollouts done", flush=True)

    model = pipe.generator.model
    apply_lora(model, rank=16)
    sd = torch.load(os.path.join(OUT, CKPT), map_location="cpu")["lora"]
    for i, p in enumerate(lora_parameters(model)):
        p.data.copy_(sd[i].to(p.device, p.dtype))
    set_lora_scale(model, 1.0)
    ours = {c: rollout(pipe, gt[c:c + 1, :NCTX].to(DEVICE).to(torch.bfloat16), caps[c], c) for c in val_ids}
    print("ours rollouts done", flush=True)

    sep = np.full((base[val_ids[0]].shape[1], 4, 3), 255, np.uint8)     # white divider column
    for c in val_ids:
        frames = []
        for fb, fo in zip(base[c], ours[c]):
            frames.append(np.concatenate([label(fb, "baseline"), sep, label(fo, "ours (r_phi)")], axis=1))
        path = os.path.join(VID, f"cmp_clip{c}.mp4")
        imageio.mimsave(path, frames, fps=FPS, quality=8)
        print(f"wrote {path}  ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})", flush=True)
    print(f"\n{len(val_ids)} comparison videos in {VID}/")


if __name__ == "__main__":
    main()
