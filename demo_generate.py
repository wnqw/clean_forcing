"""Minimal demo: one prompt -> one 50s video, with or without a Clean Forcing corrector.

Examples (run from the repo root):
  # corrected (Table-1 headline checkpoint):
  LORA=weights/lora_r_phi_v2_both_adapt.pt PROMPT="a corgi surfing a wave at sunset" \
      python demo_generate.py
  # uncorrected base (watch it drift/collapse):
  LORA=none PROMPT="a corgi surfing a wave at sunset" python demo_generate.py
  # rank-sweep checkpoints: add RANK=8|32|64 to match the file.
  # shorter clip: KLAT=120 (~30s). Output: demo_<tag>.mp4 (16 fps, 832x480).

Requires: SETUP.sh completed + the adapted base checkpoint at
self_forcing/wan_cache/adapted_base_4000.pt (see RUN.md "Adapted base").
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys

import imageio.v2 as imageio
import torch
from omegaconf import OmegaConf

LORA = os.environ.get("LORA", "weights/lora_r_phi_v2_both_adapt.pt")
PROMPT = os.environ.get("PROMPT", "a steaming cup of coffee on a wooden table, morning light")
KLAT = int(os.environ.get("KLAT", 201))            # 201 latents ~= 50 s @ 16 fps
RANK = int(os.environ.get("RANK", 16))
SEED = int(os.environ.get("SEED", 0))
BASE = os.environ.get("ADAPTED_BASE", "wan_cache/adapted_base_4000.pt")
LORA_ABS = os.path.abspath(LORA) if LORA != "none" else "none"

sys.path.insert(0, "./self_forcing")
os.chdir("./self_forcing")
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline  # noqa: E402
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters  # noqa: E402


@torch.no_grad()
def main():
    torch.set_grad_enabled(False)
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device("cuda")).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None

    sd = torch.load(BASE, map_location="cpu")["merged"]
    pipe.generator.model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)

    tag = "base"
    if LORA_ABS != "none":
        apply_lora(pipe.generator.model, rank=RANK)
        lw = torch.load(LORA_ABS, map_location="cpu")["lora"]
        lw = list(lw.values()) if isinstance(lw, dict) else lw
        for p, w in zip(lora_parameters(pipe.generator.model), lw):
            p.data.copy_(w.to(p.device, p.dtype))
        set_lora_scale(pipe.generator.model, 1.0)
        tag = os.path.splitext(os.path.basename(LORA_ABS))[0]

    torch.manual_seed(SEED)
    noise = torch.randn(1, KLAT, 16, 60, 104, device="cuda", dtype=torch.bfloat16)
    video = pipe.inference(noise=noise, text_prompts=[PROMPT])
    frames = (video[0].permute(0, 2, 3, 1).float() * 255).byte().cpu().numpy()
    out = f"../demo_{tag}.mp4"
    imageio.mimsave(out, frames, fps=16, quality=8)
    print(f"wrote {os.path.abspath(out)}  ({len(frames)} frames, prompt: {PROMPT!r})")


if __name__ == "__main__":
    main()
