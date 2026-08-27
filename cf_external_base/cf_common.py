"""Shared loader for the Causal-Forcing (CF) paper-row scripts.

Builds our CausalDiffusionInferencePipeline (rolling 21-frame KV window, sink 0,
20-step UniPC, shift 5 — the locked CF-row protocol, same as the feasibility gate)
and loads the CF ar_diffusion generator weights strict (825/825 key match).

All scripts must run from /localhome/local-wenqingw/projs/Self-Forcing in the
df-gb300 env.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys

import torch
from omegaconf import OmegaConf

SF_REPO = "/localhome/local-wenqingw/projs/Self-Forcing"
sys.path.insert(0, SF_REPO)

from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline  # noqa: E402

ROW = "/localhome/local-wenqingw/projs/benchmarking/cf_row"
CKPTS = os.path.join(ROW, "ckpts")
CF_CKPT = "/localhome/local-wenqingw/projs/benchmarking/cf_feasibility/weights/ar_diffusion.pt"
CLIPS = os.path.join(SF_REPO, "wan_cache/synth_clips.pt")
PROMPTS = os.path.join(SF_REPO, "wan_cache/finals128/prompts_used.txt")
DEVICE = "cuda"
K, NCTX, W = 21, 3, 9


def load_cf_pipe():
    cfg = OmegaConf.merge(OmegaConf.load(os.path.join(SF_REPO, "configs/default_config.yaml")),
                          OmegaConf.load(os.path.join(SF_REPO, "configs/_undistilled_smoke.yaml")))
    # Robustness-matrix overrides (Joonghyuk): ATTN=-1 -> full KV cache; SINK=N -> N-frame attention sink
    if "ATTN" in os.environ:
        cfg.model_kwargs.local_attn_size = int(os.environ["ATTN"])
    if "SINK" in os.environ:
        cfg.model_kwargs.sink_size = int(os.environ["SINK"])
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE))
    sd = torch.load(CF_CKPT, map_location="cpu")["generator"]
    pipe.generator.load_state_dict(sd)  # strict — 825/825 verified in feasibility
    pipe = pipe.to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    print("CF ar_diffusion base loaded (strict) | local_attn_size="
          f"{pipe.generator.model.local_attn_size} sink={pipe.generator.model.sink_size} "
          f"steps={cfg.sampling_steps}", flush=True)
    return pipe


def load_lora(model, path, scale=1.0, rank=16):
    from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters
    apply_lora(model, rank=rank)
    lw = torch.load(path, map_location="cpu")["lora"]
    lw = list(lw.values()) if isinstance(lw, dict) else lw
    for p, w in zip(lora_parameters(model), lw):
        p.data.copy_(w.to(p.device, p.dtype))
    set_lora_scale(model, scale)
    print(f"corrector LoRA loaded: {path} @ scale {scale}", flush=True)
