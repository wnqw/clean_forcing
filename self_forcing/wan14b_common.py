"""Shared plumbing for the Wan2.1-T2V-14B Clean Forcing viability gate (scale-up feasibility).

Replicates the 1.3B paper-host protocol on the UNADAPTED 14B (raw bidirectional weights in the
causal wrapper): block-causal 3-latent chunks, 20-step UniPC, rolling 21-latent KV window,
CFG 6 / shift 8, 832x480, bf16, batch 1. At 480x832 the latent grid (60x104) and patch size are
unchanged, so frame_seq_length=1560 and all window math carry over; only the transformer dims
change (dim 5120, 40 heads, 40 layers; head_dim stays 128 -> identical RoPE tables).
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import numpy as np
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline

FINALS_PROMPTS = "wan_cache/finals128/prompts_used.txt"


def load_cfg():
    return OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                           OmegaConf.load("configs/_undistilled_smoke.yaml"),
                           OmegaConf.load("configs/_wan14b_gate.yaml"))


class CausalDiffusionInferencePipeline14B(CausalDiffusionInferencePipeline):
    """1.3B inference pipeline with its hardcoded cache dims (30 blocks, 12 heads x 128)
    parametrized from the loaded model config. Chunking/KV rolling/solver unchanged."""

    def __init__(self, args, device):
        super().__init__(args, device)
        m = self.generator.model
        self.num_transformer_blocks = len(m.blocks)
        self._kv_heads = m.num_heads
        self._kv_head_dim = m.dim // m.num_heads

    def _initialize_kv_cache(self, batch_size, dtype, device):
        assert not getattr(self, "hg_scale", 0) and not getattr(self, "ttc_steps", None), \
            "14B gate pipeline: HG/TTC caches not ported"
        kv_size = self.local_attn_size * self.frame_seq_length if self.local_attn_size != -1 else 32760

        def _mk():
            return [{"k": torch.zeros([batch_size, kv_size, self._kv_heads, self._kv_head_dim],
                                      dtype=dtype, device=device),
                     "v": torch.zeros([batch_size, kv_size, self._kv_heads, self._kv_head_dim],
                                      dtype=dtype, device=device),
                     "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                     "local_end_index": torch.tensor([0], dtype=torch.long, device=device)}
                    for _ in range(self.num_transformer_blocks)]
        self.kv_cache_pos, self.kv_cache_neg = _mk(), _mk()

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        def _mk():
            return [{"k": torch.zeros([batch_size, 512, self._kv_heads, self._kv_head_dim],
                                      dtype=dtype, device=device),
                     "v": torch.zeros([batch_size, 512, self._kv_heads, self._kv_head_dim],
                                      dtype=dtype, device=device),
                     "is_init": False}
                    for _ in range(self.num_transformer_blocks)]
        self.crossattn_cache_pos, self.crossattn_cache_neg = _mk(), _mk()


def ref_prompt_pool(n):
    """First n prompts of the zero-real training pool (wan_gen_synthetic.py logic:
    MovieGen extended minus the 16 wan_bagger_eval indices). The finals-128 prompts are
    extended-file indices 320..955, so the head of the pool is disjoint -- asserted."""
    allp = [l.strip() for l in open("prompts/MovieGenVideoBench_extended.txt") if l.strip()]
    eval_base = [l.strip() for l in open("prompts/MovieGenVideoBench.txt") if l.strip()]
    eval_idx = set(range(0, len(eval_base), max(1, len(eval_base) // 16)))
    pool = [p for i, p in enumerate(allp) if i not in eval_idx]
    prompts = pool[:n]
    finals = {l.strip() for l in open(FINALS_PROMPTS) if l.strip()}
    overlap = set(prompts) & finals
    assert not overlap, f"ref prompts overlap finals-128: {len(overlap)}"
    return prompts


def save_strip(video, path, n=8):
    """video: (F,3,H,W) float [0,1] -> horizontal strip PNG of n evenly spaced frames."""
    import imageio.v2 as iio
    idx = np.linspace(0, video.shape[0] - 1, n).astype(int)
    strip = torch.cat([video[int(i)] for i in idx], dim=-1)
    arr = (strip.permute(1, 2, 0).clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
    iio.imwrite(path, arr)
    return [int(i) for i in idx]


def mean_saturation(video):
    """Per-frame mean HSV-style saturation (max-min)/max -- the 1.3B drift lives in
    low-freq color/saturation, so this is the cheap objective drift proxy. -> (F,) np"""
    v = video.float()
    mx = v.max(dim=1).values
    mn = v.min(dim=1).values
    s = (mx - mn) / mx.clamp_min(1e-6)
    return s.mean(dim=(1, 2)).cpu().numpy()


def write_timing(stage, wall_s, n_items, extra=None):
    d = {"stage": stage, "wall_s": round(wall_s, 1), "n_items": n_items,
         "per_item_s": round(wall_s / max(n_items, 1), 1),
         "gpu": torch.cuda.get_device_name(0)}
    if extra:
        d.update(extra)
    path = f"wan_cache/wan14b_timing_{stage}.json"
    with open(path, "w") as f:
        json.dump(d, f, indent=1)
    print(f"timing -> {path}: {d}", flush=True)
