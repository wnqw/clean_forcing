"""
Causal adaptation with BLOCK-SIZE JITTER (issue-3 retrain arm): identical to wan_train_adapt.py
except each step samples a random partition of the 21 latents into chunks of size {2,3,4}
(canonical [3]*7 included in the pool). Rationale: the seam pulse is structural detail loss on
block-edge frames baked in by fixed-cadence adaptation; jitter makes no frame systematically an
edge, so the defect should not imprint. Inference cadence stays nfb=3 (deployment unchanged).
Go/no-go after training: pulse-ac12 of the UNCORRECTED jittered base vs abase's 0.80.
Run: STEPS=4000 python -u wan_train_adapt_jitter.py -> adapted_base_jitter_{step}.pt
"""
import math
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:native")
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.attention.flex_attention import create_block_mask
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, lora_parameters, num_lora_params, LoRALinear

DEVICE = "cuda"
STEPS = int(os.environ.get("STEPS", 4000))
RANK = 64
LR = 1e-4
WARMUP = 100
CKPT_EVERY = 2000
EVAL_EVERY = 200
K = 21
N_PARTS = 12
OUT = "wan_cache"
TAGP = os.environ.get("TAGP", "jitter")


def merged_state(model):
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            w = module.base.weight.data.float() + module.B.weight.data.float() @ module.A.weight.data.float()
            sd[f"{name}.weight".replace(".base", "")] = w.to(module.base.weight.dtype).cpu()
    return sd


def sample_partitions(rng, n):
    """n random partitions of K frames into chunks of {2,3,4}; canonical [3]*7 first."""
    parts = [tuple([3] * 7)]
    while len(parts) < n:
        sizes, left = [], K
        while left > 0:
            s = int(rng.choice([2, 3, 4]))
            s = min(s, left)
            if left - s == 1:  # avoid a trailing 1-frame chunk
                s = left
            sizes.append(s)
            left -= s
        parts.append(tuple(sizes))
    return list(dict.fromkeys(parts))  # dedupe, keep order


def partition_mask(sizes, frame_seqlen, local_attn_size, device):
    """Variable-partition block-causal mask: same semantics as the uniform builder,
    with `ends` filled from cumulative chunk boundaries."""
    total = K * frame_seqlen
    padded = math.ceil(total / 128) * 128 - total
    ends = torch.zeros(total + padded, device=device, dtype=torch.long)
    f0 = 0
    for s in sizes:
        a, b = f0 * frame_seqlen, (f0 + s) * frame_seqlen
        ends[a:b] = b
        f0 += s

    def attention_mask(bb, hh, q_idx, kv_idx):
        if local_attn_size == -1:
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
        return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

    return create_block_mask(attention_mask, B=None, H=None, Q_LEN=total + padded,
                             KV_LEN=total + padded, _compile=False, device=device)


def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    cfg.guidance_scale, cfg.timestep_shift = 6.0, 8.0
    torch.set_grad_enabled(True)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    model = pipe.generator.model
    for p in model.parameters():
        p.requires_grad_(False)
    apply_lora(model, rank=RANK)
    model.gradient_checkpointing = True
    print(f"JITTER adaptation | LoRA r{RANK} = {num_lora_params(model)/1e6:.1f}M | {STEPS} steps", flush=True)
    opt = torch.optim.AdamW(lora_parameters(model), lr=LR)

    d = torch.load(os.path.join(OUT, "synth_clips.pt"), map_location="cpu")
    clips, caps = d["gt"], d["captions"]
    N = clips.shape[0]
    nval = 24
    train_ids, val_ids = list(range(N - nval)), list(range(N - nval, N))
    rng = np.random.default_rng(0)

    frame_seqlen = 60 * 104 // (model.patch_size[1] * model.patch_size[2])
    if os.environ.get("PARTS"):  # fixed partition override, e.g. PARTS=7,7,7 for the larger-chunk arm
        parts = [tuple(int(x) for x in os.environ["PARTS"].split(","))]
        assert sum(parts[0]) == K
    else:
        parts = sample_partitions(np.random.default_rng(1), N_PARTS)
    print(f"partition pool ({len(parts)}): {parts}", flush=True)
    masks = [partition_mask(p, frame_seqlen, model.local_attn_size, DEVICE) for p in parts]

    cond_cache = {}

    def get_cond(c):
        if c not in cond_cache:
            with torch.no_grad():
                cond_cache[c] = pipe.text_encoder(text_prompts=[caps[c]])
        return cond_cache[c]

    def df_loss(c, grad):
        pi = int(rng.integers(len(parts)))
        model.block_mask = masks[pi]
        sizes = torch.tensor(parts[pi], device=DEVICE)
        x0 = clips[c:c + 1].to(DEVICE).to(torch.bfloat16)
        u = torch.from_numpy(rng.random(len(sizes))).float()
        sig_c = (8 * u / (1 + 7 * u)).to(DEVICE)
        sig = sig_c.repeat_interleave(sizes).view(1, -1, 1, 1, 1).to(torch.bfloat16)
        ts = (sig_c * 1000).repeat_interleave(sizes).view(1, -1).to(DEVICE).float()
        eps = torch.randn_like(x0)
        z = (1 - sig) * x0 + sig * eps
        target = (eps - x0).float()
        gctx = torch.enable_grad() if grad else torch.no_grad()
        with gctx:
            flow, _ = pipe.generator(noisy_image_or_video=z, conditional_dict=get_cond(c), timestep=ts)
            return ((flow.float() - target) ** 2).mean()

    @torch.no_grad()
    def val_loss(n=8):
        return float(np.mean([df_loss(int(rng.choice(val_ids)), grad=False).item() for _ in range(n)]))

    START = 1
    _pp = os.path.join(OUT, f"adapt_{TAGP}.partial")
    if os.environ.get("RESUME", "1") == "1" and os.path.exists(_pp):
        _pd = torch.load(_pp, map_location="cpu")
        for p, w in zip(lora_parameters(model), _pd["lora"]):
            p.data.copy_(w.to(p.device, p.dtype))
        START = _pd["step"] + 1
        print(f"resumed from partial step {_pd['step']}", flush=True)

    for step in range(START, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        loss = df_loss(int(rng.choice(train_ids)), grad=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            print(f"step {step:5d} | df-loss {loss.item():.4f} | val {val_loss():.4f}", flush=True)
        if step % 200 == 0:
            torch.save({"lora": [p.detach().cpu() for p in lora_parameters(model)], "step": step}, _pp)
        if step % CKPT_EVERY == 0:
            torch.save({"merged": merged_state(model)}, os.path.join(OUT, f"adapted_base_{TAGP}_{step}.pt"))
            print(f"saved adapted_base_{TAGP}_{step}.pt", flush=True)

    print(f"done | final val {val_loss(16):.4f}", flush=True)


if __name__ == "__main__":
    main()
