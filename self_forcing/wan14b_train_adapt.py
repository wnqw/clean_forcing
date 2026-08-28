"""Stage 2 of the 14B scale-up: causal adaptation — port of wan_train_adapt.py to Wan2.1-T2V-14B.

Mirrors the 1.3B recipe exactly: DF objective on the block-causal model (7 chunks x 3 latents,
per-chunk independent shift-8 sigma, velocity target, plain _forward_train path), LoRA rank 64 on
self-attn q/k/v/o, LR 1e-4, warmup 100, grad-clip 1.0, 6K steps, merged ckpts every 2K.
Data: the 300 synthetic bidi-teacher clips (wan_cache/wan14b_refs.pt — zero real videos).
14B deltas only: pipeline/config from wan14b_common; partial (LoRA+step) resume ckpt every 200
steps per the scale-up checkpointing rule. Run from repo root (STEPS env; RESUME=1 default).
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import numpy as np
import torch
from wan.modules.lora import apply_lora, lora_parameters, num_lora_params, LoRALinear
from wan14b_common import CausalDiffusionInferencePipeline14B, load_cfg, write_timing

DEVICE = "cuda"
STEPS = int(os.environ.get("STEPS", 6000))
RANK = 64
LR = 1e-4
WARMUP = 100
CKPT_EVERY = 2000
EVAL_EVERY = 200
NCHUNK, CHUNK = 7, 3
OUT = "wan_cache"
CLIPS = "wan_cache/wan14b_refs.pt"
PARTIAL = os.path.join(OUT, "wan14b_adapt.partial")


def merged_state(model):
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            w = module.base.weight.data.float() + module.B.weight.data.float() @ module.A.weight.data.float()
            sd[f"{name}.weight".replace(".base", "")] = w.to(module.base.weight.dtype).cpu()
    return sd


def main():
    cfg = load_cfg()
    torch.set_grad_enabled(True)
    pipe = CausalDiffusionInferencePipeline14B(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    model = pipe.generator.model
    for p in model.parameters():
        p.requires_grad_(False)
    apply_lora(model, rank=RANK)
    model.gradient_checkpointing = True
    print(f"14B DF adaptation | LoRA r{RANK} = {num_lora_params(model)/1e6:.1f}M | {STEPS} steps "
          f"| shift-8 t-sampling", flush=True)
    opt = torch.optim.AdamW(lora_parameters(model), lr=LR)
    START_STEP = 1
    if os.environ.get("RESUME", "1") == "1" and os.path.exists(PARTIAL):
        pd = torch.load(PARTIAL, map_location="cpu")
        for p, w in zip(lora_parameters(model), pd["lora"]):
            p.data.copy_(w.to(p.device, p.dtype))
        START_STEP = pd["step"] + 1
        print(f"resumed from partial step {pd['step']}", flush=True)

    d = torch.load(CLIPS, map_location="cpu")
    clips, caps = d["gt"], d["captions"]
    N = clips.shape[0]
    assert N >= 100, f"need the full ref set before adaptation (got {N})"
    nval = 24
    train_ids, val_ids = list(range(N - nval)), list(range(N - nval, N))
    rng = np.random.default_rng(0)
    cond_cache = {}

    def get_cond(c):
        if c not in cond_cache:
            with torch.no_grad():
                cond_cache[c] = pipe.text_encoder(text_prompts=[caps[c]])
        return cond_cache[c]

    def df_loss(c, grad):
        x0 = clips[c:c + 1].to(DEVICE).to(torch.bfloat16)
        u = torch.from_numpy(rng.random(NCHUNK)).float()
        sig_c = (8 * u / (1 + 7 * u)).to(DEVICE)
        sig = sig_c.repeat_interleave(CHUNK).view(1, -1, 1, 1, 1).to(torch.bfloat16)
        ts = (sig_c * 1000).repeat_interleave(CHUNK).view(1, -1).to(DEVICE).float()
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

    t0 = time.time()
    for step in range(START_STEP, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        loss = df_loss(int(rng.choice(train_ids)), grad=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
        opt.step()
        if step % 200 == 0:
            torch.save({"lora": [p.detach().cpu() for p in lora_parameters(model)], "step": step}, PARTIAL)
        if step % EVAL_EVERY == 0 or step == 1:
            print(f"step {step:5d} | df-loss {loss.item():.4f} | val {val_loss():.4f} "
                  f"| {(time.time() - t0) / max(step - START_STEP + 1, 1):.1f}s/step", flush=True)
        if step % CKPT_EVERY == 0:
            torch.save({"merged": merged_state(model)}, os.path.join(OUT, f"wan14b_adapted_base_{step}.pt"))
            print(f"saved wan14b_adapted_base_{step}.pt", flush=True)

    write_timing("adapt", time.time() - t0, STEPS - START_STEP + 1, {"rank": RANK, "steps": STEPS})
    print(f"done | final val {val_loss(16):.4f}", flush=True)


if __name__ == "__main__":
    main()
