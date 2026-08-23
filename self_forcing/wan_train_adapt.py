"""
Causal adaptation of the base (Option B): Diffusion-Forcing objective on the block-causal model —
per-CHUNK independent noise levels (7 chunks x 3 latent frames), velocity target, block-causal
attention (plain _forward_train path, no clean_x). Trained on the 300 SYNTHETIC bidi-teacher clips
(zero real videos). LoRA rank 64 on self-attn q/k/v/o; after training the deltas are MERGED into
base weights (adapted_base_{step}.pt) so the corrector LoRA can stack cleanly later.
Timestep sampling: u~U(0,1) -> shift-8 transform (BAgger training recipe): sigma = 8u/(1+7u).
Run: python -u wan_train_adapt.py   (STEPS env; ckpt every 2K steps)
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, lora_parameters, num_lora_params, LoRALinear

DEVICE = "cuda"
STEPS = int(os.environ.get("STEPS", 6000))
RANK = 64
LR = 1e-4
WARMUP = 100
CKPT_EVERY = 2000
EVAL_EVERY = 200
NCHUNK, CHUNK = 7, 3
OUT = "wan_cache"


def merged_state(model):
    """Bake LoRA deltas into base weights -> plain state_dict for the adapted base."""
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            w = module.base.weight.data.float() + module.B.weight.data.float() @ module.A.weight.data.float()
            sd[f"{name}.weight".replace(".base", "")] = w.to(module.base.weight.dtype).cpu()
    return sd


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
    print(f"DF adaptation | LoRA r{RANK} = {num_lora_params(model)/1e6:.1f}M | {STEPS} steps | shift-8 t-sampling", flush=True)
    opt = torch.optim.AdamW(lora_parameters(model), lr=LR)

    d = torch.load(os.path.join(OUT, "synth_clips.pt"), map_location="cpu")
    clips, caps = d["gt"], d["captions"]
    N = clips.shape[0]
    nval = 24
    train_ids, val_ids = list(range(N - nval)), list(range(N - nval, N))
    rng = np.random.default_rng(0)
    with torch.no_grad():
        cond_cache = {}

    def get_cond(c):
        if c not in cond_cache:
            with torch.no_grad():
                cond_cache[c] = pipe.text_encoder(text_prompts=[caps[c]])
        return cond_cache[c]

    def df_loss(c, grad):
        x0 = clips[c:c + 1].to(DEVICE).to(torch.bfloat16)               # (1,21,16,60,104)
        u = torch.from_numpy(rng.random(NCHUNK)).float()
        sig_c = (8 * u / (1 + 7 * u)).to(DEVICE)                        # shift-8 per-chunk sigma
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

    for step in range(1, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        loss = df_loss(int(rng.choice(train_ids)), grad=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            print(f"step {step:5d} | df-loss {loss.item():.4f} | val {val_loss():.4f}", flush=True)
        if step % CKPT_EVERY == 0:
            torch.save({"merged": merged_state(model)}, os.path.join(OUT, f"adapted_base_{step}.pt"))
            print(f"saved adapted_base_{step}.pt", flush=True)

    print(f"done | final val {val_loss(16):.4f}", flush=True)


if __name__ == "__main__":
    main()
