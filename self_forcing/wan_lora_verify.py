"""Verify LoRA r_phi on the Wan DiT: scale=0 == baseline; perturbed LoRA changes the rollout."""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters, num_lora_params

DEVICE = "cuda"
torch.set_grad_enabled(False)
cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
pipe.corrector = None
wrapped = apply_lora(pipe.generator.model, rank=16)
print(f"LoRA on {len(wrapped)} linears | trainable params {num_lora_params(pipe.generator.model)/1e6:.2f}M")
pipe.sampling_steps = 6
F = (1 + 3 * 2) if pipe.independent_first_frame else (3 * 3)
prompt = ["A serene mountain lake at sunrise."]

def gen(seed):
    torch.manual_seed(seed)
    noise = torch.randn(1, F, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
    return pipe.inference(noise=noise, text_prompts=prompt, return_latents=True)[1].float()

set_lora_scale(pipe.generator.model, 0.0); lat0 = gen(0)                    # baseline
set_lora_scale(pipe.generator.model, 1.0); lat_zero = gen(0)               # scale=1, B=0 -> == base
for p in lora_parameters(pipe.generator.model):                            # perturb B (and A) -> non-identity
    p.add_(0.02 * torch.randn_like(p))
lat_pert = gen(0)
d_zero = (lat0 - lat_zero).abs().max().item()
d_pert = (lat0 - lat_pert).abs().max().item()
print(f"latents {tuple(lat0.shape)} | scale1/B=0 vs baseline max|Δ|={d_zero:.2e} | perturbed vs baseline max|Δ|={d_pert:.3f}")
print("LoRA PORT OK ✓" if d_zero < 1e-2 and d_pert > 1e-2 else "CHECK ✗")
