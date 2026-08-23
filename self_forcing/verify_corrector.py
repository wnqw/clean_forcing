"""Verify the r_phi port: module shapes/zero-init/grad + pipeline-seam integration."""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.corrector import WanVelocityResidual

# --- 1. module unit test (cheap) ---
corr = WanVelocityResidual().cuda()
x = torch.randn(1, 3, 16, 60, 104, device="cuda")
t = torch.full((1, 3), 500.0, device="cuda")
hist = torch.randn(1, 5, 16, 60, 104, device="cuda")
res = corr(x, t, history=hist)
assert res.shape == x.shape, res.shape
print(f"module: out {tuple(res.shape)} | params {corr.num_params()/1e6:.2f}M | zero-init max|res|={res.abs().max():.2e}")
res.sum().backward()
print("grad flows:", any(p.grad is not None and p.grad.abs().sum() > 0 for p in corr.parameters()))

# --- 2. pipeline seam integration ---
torch.set_grad_enabled(False)
cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device("cuda")).to(dtype=torch.bfloat16).cuda()
pipe.sampling_steps = 6
indep = pipe.independent_first_frame
F = (1 + 3 * 2) if indep else (3 * 3)  # 7 or 9 frames
prompt = ["A serene mountain lake at sunrise."]

def gen(seed):
    torch.manual_seed(seed)
    noise = torch.randn(1, F, 16, 60, 104, device="cuda", dtype=torch.bfloat16)
    return pipe.inference(noise=noise, text_prompts=prompt, return_latents=True)[1].float()

pipe.corrector = None
lat0 = gen(0)
corr2 = WanVelocityResidual().cuda()  # float32; outputs cast to latent dtype
pipe.attach_corrector(corr2, alpha=1.0)
lat_zero = gen(0)
with torch.no_grad():
    for p in corr2.out_conv.parameters():
        p.add_(0.1 * torch.randn_like(p))
lat_pert = gen(0)

d_zero = (lat0 - lat_zero).abs().max().item()
d_pert = (lat0 - lat_pert).abs().max().item()
print(f"\nlatents {tuple(lat0.shape)} | alpha=1 zero-init vs baseline max|Δ|={d_zero:.2e} | perturbed vs baseline max|Δ|={d_pert:.3f}")
print("PORT OK ✓" if d_zero < 1e-2 and d_pert > 1e-2 else "CHECK ✗")
