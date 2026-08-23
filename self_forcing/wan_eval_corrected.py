"""
Corrected-rollout drift eval on Wan (closed-loop): baseline (LoRA scale 0) vs corrected
(scale 1, 0.5). Tests (a) does the teacher-forcing-trained LoRA transfer to the KV-cache
rollout, (b) drift reduction. Metric: mean |Δsaturation from start| on decoded frames.
Caveat: K<=21 horizon is short (KV-cache eviction ceiling) — drift may be mild.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters

DEVICE = "cuda"
K, NCTX, NCLIPS = 21, 3, 3
SCALES = [0.0, 1.0, 0.5]     # 0 == baseline
OUT = "wan_cache"


def saturation(pix):  # pix (F,3,H,W) in [-1,1] -> mean HSV-sat per frame (F,)
    x = ((pix.float() + 1) / 2).clamp(0, 1)
    cmax, cmin = x.amax(1), x.amin(1)
    return ((cmax - cmin) / (cmax + 1e-6)).mean(dim=(1, 2)).cpu().numpy()


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    model = pipe.generator.model
    apply_lora(model, rank=16)
    sd = torch.load(os.path.join(OUT, "lora_r_phi.pt"), map_location="cpu")["lora"]
    for i, p in enumerate(lora_parameters(model)):
        p.data.copy_(sd[i].to(p.device, p.dtype))
    print(f"loaded LoRA ({len(sd)} tensors)", flush=True)

    d = torch.load(os.path.join(OUT, "pairs.pt"), map_location="cpu")
    gt, caps = d["gt"], d["captions"]
    val_ids = list(range(gt.shape[0] - NCLIPS, gt.shape[0]))
    ctx_early = slice(NCTX, NCTX + 8)

    drift = {s: [] for s in SCALES}
    for c in val_ids:
        gtc = gt[c:c + 1].to(DEVICE).to(torch.bfloat16)
        for s in SCALES:
            set_lora_scale(model, s)
            torch.manual_seed(c)
            noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
            _, lat = pipe.inference(noise=noise, text_prompts=[caps[c]],
                                    initial_latent=gtc[:, :NCTX], return_latents=True)
            pix = pipe.vae.decode_to_pixel(lat[:, :K])[0]            # (F,3,H,W)
            sat = saturation(pix)
            ref = sat[ctx_early].mean()
            drift[s].append(float(np.mean(np.abs(sat[NCTX:] - ref))))
        print(f"clip {c}: " + " ".join(f"s{sc}={np.mean(drift[sc]):.4f}" for sc in SCALES), flush=True)

    print("\n================ Wan corrected-rollout drift ================")
    base = np.mean(drift[0.0])
    for s in SCALES:
        dr = np.mean(drift[s])
        red = "" if s == 0.0 else f"  | drift reduction vs baseline: {100*(1-dr/(base+1e-9)):+.0f}%"
        print(f"scale {s}: mean|Δsat| = {dr:.4f}{red}")


if __name__ == "__main__":
    main()
