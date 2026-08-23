"""
Verify the rolling KV-cache fix: self-rollout to K=48 latent frames (>> old 21-frame ceiling).
Confirms (a) no eviction crash, (b) no NaN/Inf, (c) prints the per-frame saturation trajectory
so we can see whether a long-horizon drift curve is now measurable.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline

DEVICE = "cuda"
K, NCTX = 48, 3
OUT = "wan_cache"


def saturation(pix):  # pix (F,3,H,W) in [-1,1] -> mean HSV-sat per frame (F,)
    x = ((pix.float() + 1) / 2).clamp(0, 1)
    cmax, cmin = x.amax(1), x.amin(1)
    return ((cmax - cmin) / (cmax + 1e-6)).mean(dim=(1, 2)).cpu().numpy()


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    print(f"model.local_attn_size={pipe.generator.model.local_attn_size} "
          f"sink_size={pipe.generator.model.blocks[0].self_attn.sink_size} "
          f"max_attn_size={pipe.generator.model.blocks[0].self_attn.max_attention_size} "
          f"kv_cache alloc={pipe.local_attn_size * pipe.frame_seq_length}", flush=True)

    d = torch.load(os.path.join(OUT, "pairs.pt"), map_location="cpu")
    gt, caps = d["gt"], d["captions"]
    c = 0
    gtc = gt[c:c + 1].to(DEVICE).to(torch.bfloat16)
    torch.manual_seed(c)
    noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
    _, lat = pipe.inference(noise=noise, text_prompts=[caps[c]],
                            initial_latent=gtc[:, :NCTX], return_latents=True)
    lat = lat[:, :K]
    print(f"rollout ok: latents {tuple(lat.shape)} | "
          f"finite={torch.isfinite(lat).all().item()} "
          f"min={lat.float().min():.2f} max={lat.float().max():.2f}", flush=True)

    pix = pipe.vae.decode_to_pixel(lat)[0]          # (F,3,H,W)
    sat = saturation(pix)
    ref = sat[NCTX:NCTX + 8].mean()
    print("per-frame |Δsat from start|:")
    for f in range(NCTX, K, 3):
        print(f"  frame {f:2d}: sat={sat[f]:.4f}  |Δ|={abs(sat[f]-ref):.4f}")
    late = np.mean(np.abs(sat[K - 12:] - ref))
    early = np.mean(np.abs(sat[NCTX:NCTX + 12] - ref))
    print(f"\nearly(|Δsat| frames {NCTX}-{NCTX+12})={early:.4f}  late(last 12)={late:.4f}  "
          f"-> {'drift grows with horizon' if late > early else 'flat'}", flush=True)


if __name__ == "__main__":
    main()
