"""Subset closed-loop eval for a corrector variant: N finals prompts x 50s on the adapted base,
reports MUSIQ overall, Delta-drift, and DINO late-sim (progression). For the contraction-sweep arms.

Usage: LORA=wan_cache/lora_r_phi_v2s_cw0.pt TAG=cw0 N=16 python eval_corrector_subset.py
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import timm
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

SF = "./self_forcing"
import sys

sys.path.insert(0, SF)
os.chdir(SF)
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline  # noqa: E402
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters  # noqa: E402

torch.set_num_threads(8)
DEVICE = "cuda"
KLAT = int(os.environ.get("KLAT", 201))
NFB = int(os.environ.get("NFB", 0))  # override num_frame_per_block (larger-chunk arm)
N = int(os.environ.get("N", 16))
LORA = os.environ["LORA"]
TAG = os.environ["TAG"]
SAVE = os.environ.get("SAVE", "1") == "1"  # save videos as finals128/{TAG}_p{idx}.mp4 for eyeball/post-hoc metrics
D = f"{SF}/wan_cache/finals128"
PIDS = list(range(0, 128, 128 // N))[:N]


@torch.no_grad()
def main():
    import pyiqa
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    if NFB:
        cfg.num_frame_per_block = NFB
    torch.set_grad_enabled(False)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    pipe.overlap_blend = int(os.environ.get("OVERLAP", "0"))  # seam-pulse fix (a): 1=pin, 2=regen+blend; +1/3 NFE
    _base = os.environ.get("ADAPTED_BASE", "wan_cache/adapted_base_4000.pt")
    sd = torch.load(_base, map_location="cpu")["merged"]
    pipe.generator.model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
    model = pipe.generator.model
    if LORA != "none":
        apply_lora(model, rank=16)
        lw = torch.load(LORA, map_location="cpu")["lora"]
        lw = list(lw.values()) if isinstance(lw, dict) else lw
        for p, w in zip(lora_parameters(model), lw):
            p.data.copy_(w.to(p.device, p.dtype))
        set_lora_scale(model, 1.0)

    prompts = [ln.strip() for ln in open(f"{D}/prompts_used.txt") if ln.strip()]
    suffix = os.environ.get("SUFFIX", "")
    if suffix:
        prompts = [p + " " + suffix for p in prompts]
    musiq = pyiqa.create_metric("musiq", device=DEVICE)
    dino = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).cuda().eval()
    NORM = (torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(),
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())

    deltas, overalls, latesims = [], [], []
    for pi in PIDS:
        torch.manual_seed(pi)
        noise = torch.randn(1, KLAT, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
        video = pipe.inference(noise=noise, text_prompts=[prompts[pi]])
        if SAVE:
            import imageio.v2 as imageio
            fr = (video[0].permute(0, 2, 3, 1).float() * 255).byte().cpu().numpy()
            imageio.mimsave(f"{D}/{TAG}_p{pi:03d}.mp4", fr, fps=16, quality=8)
        xs = video[0][::4].float()
        pf = torch.cat([musiq(xs[i:i + 16]).flatten() for i in range(0, len(xs), 16)]).cpu().numpy()
        deltas.append(pf[:len(pf) // 5].mean() - pf.mean())
        overalls.append(pf.mean())
        xd = video[0][::16].float()
        xd = F.interpolate(xd, size=(224, 224), mode="bicubic", align_corners=False)
        xd = (xd - NORM[0]) / NORM[1]
        f = F.normalize(dino(xd).float(), dim=-1)
        ref = F.normalize(f[:3].mean(0, keepdim=True), dim=-1)
        sims = (f @ ref.T).squeeze(-1).cpu().numpy()
        latesims.append(float(np.mean(sims[-max(1, len(sims) // 5):])))
        print(f"[{TAG}] p{pi:03d} musiq {pf.mean():.1f} delta {deltas[-1]:+.2f} latesim {latesims[-1]:.3f}", flush=True)

    print(f"RESULT {TAG}: MUSIQ {np.mean(overalls):.1f} | Delta {np.mean(deltas):+.2f}±{np.std(deltas)/np.sqrt(N):.2f} | "
          f"DINO-latesim {np.mean(latesims):.3f}", flush=True)


if __name__ == "__main__":
    main()
