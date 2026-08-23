"""Issue-2 arm 1: filter training references by their OWN identity persistence.

Decodes each synthetic reference clip, scores DINO lag-2s persistence, keeps the top
KEEP fraction, and writes filtered copies of the training pools (same keys, subset rows).
The corrector's identity cost is data-bound; morph-y references teach morphing.

Run from Self-Forcing: python filter_refs_identity.py -> pairs_synth_adapt_idfilt.pt etc.
"""
import os

import numpy as np
import timm
import torch
import torch.nn.functional as F

SF = "./self_forcing"
import sys

sys.path.insert(0, SF)
os.chdir(SF)
from utils.wan_wrapper import WanVAEWrapper  # noqa: E402

DEVICE = "cuda"
KEEP = float(os.environ.get("KEEP", 0.6))
POOLS = ["pairs_synth_adapt.pt", "pairs_synth_dagger_adapt.pt"]


@torch.no_grad()
def main():
    vae = WanVAEWrapper().to(DEVICE, torch.bfloat16)
    dino = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    d = torch.load("wan_cache/synth_clips.pt", map_location="cpu")
    clips = d["gt"]
    pers = []
    for i in range(clips.shape[0]):
        lat = clips[i:i + 1].to(DEVICE, torch.bfloat16)
        px = vae.decode_to_pixel(lat)[0]  # [T, 3, H, W] in [-1, 1]
        px = (px * 0.5 + 0.5).clamp(0, 1)[::16].float()  # ~1 fps
        x = F.interpolate(px, size=(224, 224), mode="bicubic", align_corners=False)
        f = F.normalize(dino((x - mean) / std).float(), dim=-1)
        pers.append(float((f[:-2] * f[2:]).sum(-1).mean()))
        if i % 50 == 0:
            print(f"clip {i} lag2s {pers[-1]:.3f}", flush=True)
    pers = np.array(pers)
    thr = np.quantile(pers, 1 - KEEP)
    keep = np.where(pers >= thr)[0]
    print(f"persistence mean {pers.mean():.3f} | thr {thr:.3f} | keep {len(keep)}/{len(pers)}", flush=True)
    np.save("wan_cache/ref_persistence.npy", pers)

    for pool in POOLS:
        p = torch.load(f"wan_cache/{pool}", map_location="cpu")
        n = p["gt"].shape[0]
        sel = keep[keep < n]
        out = {}
        for k, v in p.items():
            if isinstance(v, torch.Tensor) and v.shape[:1] == (n,):
                out[k] = v[torch.from_numpy(sel)]
            elif isinstance(v, list) and len(v) == n:
                out[k] = [v[j] for j in sel]
            else:
                out[k] = v
        name = pool.replace(".pt", "_idfilt.pt")
        torch.save(out, f"wan_cache/{name}")
        print(f"{name}: {len(sel)}/{n} rows", flush=True)


if __name__ == "__main__":
    main()
