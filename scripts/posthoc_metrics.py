"""Post-hoc per-video metrics for finals128 configs, one code path for all four eyeball axes:
  latesim   — DINO sim of last fifth to opening (anchoring; matches eval_corrector_subset)
  lag2s     — mean DINO sim between frames 2s apart (identity persistence)
  cuts      — adjacent-frame cut count at full frame rate (mean-abs-diff > max(0.10, mu+5*sigma))
  pulse     — luminance autocorrelation at lag 12 frames (chunk-boundary seam)
Plus paired progression vs the SF reference on the same prompts (paper protocol):
  excess = mean(latesim_cfg - latesim_sf); stuck = frac(cfg > 0.8 & sf < 0.6)

Usage: TAGS=v3b,av2s N=16 python posthoc_metrics.py   (PIDS = range(0,128,128//N))
Saves finals128/{tag}_posthoc.npz; prints RESULT lines.
"""
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import timm

D = "./self_forcing/wan_cache/finals128"
TAGS = os.environ.get("TAGS", "v3b,av2s").split(",")
N = int(os.environ.get("N", 16))
LAG = int(os.environ.get("LAG", 12))  # pulse lag in pixel frames (= 4 * chunk latents)
PIDS = list(range(0, 128, 128 // N))[:N]
DEVICE = "cuda"


@torch.no_grad()
def main():
    dino = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    sf_late = np.load(f"{D}/sfd_dino_latesim.npy")

    for tag in TAGS:
        late, lag2, cuts, pulse = [], [], [], []
        for pi in PIDS:
            reader = imageio.get_reader(f"{D}/{tag}_p{pi:03d}.mp4")
            lum, diffs, feats, prev = [], [], [], None
            for i, fr in enumerate(reader):
                g = fr.astype(np.float32) / 255.0
                lum.append(g.mean())
                if prev is not None:
                    diffs.append(np.abs(g - prev).mean())
                prev = g
                if i % 16 == 0:  # 1 fps for DINO
                    x = torch.from_numpy(g).permute(2, 0, 1)[None].cuda()
                    x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
                    feats.append(F.normalize(dino((x - mean) / std).float(), dim=-1))
            reader.close()
            f = torch.cat(feats)
            ref = F.normalize(f[:3].mean(0, keepdim=True), dim=-1)
            sims = (f @ ref.T).squeeze(-1).cpu().numpy()
            late.append(float(np.mean(sims[-max(1, len(sims) // 5):])))
            lag2.append(float((f[:-2] * f[2:]).sum(-1).mean()))
            dif = np.array(diffs)
            thr = max(0.10, dif.mean() + 5 * dif.std())
            cuts.append(int((dif > thr).sum()))
            lu = np.array(lum) - np.mean(lum)
            pulse.append(float((lu[:-LAG] * lu[LAG:]).mean() / (lu.var() + 1e-9)))
            print(f"[{tag}] p{pi:03d} latesim {late[-1]:.3f} lag2s {lag2[-1]:.3f} cuts {cuts[-1]} pulse {pulse[-1]:.2f}",
                  flush=True)
            np.savez(f"{D}/{tag}_posthoc.npz", pids=np.array(PIDS[:len(late)]), latesim=np.array(late),
                     lag2s=np.array(lag2), cuts=np.array(cuts), pulse=np.array(pulse))
        sf = sf_late[PIDS]
        excess = float(np.mean(np.array(late) - sf))
        stuck = float(np.mean((np.array(late) > 0.8) & (sf < 0.6)))
        print(f"RESULT {tag}: latesim {np.mean(late):.3f} | paired-excess-vs-SF {excess:+.3f} | "
              f"stuck-where-SF-moves {stuck:.2f} | lag2s {np.mean(lag2):.3f} | cuts/video {np.mean(cuts):.2f} | "
              f"pulse-ac{LAG} {np.mean(pulse):.2f}", flush=True)


if __name__ == "__main__":
    main()
