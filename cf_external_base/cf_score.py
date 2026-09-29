"""Score a CF-row finals tag: MUSIQ drift Delta (locked finals protocol) + post-hoc
eyeball axes (DINO latesim, lag2s identity, cuts/video, chunk pulse) in one pass.

Protocol matches score_finals_row.py (Delta = MUSIQ first-20% - overall, latent-frame
stride, SEM over prompts) and posthoc_metrics.py (same thresholds/lag), so numbers are
directly comparable to the locked paper rows. Paired latesim excess vs the SF reference
row (sfd_dino_latesim.npy) reported when PIDS align.

Usage: TAG=cfb N=128 python -u cf_score.py   (videos in cf_row/finals, first-N pids)
Saves {tag}_musiq_p{idx}.npy per video + {tag}_scores.npz; prints RESULT line.
"""
import os

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np
import pyiqa
import timm
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
ROW = os.environ.get("CF_ROW", _HERE)
SF_REPO = os.environ.get("SF_REPO", os.path.join(os.path.dirname(_HERE), "self_forcing"))
D = os.environ.get("OUTD", os.path.join(ROW, "finals"))
SF_LATE = os.environ.get("SF_LATE", os.path.join(SF_REPO, "wan_cache/finals128/sfd_dino_latesim.npy"))
TAG = os.environ.get("TAG", "cfb")
N = int(os.environ.get("N", 128))
LAG = 12
DEVICE = "cuda"


@torch.no_grad()
def main():
    musiq = pyiqa.create_metric("musiq", device=torch.device(DEVICE))
    dino = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    pids = list(range(N))
    deltas, overalls, late, lag2, cuts, pulse = [], [], [], [], [], []
    for pi in pids:
        f = f"{D}/{TAG}_p{pi:03d}.mp4"
        # --- MUSIQ per latent frame (stride 4), cached ---
        npy = f.replace(".mp4", "_musiq.npy")
        if os.path.exists(npy):
            pf = np.load(npy)
        else:
            vid = iio.imread(f)[::4]
            scores = []
            for i in range(0, len(vid), 16):
                batch = torch.from_numpy(vid[i:i + 16].copy()).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
                scores.append(musiq(batch).squeeze(-1).float().cpu().numpy())
            pf = np.concatenate(scores).astype(np.float32)
            np.save(npy, pf)
        n20 = max(1, len(pf) // 5)
        deltas.append(pf[:n20].mean() - pf.mean())
        overalls.append(pf.mean())
        # --- posthoc axes at full frame rate ---
        reader = imageio.get_reader(f)
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
        ft = torch.cat(feats)
        ref = F.normalize(ft[:3].mean(0, keepdim=True), dim=-1)
        sims = (ft @ ref.T).squeeze(-1).cpu().numpy()
        late.append(float(np.mean(sims[-max(1, len(sims) // 5):])))
        lag2.append(float((ft[:-2] * ft[2:]).sum(-1).mean()))
        dif = np.array(diffs)
        thr = max(0.10, dif.mean() + 5 * dif.std())
        cuts.append(int((dif > thr).sum()))
        lu = np.array(lum) - np.mean(lum)
        pulse.append(float((lu[:-LAG] * lu[LAG:]).mean() / (lu.var() + 1e-9)))
        print(f"[{TAG}] p{pi:03d} MUSIQ {overalls[-1]:.1f} d {deltas[-1]:+.2f} latesim {late[-1]:.3f} "
              f"lag2s {lag2[-1]:.3f} cuts {cuts[-1]} pulse {pulse[-1]:.2f}", flush=True)
        np.savez(f"{D}/{TAG}_scores.npz", pids=np.array(pids[:len(late)]),
                 musiq=np.array(overalls), delta=np.array(deltas), latesim=np.array(late),
                 lag2s=np.array(lag2), cuts=np.array(cuts), pulse=np.array(pulse))

    deltas, overalls = np.array(deltas), np.array(overalls)
    sem = lambda x: x.std() / np.sqrt(len(x))  # noqa: E731
    line = (f"RESULT {TAG} (n={N}): MUSIQ {overalls.mean():.1f}±{sem(overalls):.2f} | "
            f"Delta {deltas.mean():+.2f}±{sem(deltas):.2f} | latesim {np.mean(late):.3f} | "
            f"lag2s {np.mean(lag2):.3f} | cuts/video {np.mean(cuts):.2f} | pulse-ac{LAG} {np.mean(pulse):.2f}")
    if os.path.exists(SF_LATE):
        sf = np.load(SF_LATE)[pids]
        line += (f" | paired-excess-vs-SF {np.mean(np.array(late) - sf):+.3f} | "
                 f"stuck-where-SF-moves {np.mean((np.array(late) > 0.8) & (sf < 0.6)):.2f}")
    print(line, flush=True)


if __name__ == "__main__":
    main()
