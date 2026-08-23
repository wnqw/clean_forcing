"""Score any finals config row: per-frame MUSIQ at latent-frame stride -> npy, Delta-drift + CI.

Protocol matches the locked finals rows: Delta = MUSIQ(first 20%) - MUSIQ(overall) per BAgger;
CI = SEM over 128 prompts. Works for any tag ({tag}_p{idx:03d}.mp4 in finals128/), any fps:
one pixel frame per latent frame (stride 4).

Usage: python score_finals_row.py --tag attc
"""
import argparse
import glob
import os

import imageio.v3 as iio
import numpy as np
import pyiqa
import torch

D = "./self_forcing/wan_cache/finals128"
DEVICE = torch.device("cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    musiq = pyiqa.create_metric("musiq", device=DEVICE)
    files = sorted(glob.glob(f"{D}/{args.tag}_p*.mp4"))
    assert len(files) == 128, f"expected 128 videos, got {len(files)}"

    deltas, overalls = [], []
    for f in files:
        npy = f.replace(".mp4", ".npy")
        if os.path.exists(npy):
            pf = np.load(npy)
        else:
            vid = iio.imread(f)[::4]
            scores = []
            for i in range(0, len(vid), 16):
                batch = torch.from_numpy(vid[i:i + 16]).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
                with torch.no_grad():
                    scores.append(musiq(batch).squeeze(-1).float().cpu().numpy())
            pf = np.concatenate(scores).astype(np.float32)
            np.save(npy, pf)
        n20 = max(1, len(pf) // 5)
        deltas.append(pf[:n20].mean() - pf.mean())
        overalls.append(pf.mean())
        print(f"SCORED {os.path.basename(f)} overall {pf.mean():.1f} delta {deltas[-1]:+.2f}", flush=True)

    deltas, overalls = np.array(deltas), np.array(overalls)
    sem = lambda x: x.std() / np.sqrt(len(x))  # noqa: E731
    print(f"RESULT {args.tag}: MUSIQ {overalls.mean():.1f}±{sem(overalls):.2f} | "
          f"Delta {deltas.mean():+.2f}±{sem(deltas):.2f}", flush=True)


if __name__ == "__main__":
    main()
