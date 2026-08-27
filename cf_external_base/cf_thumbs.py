"""Ranked pair thumb sheets for owner qualitative picks.

Ranks prompts by corrector advantage (cfc_late - cfb_late MUSIQ), takes the top N,
and writes sheets of interleaved cfb/cfc contact-strip rows (base above, corrected
below, same prompt) so the owner can pick paper-figure pairs fast.

Usage: python cf_thumbs.py   (writes thumbs/pairsheet_{i}.png + thumbs/ranking.txt)
"""
import os

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROW = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROW, "finals")
OUTD = os.path.join(ROW, "thumbs")
NCOLS, TOPN, PER_SHEET, GUTTER = 10, 12, 4, 300


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def strip(path, lines):
    fr = iio.imread(path)
    idx = np.linspace(0, len(fr) - 1, NCOLS).astype(int)
    row = np.concatenate([fr[i] for i in idx], axis=1)
    gut = Image.new("RGB", (GUTTER, row.shape[0]), (255, 255, 255))
    d = ImageDraw.Draw(gut)
    y = row.shape[0] // 2 - 30 * len(lines)
    for ln in lines:
        d.text((14, y), ln, fill=(0, 0, 0), font=font(44))
        y += 60
    return np.concatenate([np.asarray(gut), row], axis=1)


def main():
    os.makedirs(OUTD, exist_ok=True)
    b = np.load(f"{D}/cfb_scores.npz")
    c = np.load(f"{D}/cfc_scores.npz")
    pb = {int(p): v for p, v in zip(b["pids"], b["musiq"])}
    pc = {int(p): v for p, v in zip(c["pids"], c["musiq"])}
    common = sorted(set(pb) & set(pc))
    ranked = sorted(common, key=lambda p: pc[p] - pb[p], reverse=True)[:TOPN]
    with open(f"{OUTD}/ranking.txt", "w") as f:
        for p in ranked:
            f.write(f"p{p:03d}  cfb_musiq={pb[p]:.1f}  cfc_musiq={pc[p]:.1f}  gap={pc[p]-pb[p]:+.1f}\n")
    for s in range(0, len(ranked), PER_SHEET):
        rows = []
        for p in ranked[s:s + PER_SHEET]:
            rows.append(strip(f"{D}/cfb_p{p:03d}.mp4", [f"p{p:03d} base", f"MUSIQ {pb[p]:.1f}"]))
            rows.append(strip(f"{D}/cfc_p{p:03d}.mp4",
                              [f"p{p:03d} +CF", f"MUSIQ {pc[p]:.1f}", f"gap {pc[p]-pb[p]:+.1f}"]))
        iio.imwrite(f"{OUTD}/pairsheet_{s // PER_SHEET}.png", np.concatenate(rows, axis=0))
    print(f"RESULT thumbs: {len(ranked)} pairs -> {OUTD}/pairsheet_*.png (rows alternate cfb/cfc)", flush=True)


if __name__ == "__main__":
    main()
