"""Post-hoc temporal luminance deflicker for the chunk-boundary pulse (cosmetic, labeled
post-process — scored metrics stay on raw videos).

Per frame: scale luma so its mean matches the centered moving median (window 7), gain
clipped to [0.92, 1.08] so real luminance changes (day/night, shadows) pass through.
Prints pulse-ac12 before/after. Usage: IN=path.mp4 [OUT=path] python seam_deflicker.py
"""
import os

import imageio.v2 as imageio
import numpy as np

IN = os.environ["IN"]
OUT = os.environ.get("OUT", IN.replace(".mp4", "_deflick.mp4"))
WIN = int(os.environ.get("WIN", 7))
CLIP = float(os.environ.get("CLIP", 0.08))


def pulse_ac12(lum):
    lu = np.asarray(lum) - np.mean(lum)
    return float((lu[:-12] * lu[12:]).mean() / (lu.var() + 1e-9))


def main():
    frames = [f.astype(np.float32) / 255.0 for f in imageio.mimread(IN, memtest=False)]
    # per-channel mean and std: the wash is contrast/saturation loss, not just brightness
    mu = np.array([f.reshape(-1, 3).mean(0) for f in frames])          # [T, 3]
    sd = np.array([f.reshape(-1, 3).std(0) for f in frames])           # [T, 3]
    half = WIN // 2
    med_mu = np.array([np.median(mu[max(0, i - half):i + half + 1], 0) for i in range(len(mu))])
    med_sd = np.array([np.median(sd[max(0, i - half):i + half + 1], 0) for i in range(len(sd))])
    g_sd = np.clip(med_sd / (sd + 1e-9), 1 - CLIP, 1 + CLIP)
    out = []
    for f, m, gm, gs in zip(frames, mu, med_mu, g_sd):
        shift = np.clip(gm - m, -CLIP, CLIP)
        out.append(np.clip((f - m) * gs + m + shift, 0, 1))
    imageio.mimsave(OUT, [(f * 255).astype(np.uint8) for f in out], fps=16, quality=8)
    lum = mu.mean(1)
    print(f"pulse-ac12 before {pulse_ac12(lum):.3f} | after {pulse_ac12([f.mean() for f in out]):.3f} "
          f"| mean|std-gain-1| {np.abs(g_sd - 1).mean():.4f} | saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
