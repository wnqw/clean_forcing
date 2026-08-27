"""Seam-pulse + progression/stagnation metrics (coordinator eval addenda).

Per video, from the full-rate frame stream:
  PULSE  — motion series d[t] = mean|frame[t+1]-frame[t]| (gray, [0,1]); detrended
           periodogram power at the chunk/window cadence bin(s) vs broadband median
           (ratio >~3 = boundary pulsing), plus autocorrelation of d at the cadence lag.
  MOTION — mean d in early/mid/late windows (first/mid/last 20%); late->0 = freezing.
  PROG   — DINO late-vs-early content similarity: mean cosine of late-20%-window
           frames (1 fps) vs the early-20% mean feature; high = looping/no new content.

Usage: TAG=cfb LAGS=12,84 python -u cf_pulse_prog.py vid1.mp4 [vid2.mp4 ...]
LAGS = cadence in *pixel frames* (CF: 12 = 3-latent chunk, 84 = 21-latent window;
SkyReels 24fps: 80 = 97-frame window minus 17 overlap).
"""
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import timm

TAG = os.environ.get("TAG", "vid")
LAGS = [int(x) for x in os.environ.get("LAGS", "12,84").split(",")]


@torch.no_grad()
def main():
    dino = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    agg = {f"pulse_ratio@{L}": [] for L in LAGS}
    agg.update({f"ac@{L}": [] for L in LAGS})
    agg.update({"m_early": [], "m_mid": [], "m_late": [], "prog_late2early": []})

    for f in sys.argv[1:]:
        reader = imageio.get_reader(f)
        diffs, feats, prev = [], [], None
        for i, fr in enumerate(reader):
            g = fr.astype(np.float32).mean(-1) / 255.0
            if prev is not None:
                diffs.append(np.abs(g - prev).mean())
            prev = g
            if i % 16 == 0:
                x = torch.from_numpy(fr.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
                x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
                feats.append(F.normalize(dino((x - mean) / std).float(), dim=-1))
        reader.close()
        d = np.array(diffs)
        T = len(d)
        n20 = max(1, T // 5)
        # --- motion-energy trend ---
        m_e, m_m, m_l = d[:n20].mean(), d[T // 2 - n20 // 2:T // 2 + n20 // 2].mean(), d[-n20:].mean()
        # --- pulse: detrended periodogram + autocorr at cadence ---
        dd = d - np.convolve(d, np.ones(25) / 25, mode="same")  # remove slow trend
        ps = np.abs(np.fft.rfft(dd * np.hanning(T))) ** 2
        freqs = np.fft.rfftfreq(T)
        broadband = np.median(ps[1:])
        line = f"[{TAG}] {os.path.basename(f)} motion e/m/l {m_e:.4f}/{m_m:.4f}/{m_l:.4f}"
        dz = dd - dd.mean()
        for L in LAGS:
            fi = np.argmin(np.abs(freqs - 1.0 / L))  # bin for period-L cadence
            lo, hi = max(1, fi - 1), min(len(ps) - 1, fi + 1)
            p_cad = ps[lo:hi + 1].max()
            ratio = float(p_cad / (broadband + 1e-12))
            ac = float((dz[:-L] * dz[L:]).mean() / (dz.var() + 1e-12)) if T > 2 * L else float("nan")
            agg[f"pulse_ratio@{L}"].append(ratio)
            agg[f"ac@{L}"].append(ac)
            line += f" | pulse@{L}f ratio {ratio:.1f} ac {ac:+.2f}"
        # --- content progression ---
        ft = torch.cat(feats)
        ns = max(1, len(ft) // 5)
        early_ref = F.normalize(ft[:ns].mean(0, keepdim=True), dim=-1)
        prog = float((ft[-ns:] @ early_ref.T).mean())
        line += f" | late2early-sim {prog:.3f}"
        agg["m_early"].append(m_e); agg["m_mid"].append(m_m); agg["m_late"].append(m_l)
        agg["prog_late2early"].append(prog)
        print(line, flush=True)

    res = " | ".join(f"{k} {np.nanmean(v):.4g}" for k, v in agg.items())
    print(f"RESULT {TAG} (n={len(sys.argv)-1}): {res}", flush=True)


if __name__ == "__main__":
    main()
