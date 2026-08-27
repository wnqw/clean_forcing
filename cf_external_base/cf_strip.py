"""Contact strips for eyes-on reads: one row of NCOLS frames per input video, stacked.

Usage: python cf_strip.py out.png video1.mp4 [video2.mp4 ...]
"""
import sys

import imageio.v3 as iio
import numpy as np

NCOLS = 10


def main():
    out, vids = sys.argv[1], sys.argv[2:]
    rows = []
    for v in vids:
        fr = iio.imread(v)
        idx = np.linspace(0, len(fr) - 1, NCOLS).astype(int)
        rows.append(np.concatenate([fr[i] for i in idx], axis=1))
    iio.imwrite(out, np.concatenate(rows, axis=0))
    print(f"strip ({len(vids)} rows x {NCOLS}) -> {out}")


if __name__ == "__main__":
    main()
