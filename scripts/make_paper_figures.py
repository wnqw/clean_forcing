"""Publication figures for the ICLR paper, from cached finals data.

Outputs to iclr2027/figures/:
  quality_vs_time_finals128.{pdf,png} -- money curve (per-frame MUSIQ, mean +/- SEM over 128 prompts)
  premise.{pdf,png}                   -- same weights: single-shot clean vs AR-rollout collapse
"""
import glob

import imageio.v3 as iio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SF = "./self_forcing/wan_cache"
D = f"{SF}/finals128"
OUT = "./figures_out"

CONFIGS = [
    ("ahg",   "history guidance $w{=}1.2$", "#b0b0b0", "--", 1.2),
    ("adfs",  "context noise $\\sigma{=}0.2$", "#808080", "--", 1.2),
    ("abase", "adapted base (host)", "#404040", "-", 1.4),
    ("sfd",   "Self Forcing", "#1f77b4", "-", 1.6),
    ("av2s",  "+ Clean Forcing (zero real)", "#ff7f0e", "-", 2.2),
    ("av2",   "+ Clean Forcing (+real)", "#d62728", "-", 2.2),
]


def money_figure():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.7})
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    t = np.arange(201) / 4.0
    k = np.ones(5) / 5
    for tag, label, color, ls, lw in CONFIGS:
        import re
        files = [f for f in sorted(glob.glob(f"{D}/{tag}_p*.npy")) if re.search(r"_p\d+\.npy$", f)]
        arrs = np.stack([np.load(f) for f in files])
        assert arrs.shape == (128, 201), (tag, arrs.shape)
        m = np.convolve(np.pad(arrs.mean(0), 2, mode="edge"), k, mode="valid")
        sem = arrs.std(0) / np.sqrt(len(arrs))
        ax.plot(t, m, color=color, ls=ls, lw=lw, label=label)
        ax.fill_between(t, m - sem, m + sem, color=color, alpha=0.15, lw=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MUSIQ")
    ax.set_xlim(0, 50)
    ax.set_ylim(28, 74)
    ax.axvspan(0, 10, color="k", alpha=0.05, lw=0)
    ax.text(5, 72.5, "first 20%", ha="center", fontsize=7, color="0.4")
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=7, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/quality_vs_time_finals128.pdf")
    fig.savefig(f"{OUT}/quality_vs_time_finals128.png", dpi=200)


def premise_figure():
    """3 rows: single-shot teacher / AR rollout (p027, collapse unfolding) / AR + corrector (p027).
    Non-uniform timestamps so the collapse is seen in progress rather than already dead."""
    fracs = (0.0, 0.1, 0.2, 0.4, 1.0)
    labels = ("0s", "5s", "10s", "20s", "50s")
    base = iio.imread(f"{D}/abase_p027.mp4")
    corr = iio.imread(f"{D}/av2_p027.mp4")
    single = iio.imread(f"{SF}/single_shot_p027.mp4")
    nb, nc, ns = len(base), len(corr), len(single)
    ar = [base[int(f * (nb - 1))] for f in fracs]
    cr = [corr[int(f * (nc - 1))] for f in fracs]
    ss = [single[int(f * (ns - 1))] for f in (0.0, 0.25, 0.5, 0.75, 1.0)]

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, axes = plt.subplots(3, 5, figsize=(5.6, 2.85), gridspec_kw={"wspace": 0.02, "hspace": 0.06})
    for j in range(5):
        axes[0, j].imshow(ss[j])
        axes[1, j].imshow(ar[j])
        axes[2, j].imshow(cr[j])
        for i in (0, 1, 2):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[2, j].set_xlabel(labels[j], fontsize=7, labelpad=1.5)
    axes[0, 0].set_ylabel("single-shot", fontsize=7)
    axes[1, 0].set_ylabel("AR rollout", fontsize=7)
    axes[2, 0].set_ylabel("AR $+$ corrector", fontsize=7)
    fig.savefig(f"{OUT}/premise.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{OUT}/premise.png", bbox_inches="tight", dpi=200)


def scaling_figure():
    """Host adaptation scaling (same-16, uncorrected base): quality is not saturated at 20K,
    and the 14K->20K late jump shows adaptation is a real lever - yet the 4K host + corrector
    (66.9) beats every point on this curve with a small fraction of the training data."""
    steps = np.array([4, 8, 14, 20])
    musiq = np.array([44.8, 48.1, 48.9, 57.8])
    delta = np.array([11.27, 11.72, 12.73, 9.68])
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.7})
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(steps, musiq, "o-", color="#404040", lw=1.6, ms=4, label="base MUSIQ $\\uparrow$")
    ax.axhline(66.9, color="#ff7f0e", lw=1.4, ls="--",
               label="4K base + corrector (zero real)")
    ax.annotate("corrector on 20K base: 62.3", xy=(19.6, 62.3), fontsize=7, color="#ff7f0e",
                ha="right", xytext=(18.6, 59.8), arrowprops=dict(arrowstyle="-", lw=0.6, color="#ff7f0e"))
    ax.plot([20], [62.3], "s", color="#ff7f0e", ms=4)
    ax2 = ax.twinx()
    ax2.plot(steps, delta, "^-", color="#1f77b4", lw=1.2, ms=4, label="base $\\Delta$-drift $\\downarrow$")
    ax2.set_ylabel("$\\Delta$-drift", color="#1f77b4", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#1f77b4", labelsize=7)
    ax.set_xlabel("causal-adaptation steps (thousands)", fontsize=8)
    ax.set_ylabel("MUSIQ (overall, 50 s)", fontsize=8)
    ax.set_xticks(steps)
    ax.tick_params(labelsize=7)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, loc="lower right")
    fig.savefig(f"{OUT}/adaptation_scaling.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{OUT}/adaptation_scaling.png", bbox_inches="tight", dpi=200)


def trade_figure():
    """Drift-progression trade frontier (reconstructed; labels use public names, no config tags)."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.7})
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    # (anchoring, delta, label, dx, dy, ha)
    frontier = [
        (0.537, 5.17, "$\\lambda{=}0$", -0.004, 0.25, "right"),
        (0.572, 6.02, "$\\lambda{=}0.1$", 0.0, 0.25, "center"),
        (0.538, 4.77, "ours v1", 0.008, -0.05, "left"),
        (0.603, 4.01, "mixed pools", 0.008, 0.15, "left"),
        (0.599, 2.98, "motion refs", 0.008, 0.15, "left"),
        (0.679, 1.65, "ours zero-real ($\\lambda{=}0.5$)", 0.008, 0.15, "left"),
        (0.795, -0.16, "ours $+$real", -0.005, 0.35, "right"),
    ]
    others = [(0.485, 1.08, "Self Forcing", 0.008, 0.15, "left"),
              (0.697, 1.00, "SkyReels-V2", 0.004, -0.45, "left")]
    env = [(0.538, 4.77), (0.599, 2.98), (0.679, 1.65), (0.795, -0.16)]  # lower envelope
    ax.plot([p[0] for p in env], [p[1] for p in env], "--", color="#aaaaaa", lw=1.0, zorder=1)
    for x, y, lab, dx, dy, ha in frontier:
        light = y > 4.5
        ax.plot([x], [y], "o", ms=7, color="#f4a5a5" if light else "#d62728",
                markeredgecolor="#333333", markeredgewidth=0.6, zorder=3)
        ax.annotate(lab, xy=(x, y), xytext=(x + dx, y + dy), fontsize=7.5, ha=ha)
    for x, y, lab, dx, dy, ha in others:
        ax.plot([x], [y], "s", ms=7, color="#1f77b4", markeredgecolor="#333333",
                markeredgewidth=0.6, zorder=3)
        ax.annotate(lab, xy=(x, y), xytext=(x + dx, y + dy), fontsize=7.5, ha=ha)
    ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.set_xlabel("anchoring (late-frame sim. to start)", fontsize=9)
    ax.set_ylabel("$\\Delta$-drift (lower = better)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{OUT}/trade_curve.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{OUT}/trade_curve.png", bbox_inches="tight", dpi=200)


def radar_figure():
    """VBench dimension radar (Table 1 data, min-max normalized per dimension)."""
    dims = ["subject", "background", "aesthetic", "imaging", "smoothness", "dynamic"]
    systems = [
        ("adapted base (host)", "#404040", [65.7, 76.7, 44.9, 47.9, 91.1, 29.7]),
        ("Self Forcing", "#1f77b4", [80.2, 84.4, 60.9, 67.7, 94.9, 39.8]),
        ("SkyReels-V2-DF", "#2ca02c", [87.0, 90.7, 61.1, 62.8, 96.6, 50.8]),
        ("ours (zero real)", "#ff7f0e", [81.0, 86.6, 60.5, 63.9, 89.0, 48.4]),
        ("ours ($+$real)", "#d62728", [80.9, 86.4, 61.4, 64.1, 89.1, 49.2]),
    ]
    vals = np.array([s[2] for s in systems])
    lo, hi = vals.min(0), vals.max(0)
    norm = 0.25 + 0.75 * (vals - lo) / (hi - lo)
    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(4.2, 3.4), subplot_kw={"polar": True})
    for (name, color, _), row in zip(systems, norm):
        r = np.concatenate([row, row[:1]])
        ax.plot(ang_c, r, color=color, lw=1.4, label=name)
        ax.fill(ang_c, r, color=color, alpha=0.06)
    ax.set_xticks(ang)
    ax.set_xticklabels(dims, fontsize=8)
    ax.set_yticks([])
    ax.set_ylim(0, 1.05)
    ax.spines["polar"].set_alpha(0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=7, frameon=False, ncol=2)
    fig.savefig(f"{OUT}/vbench_radar.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{OUT}/vbench_radar.png", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    money_figure()
    premise_figure()
    scaling_figure()
    trade_figure()
    print("figures written to", OUT)
