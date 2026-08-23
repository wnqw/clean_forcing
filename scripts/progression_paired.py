"""Intent-conditioned progression analysis: paired per-prompt DINO late-sim vs the SF reference.

Stickiness = staying anchored where SF progresses ON THE SAME PROMPT. Static-intent prompts
(where SF also stays put) do not count against a config. Two outputs per config:
  paired excess = mean(latesim_cfg - latesim_sf) over prompts        (0 = matches SF's profile)
  stuck-where-SF-moves = frac(latesim_cfg > 0.8 AND latesim_sf < 0.6)
Plus a text-cue stratification (motion-implying vs static-implying prompts).

Inputs: finals128/{tag}_dino_latesim.npy (per-video, prompt order) + prompts_used.txt.
"""
import re

import numpy as np

D = "./self_forcing/wan_cache/finals128"
TAGS = ["av1s", "av2s", "av2", "skyr"]
REF = "sfd"

MOTION_CUES = re.compile(
    r"\b(pan|pans|panning|track|tracking|dolly|arc shot|aerial|drone|fly|flying|zoom|"
    r"travel|traveling|moving camera|camera moves|walks through|drives|riding|journey|glides)\b", re.I)


def main():
    prompts = [ln.strip() for ln in open(f"{D}/prompts_used.txt") if ln.strip()]
    ref = np.load(f"{D}/{REF}_dino_latesim.npy")
    motion_mask = np.array([bool(MOTION_CUES.search(p)) for p in prompts])
    print(f"prompts with explicit motion cues: {motion_mask.sum()}/128")
    print(f"{REF} late-sim: overall {ref.mean():.3f} | motion-cue {ref[motion_mask].mean():.3f} "
          f"| no-cue {ref[~motion_mask].mean():.3f}\n")

    header = f"{'config':7s} {'paired excess':>14s} {'stuck-where-SF-moves':>21s} {'excess(motion)':>15s} {'excess(no-cue)':>15s}"
    print(header)
    for tag in TAGS:
        v = np.load(f"{D}/{tag}_dino_latesim.npy")
        excess = v - ref
        stuck = float(np.mean((v > 0.8) & (ref < 0.6)))
        print(f"{tag:7s} {excess.mean():+14.3f} {stuck:21.2%} "
              f"{excess[motion_mask].mean():+15.3f} {excess[~motion_mask].mean():+15.3f}")


if __name__ == "__main__":
    main()
