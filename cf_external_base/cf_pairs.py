"""CF-row drift pairs (prompts-only regime), all-300 synth recipe.

h_clean = wan_cache/synth_clips.pt (bidi-Wan 21-latent clips, shared clean manifold);
h_gen = K=21 CF-base rollout seeded from each clip's first NCTX latents (seed = clip idx,
identical to cf_build_pairs.py, so the feasibility 120-pair file resumes cleanly).
CORRECTOR env = LoRA ckpt -> DAgger-round rollouts.
Resumable. Run from Self-Forcing repo:  NCLIPS=300 OUT=... python -u cf_pairs.py
"""
import os

import torch

from cf_common import ROW, CLIPS, DEVICE, K, NCTX, load_cf_pipe, load_lora

OUT = os.environ.get("OUT", os.path.join(ROW, "pairs_synth_cf300.pt"))
CORRECTOR = os.environ.get("CORRECTOR")
NCLIPS = int(os.environ.get("NCLIPS", 300))


@torch.no_grad()
def main():
    torch.set_grad_enabled(False)
    pipe = load_cf_pipe()
    if CORRECTOR:
        load_lora(pipe.generator.model, CORRECTOR, scale=1.0)
        print("DAgger round: rollouts from CF base + corrector", flush=True)

    d = torch.load(CLIPS, map_location="cpu")
    gt_all, caps = d["gt"][:NCLIPS], list(d["captions"])[:NCLIPS]
    N = gt_all.shape[0]

    gens, done = [], 0
    if os.path.exists(OUT):
        prev = torch.load(OUT, map_location="cpu")
        gens = list(prev["gen"].unbind(0)); done = len(gens)
        print(f"resuming from {done}", flush=True)

    for c in range(done, N):
        gt = gt_all[c:c + 1].to(DEVICE).to(torch.bfloat16)
        torch.manual_seed(c)
        noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
        _, gen = pipe.inference(noise=noise, text_prompts=[caps[c]],
                                initial_latent=gt[:, :NCTX], return_latents=True)
        gens.append(gen[0, :K].half().cpu())
        if (c + 1) % 20 == 0 or c == N - 1:
            torch.save({"gt": gt_all[:len(gens)], "gen": torch.stack(gens),
                        "captions": caps[:len(gens)]}, OUT)
            print(f"{c + 1}/{N} pairs saved", flush=True)

    print(f"done: {len(gens)} CF pairs -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
