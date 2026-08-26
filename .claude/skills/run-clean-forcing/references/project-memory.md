# Distilled project memory (sanitized for release; technical findings only)

Hard-won empirical knowledge from the original development run. Trust these as verified
starting points; re-verify on your own hardware before publishing new numbers.

## Training the corrector (what actually makes it converge)
- The raw counterfactual MSE DIVERGES (initial grad-norm ~400). Required recipe: gap-normalized
  loss (divide by ||v_clean - v_gen||^2) + grad-clip 1.0 + linear warmup (~60 steps) + AdamW
  5e-4 (one-step) / 2e-4 (closed-loop). With it, one-step fits R^2 ~0.4-0.55 (up to 0.97 on
  narrow in-domain data).
- LoRA B must be zero-initialized (exact identity start); A ~ N(0, 1/r).
- Closed-loop init FROM the one-step checkpoint; contraction weight 0.5; commit x0-hat WITH
  gradient into the next chunk's history.
- Expect one-step alone to trade collapse for BLUR; the closed-loop stage must recover
  sharpness. Always sanity a 30 s rollout between stages.
- Rank: fit saturates by r8; rollout quality peaks at r16; r32/r64 measurably WORSE closed-loop.
- Training eps sampling is unseeded: each retrain is one training-noise draw.

## The gate (measure before training, always)
- alpha*(t) = ||E delta||^2 / (||E delta||^2 + Var delta) over noise seeds, per solver timestep,
  at states from cached drifted rollouts. Green hosts: flat 0.9+ with a large gap
  (rel. gap ~0.2). Measured examples: 1.3B host 0.958, 14B 0.934/0.961, Causal-Forcing base
  0.957 — all corrected successfully. Self-Forcing-distilled: 0.65-0.80 AND ~10x smaller gap —
  corrector DAMAGED that host (69.3 -> 47.5 MUSIQ). Rule: expected benefit ~ alpha* x gap.
- Anchoring matters: supervise at the rollout's own z_t (native anchoring). GT-anchored
  targets compare two different worlds and INJECT drift.

## Evaluation traps (each one bit us)
- PSNR/LPIPS-vs-ground-truth is NOT a valid drift metric (penalizes legitimate alternate
  futures). Drift here = low-frequency color/saturation creep + texture death.
- Delta-drift (first-20% minus overall MUSIQ) SATURATES at long horizons — the collapsed tail
  drags the mean; always pair with absolute MUSIQ.
- Smoothness/flicker-flavored metrics REWARD degradation (frozen or blurred videos win them);
  guard with GT-anchored dynamic degree and eyes-on strips for every batch.
- VBench on custom prompts: only 8 of 16 dims are computable (6 need suite-labeled prompts;
  human_action parses the filename; temporal_style duplicates overall_consistency).
- bf16 chaotic recurrence: 48-frame rollouts do not reproduce bitwise across torch/driver
  stacks. Reproduce paired statistics; re-run a known cell as a kill bar first.

## Teacher/objective findings
- Bidirectional single-shot generations from the SAME checkpoint work as clean references —
  the entire stack can train with zero real videos; 16->40->300 clips already saturates the
  drift metric (the residual is ~95% systematic, so sample complexity is tiny).
- GT-future-informed teachers -> blur (unpredictable future = conditional-mean target);
  model-predicted future -> re-anchors. Deterministic past-only correctors buy progression
  only as blur (paper App. E).
- Larger-chunk adaptation (7 vs 3 latents) strengthens the HOST markedly at equal budget but
  does NOT remove drift; the chunk-cadence pulse is cadence-invariant (protocol-level).

## Known corrected-rollout artifacts (documented in the paper, not bugs)
- Chunk-cadence luminance pulse (mitigation knob: OVERLAP=1|2 decode, +1/3 NFE).
- Progression anchoring (class property of reference-anchored objectives).
