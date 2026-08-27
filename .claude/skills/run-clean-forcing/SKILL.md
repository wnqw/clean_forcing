---
name: run-clean-forcing
description: Operate the Clean Forcing repo end-to-end — env setup, demo generation, corrector training, paper-protocol evaluation, and user-study reproduction — after the original author's departure. Use whenever asked to run, evaluate, retrain, or extend Clean Forcing / the drift-corrector LoRAs in this repo.
---

# Running Clean Forcing (agent runbook)

You are operating a research codebase whose author has left. Everything needed is in this repo
plus public downloads; the human you assist is a coauthor (e.g., Luke). Ground rules first,
then procedures.

## Ground rules (inherited from the project, do not relax)
1. **Phantom-numbers rule**: never report a number you cannot point to in a file on disk you
   just produced or verified. Every paper number's provenance is in `paper_tables.md`.
2. **Method name**: "Clean Forcing" — never abbreviate, never "Counterfactual Forcing" (dead name).
   Corrector stages are "one-step" and "closed-loop" (code may say v1/v2 — same objects).
3. **Protocol footnote**: ALL rollouts here use a rolling 21-frame KV window + 20-step UniPC
   (832x480@16fps, 3-latent chunks). External bases (Causal-Forcing: 50-step default) are run
   under OUR protocol — any number you produce must carry that footnote.
4. **Gate before training on a new host**: measure alpha*(t) x gap first (see paper App. F).
   Distilled/consistency-trained hosts (Self-Forcing-like) are the measured do-not-correct
   regime — a corrector can DAMAGE them. Do not skip this because training "seems to work."
5. Comparing VBench numbers across papers at different video durations is meaningless;
   compare only same-protocol rows.

## Environment
- One A100/H100-class GPU, >=48 GB, bf16. `bash SETUP.sh` creates the env and downloads the
  public Wan2.1-T2V-1.3B base. Torch >=2.4/cu124 is fine on standard NVIDIA GPUs.
- Run every script FROM THE REPO ROOT (scripts chdir into self_forcing/ themselves).
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` warnings are expected; not a bug.

## Artifact map
- `weights/` — 13+ released corrector LoRAs + SHA256SUMS. The checkpoint->base map is in
  `RUN.md` section 4. Two fully-public-runnable rows: `lora_cf_v1.pt` / `lora_cf_v2_both.pt`
  on the Causal-Forcing base (`hf download zhuhz22/Causal-Forcing`, load `ckpt["generator"]`).
- The 1.3B adapted base (`adapted_base_4000.pt`, ~2.8 GB) is NOT in git: fetch from the
  evacuation bundle link in `RUN.md` section 2, or reproduce in ~1 GPU-day (README "Reproduce"
  steps 1-2; zero real videos needed).
- `cf_external_base/` — complete external-base runbook (Causal Forcing): pair build,
  both training stages, finals, scoring, and the exact orchestrator scripts used; its
  README carries the result summary and protocol footnotes.
- `prompts_finals128.txt` — the exact locked evaluation prompts (seed = prompt index).
- `user_study/` — manifest + anonymized responses; `python user_study/analysis.py --responses
  user_study/responses_anonymized.csv` reproduces the paper's preference table exactly.

## Procedures
- **Quick demo (5 min)**: `LORA=weights/lora_r_phi_v2_both_adapt.pt PROMPT="..." python
  demo_generate.py`; rerun with `LORA=none` on the same SEED to see the base collapse.
  Rank-sweep ckpts need matching `RANK=8|32|64`.
- **Paper-protocol eval**: README "Reproduce" step 6 — `scripts/eval_corrector_subset.py`
  (MUSIQ + Delta-drift + videos), then `scripts/score_official_6dim.py` and
  `scripts/score_official_semantic.py` (official VBench), `scripts/posthoc_metrics.py`
  (anchoring/lag-identity/cuts). Delta-drift saturates at long horizons — always report
  absolute MUSIQ alongside it.
- **Retrain a corrector** (new data/host): README "Reproduce" steps 3-5. One-step first
  (~800-1500 steps, LR 5e-4, gap-normalized loss; expect val R^2 ~0.4-0.55), then DAgger
  round-1 pairs with the one-step corrector active, then closed-loop (600 steps, LR 2e-4,
  contraction weight 0.5). Sanity-check with a 30 s rollout BEFORE long evals: one-step alone
  typically trades collapse for blur; closed-loop must recover sharpness.
- **Long jobs** (>10 min): launch with `setsid nohup ... > log 2>&1 &`. CAUTION: setsid
  re-forks — the echoed PID is stale; track jobs with `pgrep -f <script_name>`, never by PID.

## Known artifacts you will see (documented, not bugs)
- Chunk-cadence luminance pulse on corrected long rollouts (paper limitation; protocol-level,
  cross-host). Mitigation knob exists: `OVERLAP=1|2` in the eval pipeline (+1/3 NFE).
- Progression anchoring: corrected videos advance the scene less (paper App. E — property of
  reference-anchored objectives as a class). Measure with `posthoc_metrics.py` before claiming.
- bf16 chaotic recurrence: per-rollout bitwise reproduction across torch/driver versions is
  impossible for 48-frame rollouts; reproduce STATISTICS (paired multi-rollout means), and
  re-run a known cell as a kill bar before trusting a rebuilt pipeline.

## Where deeper context lives
`RUN.md` (step-by-step human runbook) · `README.md` (method + reproduce) · `paper_tables.md`
(canonical numbers + provenance) · the paper's App. F (gate / when NOT to correct) and App. B
(baseline fidelity). For access to the working repo (drift_correction: full result docs,
HANDOFF.md), ask the author for collaborator access to the private GitHub.
