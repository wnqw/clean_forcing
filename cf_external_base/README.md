# External-base transfer: Causal Forcing (zhuhz22/Causal-Forcing, chunkwise AR teacher)

Scripts that produced the paper's external-base rows: the same Clean Forcing recipe
(one-step -> DAgger -> closed-loop, LoRA r16 on the frozen base, zero real videos)
applied to the Causal-Forcing chunkwise `ar_diffusion.pt` teacher with **no host
modification and no per-host tuning**.

Result (n=128 prompts, 50 s, our protocol): Delta-drift +11.99 -> +2.80 (-77%),
MUSIQ 52.0 -> 62.1; VBench subject 64.9 -> 81.8, background 75.7 -> 87.9,
aesthetic 45.8 -> 62.9, dynamic 33.6 -> 70.3. Corrector weights: `../weights/lora_cf_v1.pt`
(one-step), `../weights/lora_cf_v2_both.pt` (closed-loop, deployed).

Robustness matrix (16 prompts, same pids; MUSIQ/Delta): rolling window (ours) 51.5/+11.07 is
mid-pack for the base — full KV 46.4/+16.34 (worse; OOD rope beyond the 21-frame trained window),
sink-3 54.7/+8.48, sink-10 57.2/+5.29, ctx-noise 47.8/+12.59 (worse); corrector 58.8/+3.94;
corrector+sink-10 59.4/+3.23 (best; complementarity measured). OVERLAP=2 decode cuts the
chunk-cadence pulse 538 -> 99 with motion/anchoring unchanged. Env knobs in `cf_common.py`:
ATTN (KV horizon), SINK (sink frames), CTXSIG (context-noise sigma), OVERLAP (seam blend).

## Files
- `cf_common.py` — loads the CF generator into our `CausalDiffusionInferencePipeline`
  (strict 825/825 state-dict match; `ckpt["generator"]`). Env overrides for the
  inference-robustness matrix: `ATTN=201` (full KV, no eviction), `SINK=N` (N-frame sink).
- `cf_pairs.py`, `cf_train_v1.py`, `cf_train_v2.py` — pair build + the two corrector stages.
- `cf_finals.py` — 128-prompt 50 s finals (`TAG=cfb` base / `TAG=cfc` corrected; seed = prompt idx).
- `cf_score.py`, `cf_pulse_prog.py` — MUSIQ/Delta/latesim/cuts + seam-pulse/progression metrics.
- `cf_strip.py`, `cf_thumbs.py` — contact strips + ranked pair sheets.
- `orchestrate_cf{2,3,4}.sh` — the detached stage-chained orchestrators actually used
  (paths are machine-specific; read them as a runbook of stage order and env settings).

## Protocol notes (also footnoted in the paper)
- Their README gives no inference config ("inference environment is identical to Self
  Forcing"); we run rolling 21-frame KV, sink 0, 20-step UniPC — the same locked protocol
  as every other row. The base is teacher-forcing trained (never conditioned on its own
  outputs during training), which is why it is sharp early (first-20% MUSIQ ≈64) and
  collapses by ~10-15 s in self-rollout.
- Robustness matrix (`orchestrate_cf4.sh`): full-KV / sink-3 / sink-10 subsets bound the
  protocol effect on the base's numbers.
