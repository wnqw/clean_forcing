# RUN.md — running Clean Forcing

Step-by-step runbook for reproducing the paper with the released code and checkpoints.
One A100/H100-class GPU (>=48 GB), bf16.

## 1. Setup (once)
```bash
bash SETUP.sh          # conda env + deps + public Wan2.1-T2V-1.3B download
```

## 2. Get the checkpoints (corrector LoRAs + adapted base)
All checkpoints are on the public Hugging Face model repo
[`illustro1/clean-forcing`](https://huggingface.co/illustro1/clean-forcing). From the repo root:
```bash
huggingface-cli download illustro1/clean-forcing --include "weights/*" --local-dir .   # 14 LoRAs -> weights/ (576 MB)
huggingface-cli download illustro1/clean-forcing adapted_base_4000.pt --local-dir self_forcing/wan_cache   # 0.57 GB
(cd weights && sha256sum -c SHA256SUMS)
```
The same repo holds the 14B adapted base (`wan14b_adapted_base_4000.pt`, ~8.4 GB) for the
scale-transfer appendix. The 1.3B correctors apply to the adapted base; to rebuild it instead:
**Reproduce (~1 GPU-day, zero real videos)**: steps 1–2 of the README "Train" section
  (`wan_gen_synthetic.py` then `STEPS=6000 wan_train_adapt.py`; deploy the 4K checkpoint).

## 3. Demo (5 minutes of GPU)
```bash
# corrected — Table-1 headline checkpoint:
LORA=weights/lora_r_phi_v2_both_adapt.pt PROMPT="a corgi surfing a wave at sunset" python demo_generate.py
# same prompt, uncorrected base — watch it collapse around 10-20 s:
LORA=none PROMPT="a corgi surfing a wave at sunset" python demo_generate.py
```

## 4. Checkpoint -> base map (weights/, all LoRA r16 unless noted)
| checkpoint | base it applies to | note |
|---|---|---|
| lora_r_phi_v2_both_adapt.pt / lora_r_phi_k48_adapt.pt | adapted_base_4000.pt | +real closed-loop / one-step (Table 1 headline) |
| lora_r_phi_v2s_adapt.pt / lora_r_phi_synth_adapt.pt | adapted_base_4000.pt | zero-real closed-loop / one-step |
| lora_r_phi_k48_adapt_r{8,32,64}.pt | adapted_base_4000.pt | rank sweep (set RANK env) |
| lora_r_phi_k48.pt / lora_r_phi_v2_both.pt | raw Wan2.1-T2V-1.3B (no adaptation) | in-domain 2x2 unadapted row |
| lora_cf_v1.pt / lora_cf_v2_both.pt | zhuhz22/Causal-Forcing `chunkwise/ar_diffusion.pt` (public; load `ckpt["generator"]`) | external-base rows; our protocol = rolling 21-frame KV window, 20-step UniPC |
| wan14b_lora_v1.pt / wan14b_lora_v2_valpeak.pt | Wan2.1-T2V-14B + its rank-64 adaptation (`wan14b_adapted_base_4000.pt`, HF above) | scale-transfer appendix |
| lora_r_phi_sf.pt | Self-Forcing distilled ckpt | NEGATIVE result — damages that host; released for reproducibility only |

## 5. Full evaluation (reproduces the paper rows)
See the README "Evaluation" section: `scripts/eval_corrector_subset.py` (MUSIQ + Delta-drift + videos),
`scripts/score_official_6dim.py` (official VBench), `scripts/score_official_semantic.py`,
`scripts/posthoc_metrics.py`. Prompts: `prompts_finals128.txt` (also at
`self_forcing/wan_cache/finals128/prompts_used.txt` after eval runs). Verify checkpoints with
`(cd weights && sha256sum -c SHA256SUMS)` after the download in section 2.

## 6. User study reproduction
`user_study/analysis.py --responses user_study/responses_anonymized.csv` reproduces the paper's
preference table exactly (33 raters, catch-item analysis documented inline).

## Gotchas
- Always run scripts from the REPO ROOT (they `chdir` into self_forcing/ themselves).
- The `weights_only` torch.load warnings are expected (TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1).
- Protocol footnote for any new numbers: all our rollouts use the rolling 21-frame KV window +
  20-step UniPC (see paper App. on the external base), not upstream 50-step defaults.
