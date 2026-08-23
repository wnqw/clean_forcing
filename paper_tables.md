# Paper Tables — final (all complete, 2026-07-22)

Finals protocol: 128 LLM-refined MovieGen prompts (idx>320), 50 s / 201 latents, adapted stack, CFG 6 / shift 8,
1 seed/prompt. Every paper table below is locked and propagated to LaTeX; provenance sections at the bottom.
History (freeze incidents, run logs, decision trails) lives in git log of this file.

| # | table | in paper |
|---|---|---|
| T1 | main comparison (128p finals) | `tables/main.tex` ✅ |
| T2 | in-domain 2×2 | `tables/in_domain_2x2.tex` ✅ |
| T3 | ablations | `tables/ablations.tex` ✅ |
| T4 | data regimes | `tables/data_regimes.tex` ✅ |
| T5 | guards / gameability | prose §4.4 + suite footnotes ✅ |
| T6 | multi-metric drift suite | `tables/drift_suite.tex` ✅ |
| T7 | checkpoint transfer | prose §4.5 ✅ |
| T8 | overhead | prose §4.5 ✅ |
| T9 | gate diagnostic α*(t) | method §3.2 + ablation row ✅ |
| T10 | do-no-harm 5 s | `tables/do_no_harm.tex` ✅ |
| T11 | adaptation scale 4K→20K | appendix scaling figure ✅ (section below) |
| T12 | 14B mini-study | descoped (port covers generality) |
| A/B | hyperparams / baseline provenance | appendix ✅ |
| C/D | full metrics / qualitative | appendix table + strips ✅ |

## T1 — Main comparison (OOD 50 s un-seeded T2V; official VBench calibrated; Δ = MUSIQ first-20% − overall)

Ours = av2s (zero real videos); av2 = +40 real clips ceiling row; av1s in T3.

| config | subject↑ | backgr↑ | aesthetic↑ | imaging↑ | smooth↑ | dynamic | flicker↓ᵍ | MUSIQ | Δ-drift↓ | overhead |
|---|---|---|---|---|---|---|---|---|---|---|
| adapted base | 65.7 | 76.7 | 44.9 | 47.9 | 91.1 | 29.7 | 15.9 | 44.8±0.95 | +11.27±0.81 | — |
| + context noise σ=0.2 | 67.2 | 77.8 | 45.2 | 51.4 | 85.8 | 32.0 | 11.1 | 49.1±1.02 | +12.05±0.76 | 0 / 0% |
| + History Guidance w=1.2 | 64.6 | 76.1 | 42.4 | 47.7 | 94.5 | 0.8 (static) | 8.2* | 42.3±0.46 | **+20.55±0.65** | 0 / +100% |
| + Pathwise TTC {500,250} | 59.7 | 74.0 | 40.2 | 37.4 | 91.0 | 64.1 (churn) | 9.2* | 35.5±0.72 | +13.96±0.91 | 0 / +20% |
| + Pathwise TTC {750,500} | 61.5 | 75.4 | 42.6 | 39.9 | 91.2 | 71.1 (churn) | 9.3* | 38.0±0.80 | +14.51±0.80 | 0 / +20% |
| **Ours (zero real, av2s)** | **81.0** | **86.6** | 60.5 | 63.9 | 89.0 | 48.4 | 22.8 | 66.9±0.80 | +1.65±0.35 | +0.4% merged / **0%** |
| Ours +40 real clips (av2) | 80.9 | 86.4 | **61.4** | **64.1** | 89.1 | 49.2 | 32.8 | **67.0**±0.61 | **−0.16**±0.12 | +0.4% merged / **0%** |
| BAgger DF baseline† | 80.7 | 87.7 | 53.1 | 60.0 | 98.1 | 81.3 | — | — | +7.34 | full retrain |
| BAgger R3† | 84.1 | 89.6 | 55.4 | 63.4 | 98.6 | 76.6 | — | — | +3.57 | full retrain ×3 |
| SF-distilled (same prompts) | 80.2 | 84.4 | 60.9 | 67.7 | 94.9 | 39.8 | 9.5 | 69.3±0.60 | +1.08±0.39 | own system |
| SkyReels-V2-DF-1.3B (same prompts) | **87.0** | **90.7** | 61.1 | 62.8 | **96.6** | 50.8 | 4.5ᶠ | 61.5±1.04 | +1.00±0.30 | own system |

†published, own base/prompts (no code/checkpoints — cells "—" permanent; closest same-video proxy = our Tier-A
replications). ᵍ flicker from T6 (32-video subset). * degradation artifact: static/blurred configs "beat" the
healthy base's flicker — smoothness-flavored metrics reward death. ᶠ SkyReels 24 fps native (smaller inter-frame
diffs; not comparable to 16 fps rows). Error bars = SEM over 128 prompts.
**Drift ordering**: HG +20.5 > TTC +14.0/+14.5 > DF-σ +12.1 > abase +11.3 > BAgger-base† +7.3 > R3† +3.6 >
av2s +1.65 ≈ SF +1.08 ≈ SkyReels +1.00 > av2 −0.16. All four training-free interventions worsen drift; av2
beats SF by >3 SEM; av2s statistically ties SF/SkyReels with zero real videos; ours leads aesthetic, beats
R3/SkyReels on imaging, at ~1/1000th their data. Known weakness: smoothness 89 vs 95–99 (chunk seams; §limitations).
(av1s @128p, for T3: 77.2/83.8/55.7/62.2/91.0/38.3 · MUSIQ 61.9 · Δ +4.77)

## T2 — In-domain 2×2 (5 held-out Disney clips × 3 seeds, paired; sat-drift ↓ / MUSIQ_late ↑)

| | no corrector | + r_φ v1 | + r_φ v2 |
|---|---|---|---|
| unadapted base | 0.491±0.169 / 29.6±5.6 | 0.166±0.180 / 36.9±11.4 | 0.131±0.100 / 60.5±6.8 |
| adapted base | 0.415±0.087 / 38.5±9.7 | **0.061±0.061** / 59.4±2.0 | 0.082±0.074 / **63.3±2.9** |

Corrector works on both bases (−66/−73% unadapted, −85/−80% adapted); adaptation ≈ un-seeded competence,
corrector ≈ drift removal (complementary); the stack collapses seed variance (±2–3 vs ±5–11 MUSIQ).
Open-loop R²: v1-Disney 0.655 (unadapted) → 0.844 (adapted).

## T3 — Ablations (unadapted base, in-domain protocol; sat-drift ↓ / MUSIQ_late ↑)

| row | result | conclusion |
|---|---|---|
| baseline | 0.491±0.169 / 29.6 | — |
| **ours v1** | 0.166 (−66%) / 36.9 | one-step counterfactual teacher works |
| **ours v2** | 0.131 (−73%) / 60.5 | DAgger (−70%) then +contraction (−73%) — both add |
| full fine-tune, same loss (283M) | 0.379 (−23%) | best open-loop R² (0.763), worst rollout → **low-rank preserves rollout stability** |
| residual side-net (1.9M) | 0.367 (−25%) | external module < in-backbone LoRA |
| DMD-LoRA (no clean ref) | 0.492 (−0%) | **the counterfactual teacher is the necessary ingredient** |
| naive-LoRA (FM on clean clips) | degenerate (dyn 5.2) | motion-collapse cheat — replicates AutoRefiner's finding |
| DF noisy-context σ=0.2 | Δ +7.34 vs +5.31 | worsens drift |
| History Guidance (exact + variant) | 0.528/26.6 · 0.514/27.7 | no effect — matches BAgger's table |
| Drift-SNR gate α(t) vs flat | 0.133 ≡ 0.131 | α*≈0.95 ∀t → diagnostic, not knob |
| anchoring: GT vs native z_t (toy) | GT-anchor injects drift | native anchoring required |
| av1s @128p | Δ +4.77 / 61.9 | v1→v2 closes Δ 4.77→1.65 on the same synthetic data |

## T4 — Data regimes (clean-history source; corrector form/loss fixed)

| clean-history source | real videos | in-domain (sat-drift) | OOD finals (Δ / MUSIQ) |
|---|---|---|---|
| real clips (40 Disney) | 40 | v1 −66% · v2 −73% | av2 **−0.16** / 67.0 |
| synthetic (300 bidi clips) | **0** | — | av1s +4.77 / 61.9 · av2s **+1.65 / 66.9** |
| none (pure DMD) | 0 | −0% | — |

Prompts-only training reaches quality parity and BAgger-beating drift; a clean reference is necessary;
40 real clips buy the negative-drift ceiling.

## T5 — Guards / gameability (in-domain interim + official-unit exhibits)

| config | subj_cons ↑ | i2v-seed ↑ | bg_cons ↑ | flicker ↓ | dyn (guard) |
|---|---|---|---|---|---|
| GT clips (anchor) | — | — | — | 2.8 | 1.2 |
| baseline | 0.521 | 0.177 | 0.811 | 39.0 | 73.1 (churn) |
| + r_φ v1 | 0.582 | 0.249 | 0.841 | **6.7** | 39.7 |
| + r_φ v2 | **0.614** | **0.291** | 0.831 | 15.1 | 30.7 |

Exhibits: naive-LoRA subj 66.0/img 58.5 with dyn 12.5 (static-texture cheat) · collapse configs dyn 81–100
(churn) · fading color field scores smooth ≈97 — no single scalar is trustworthy.

## T6 — Multi-metric drift suite (finals, 32 videos/config)

| config | ΔQD (TetherCache)↓ | color-shift L1↓/corr↑ | survival↑ | t-LPIPS↓ | flicker↓ |
|---|---|---|---|---|---|
| abase | +22.9 | 1.43 / 0.16 | 0.44 | 0.23 | 15.9 |
| DF-σ | +31.4 | 1.44 / 0.18 | 0.45 | 0.19 | 11.1 |
| HG | +36.5 | 1.25 / 0.15 | 0.37 | 0.12* | 8.2* |
| SF-distilled | +4.6 | 1.33 / 0.06 | 0.64 | **0.15** | 9.5 |
| **ours (av2s)** | +3.5 | 0.95 / 0.48 | **0.70** | 0.33 | 22.8 |
| ours +real (av2) | **+0.36** | **0.85 / 0.57** | 0.69 | 0.59 | 32.8 |

Every published drift metric confirms the Δ ranking; ours worst on t-LPIPS/flicker (chunk seams — reported,
not hidden). *Static baselines "win" smoothness while dead.

## T7 — Checkpoint transfer
Sibling (unadapted-trained → adapted base): 0.320/56.6 vs native 0.061/59.4 — quality transfers, drift
calibration ~25%. Distant (multi-step → SF-distilled, same arch): open-loop R² **+0.03 vs +0.39 native** —
nothing transfers across host types. The RECIPE transfers, the weights don't; native re-instantiation ≈ 3.5 GPU-h.

## T8 — Overhead
Re-benchmarked 2026-07-30 (3 runs × 21 steady-state chunks, batch 1, synchronized timers; `overhead_bench.md` +
`scripts/bench_outputs/`): **base 2.90±0.35 s/chunk, merged 3.00±0.89 (+3.5%, Welch p=0.40 n.s., median merged
< base), unfused 3.92±0.49 (+35.3%, p<0.001)**; peak VRAM identical (29.49 GiB); ΔNFE=0 (42 forwards/chunk
asserted). Paper Table `tab:overhead` (appendix) carries these. Historical single-shot pair (merged 21.0 vs
base 24.1 s — different batch/software state) superseded; confirmed run variance.
Functional equivalence verified (latent MAD 0.108 = bf16 rounding through chaotic recurrence).

## T9 — Gate diagnostic α*(t)
α*(t)=‖bias‖²/(‖bias‖²+var), M=4 seeds, 20 steps: **0.91–0.99 at every t** (~95% systematic). Closed-loop
gated ≡ flat (0.133 vs 0.131) → gate = diagnostic justifying full correction on this host.

## T10 — Do-no-harm 5 s (AutoRefiner protocol, 200 VBench prompts; ours = merged av2s)

| dim | adapted base | **ours (merged)** | Δ |
|---|---|---|---|
| subject consistency | 91.64 | **93.86** | +2.2 |
| background consistency | 93.06 | **94.47** | +1.4 |
| aesthetic quality | 56.99 | **59.00** | +2.0 |
| imaging quality | 64.93 | **68.94** | +4.0 |
| motion smoothness | **97.15** | 95.50 | −1.65 |
| dynamic degree | 21.50 | **34.00** | +12.5 |

Better than do-no-harm: improves 5/6 dims pre-drift; the smoothness dip co-occurs with the +12.5 dynamic gain.

## T11 — Adaptation scaling (ships as the appendix figure)
Same-16, uncorrected base: 4K = 44.8/+11.27/anchor 0.336 · 8K = 48.1/+11.72/0.352 · 14K = 48.9/+12.73/0.393 ·
20K = 57.8/+9.68/0.439 (late jump 14K→20K; not saturated). Corrector on 20K base = 62.3/+3.36 — loses to av2s
66.9/+1.65 → compounding rejected; correction quality is not monotone in host quality.

## Notes that shipped as prose
- **Overshoot check (av2 −0.16 is honest)**: temporal MUSIQ profile flat (slope +0.14/100fr); the −0.16 comes
  from marginally weaker opening frames, not late over-sharpening; SF's 2nd-half slope −3.1 is genuine decay.
- **Progression/anchoring analysis**: superseded by the canonical protocol metrics below + the trade section
  (§4.6) and the dichotomy; full post-mortem in `issues.md`.

---

## Canonical protocol metrics (2026-07-21; single code path, all 128 finals prompts, 50 s)

Source for the appendix table, §4.6 identity/cuts prose, and limitations.

| config | anchoring (latesim) | paired-excess vs SF | lag-2s identity | cuts/video |
|---|---|---|---|---|
| adapted base (host) | 0.336 | −0.148 | 0.812 | 2.82 |
| + ours zero-real (av2s) | 0.679 | +0.194 | 0.830 | 0.89 |
| + ours +real (av2) | 0.795 | +0.311 | 0.746 | 0.04 |
| Self Forcing | 0.485 | ±0.000 | 0.922 | 0.26 |
| SkyReels-V2-DF | 0.697 | +0.213 | 0.927 | 0.12 |

20K-host pair (16-subset, same code path): t11 lag-2s 0.877 → +corrector 0.876 (zero identity cost on the
strong host; on the 4K host the +real variant costs 0.066, zero-real costs nothing).

## Full-frame-rate re-score (2026-07-20; stride-bias audit)
Per-frame MUSIQ, no stride: av2s 65.7/Δ+1.60 · av2 65.2/−0.10 · SF 68.5/+1.34 · SkyReels 61.5/+0.98
(abase mean 43.9). Max shifts vs protocol scoring: 1.8 MUSIQ / 0.26 Δ; all orderings unchanged (§4.2 sentence).

## Trade-analysis provenance (§4.6)
Less-anchored-half Δ: zero-real +2.2, +real −0.1 (vs base +11.3); per-video anchoring×Δ correlation −0.05.
Seam pulse: chunk-lag luminance autocorr abase 0.80 > av2s 0.72 (visibility is contrast — limitations).
Deployed host = 4K checkpoint of the 6K-step adaptation (`adapted_base_4000.pt`; T1 host row ≡ scaling 4K point).

---

# 2026-08 additions (post-freeze results, all in the paper)

## T13 semantic axis (official VBench custom-input, 128 videos/config, ±SEM)
| config | overall consistency (prompt adherence) ↑ | flicker* |
|---|---|---|
| adapted base | 58.81±1.32 | 90.25±0.72 |
| context noise | 61.68±1.33 | 86.38±1.20 |
| history guidance | 62.15±1.08 | 93.94±0.18 |
| TTC {500,250} / {750,500} | 47.00±1.52 / 50.53±1.55 | 89.94 / 89.86 |
| Self Forcing | 70.12±1.19 | 93.37±0.68 |
| SkyReels-V2-DF | 67.31±1.29 | 93.97±0.56 (24 fps) |
| **ours zero-real** | **69.85±1.23** | 87.29±0.61 |
| **ours +real** | **70.28±1.11** (top of all 9) | 86.75±0.36 |

*flicker = degradation-rewarding diagnostic only (near-static HG tops it).

## T14 horizon sweep (same videos truncated; Δ / MUSIQ)
ours +real horizon-flat: 10 s +0.39/66.8 · 30 s −0.01/67.0 · 50 s −0.16/67.0; host already drifted
at 10 s (+10.74/56.1 → 50 s +11.27/44.8). Caveat: host Δ peaks at 30 s then shrinks — Δ saturates
at long horizons; pair with absolute MUSIQ.

## T15 Wan2.1-14B (identical zero-real recipe, ~3 GPU-days, complete 2026-08-03)
Gate α* 0.934 pooled / 0.961 high-t · one-step val R² +0.482 · closed-loop val-peak +0.643 ·
in-domain 2×2 −60/−70% sat-drift, late MUSIQ up to +43 · **32-prompt OOD: Δ −94% (+15.65→+0.97),
MUSIQ 46.4→69.0, lag-2s identity 0.79→0.91, cuts halved; official VBench improves every dim at
unchanged dynamic degree 37.5**. Sources: wan_cache/wan14b_*.json, vbench_wan14b/table.json.

## T16 LoRA rank sweep (one-step, adapted host, paired 5 clips × 3 seeds)
| rank | open-loop val R² | sat-drift | late MUSIQ | paired ΔMUSIQ vs r16 |
|---|---|---|---|---|
| 8 | 0.829 | 0.048 (−89%) | 55.3 | −1.0±0.8 (n.s.) |
| **16 (deployed)** | 0.854 | 0.047 (−89%) | 56.3 | — |
| 32 | 0.843 | 0.050 (−88%) | 49.1 | −7.2±2.7 |
| 64 | 0.843 | 0.031 (−93%) | 51.7 | −4.7±1.4 |

Fit flat in rank; rollout quality peaks at r16 (capacity knee) — dose–response version of the
full-fine-tune divergence.

## T17 inter-seed diversity (DINO 1−cos, 5 clips × 3 seed-pairs)
base 0.327±0.019; corrected configs 0.307–0.377 (r8/r16/r32/r64/closed-loop) — parity ±0.05, no
collapse; LPIPS reductions for deployed correctors are confounded by base drift artifacts.

## Seed-std upgrade (T2 2×2 error bars)
±std over 3 seed means: corrected-adapted late MUSIQ ±0.2–0.5 vs unadapted ±1.2–2.1 (mixed
clip×seed std previously printed was 3–10× larger).
