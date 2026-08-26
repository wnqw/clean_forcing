# Released corrector checkpoints (LoRA r16 self-attn q/k/v/o unless noted)

All correctors apply to a FROZEN base at scale 1 (merge for zero overhead).
Verify integrity: `sha256sum -c SHA256SUMS`.

## Wan2.1-T2V-1.3B, our causally-adapted host (paper Tables 1–2)
| file | paper name | notes |
|---|---|---|
| lora_r_phi_k48_adapt.pt | one-step, +real (adapted) | tab:2x2 adapted/one-step; rank sweep r16 ref |
| lora_r_phi_v2_both_adapt.pt | closed-loop, +real (adapted) = "+ Clean Forcing (+40 real clips)" | Table 1 headline row |
| lora_r_phi_synth_adapt.pt | one-step, zero-real (adapted) | data-regimes table |
| lora_r_phi_v2s_adapt.pt | closed-loop, zero-real = "+ Clean Forcing (zero real videos)" | Table 1 |
| lora_r_phi_k48.pt / lora_r_phi_v2_both.pt | one-step / closed-loop on the UNADAPTED base | tab:2x2 unadapted row |
| lora_r_phi_k48_adapt_r{8,32,64}.pt | rank-sweep variants | tab:rank (r16 = capacity knee) |

## Wan2.1-T2V-14B (paper Appendix, scale transfer)
| file | paper name |
|---|---|
| wan14b_lora_v1.pt | one-step (val R² +0.482) |
| wan14b_lora_v2_valpeak.pt | closed-loop, deployed val-peak snapshot (OOD Δ −94%) |

## Causal-Forcing external base (paper Table 1 external rows)
| file | paper name |
|---|---|
| lora_cf_v1.pt | one-step on the Causal-Forcing base (zhuhz22/Causal-Forcing, Apache-2.0) |
| lora_cf_v2_both.pt | closed-loop (added when training completes) |

## Negative-result port (paper Appendix, "when not to correct")
| file | notes |
|---|---|
| lora_r_phi_sf.pt | corrector trained on the Self-Forcing-distilled host; deploying it DAMAGES that host — released for reproducibility of the negative result, not for use |
