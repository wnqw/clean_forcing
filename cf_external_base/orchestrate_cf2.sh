#!/usr/bin/env bash
# CF-row orchestrator v2 — adopts the already-running dagger build + cfb finals.
set -u
ROW=/localhome/local-wenqingw/projs/benchmarking/cf_row
SF=/localhome/local-wenqingw/projs/Self-Forcing
PY=/localhome/local-wenqingw/miniconda3/envs/df-gb300/bin/python
export PYTHONPATH=$ROW
cd "$SF" || exit 1
log() { echo "- $(date -u '+%Y-%m-%d %H:%M') UTC — $*" >> "$ROW/STATUS.md"; echo "[orch2] $*"; }
npairs() { "$PY" -c "import torch;print(len(torch.load('$ROW/pairs_synth_cf300_dagger1.pt',map_location='cpu',weights_only=False)['gen']))"; }

# Stage A: adopt running dagger build; resume once if it dies short
log "ORCH2: adopted running dagger build (pid ${DAGGER_PID:-unknown}) + cfb finals (pid ${CFB_PID:-unknown})."
while pgrep -f "cf_pairs.py" >/dev/null; do sleep 120; done
NP=$(npairs)
if [ "$NP" -lt 300 ]; then
  log "ORCH2: dagger at $NP/300 after exit — resuming once."
  CORRECTOR=$ROW/ckpts/lora_cf_v1.pt OUT=$ROW/pairs_synth_cf300_dagger1.pt NCLIPS=300 \
    "$PY" -u "$ROW/cf_pairs.py" >> "$ROW/pairs_dagger1.log" 2>&1
  NP=$(npairs)
fi
[ "$NP" -ge 300 ] || { log "ORCH2 STOP: dagger pairs $NP/300 after resume."; exit 1; }
log "STAGE DAGGER DONE: $NP pairs."

# Stage B: v2 training
log "ORCH2: v2 training (LOSS_MODE=both, 600 steps)."
LOSS_MODE=both POOLS=$ROW/pairs_synth_cf300.pt,$ROW/pairs_synth_cf300_dagger1.pt \
  INIT=$ROW/ckpts/lora_cf_v1.pt CKPT=lora_cf_v2_both.pt \
  "$PY" -u "$ROW/cf_train_v2.py" >> "$ROW/train_v2.log" 2>&1
[ -f "$ROW/ckpts/lora_cf_v2_both.pt" ] || { log "ORCH2 STOP: v2 ckpt missing."; exit 1; }
log "STAGE V2 DONE ($(grep -oE 'val R.?2? [-+0-9.]+' "$ROW/train_v2.log" | tail -1))."

# Stage C: v2 sanity (p000, K=120) + 3-way strip
TAG=sanity_v2 LORA=$ROW/ckpts/lora_cf_v2_both.pt KLAT=120 N=1 OUTD=$ROW/sanity \
  "$PY" -u "$ROW/cf_finals.py" >> "$ROW/sanity_v2.log" 2>&1
"$PY" "$ROW/cf_strip.py" "$ROW/sanity/sanity3way_strip.png" \
  $(ls /localhome/local-wenqingw/projs/benchmarking/cf_feasibility/videos_cf/cf_p000_k120*.mp4 2>/dev/null | head -1) \
  "$ROW/sanity/sanity_v1final_p000.mp4" $(ls "$ROW/sanity/"sanity_v2*.mp4 2>/dev/null | head -1) \
  >> "$ROW/sanity_v2.log" 2>&1 || true
log "STAGE SANITY-V2 DONE: sanity/sanity3way_strip.png (EYES-ON: v2 must fix late softness)."

# Stage D: cfc finals (concurrent with any remaining cfb work)
log "ORCH2: cfc finals launched (128x50s, resumable)."
TAG=cfc LORA=$ROW/ckpts/lora_cf_v2_both.pt KLAT=201 N=128 OUTD=$ROW/finals \
  "$PY" -u "$ROW/cf_finals.py" >> "$ROW/finals_cfc.log" 2>&1
log "STAGE CFC FINALS DONE ($(ls "$ROW/finals"/cfc_*.mp4 2>/dev/null | wc -l) videos)."

# Stage E: wait for cfb, verify counts, resume once if short, then score
[ -n "${CFB_PID:-}" ] && while kill -0 "$CFB_PID" 2>/dev/null; do sleep 60; done
NB=$(ls "$ROW/finals"/cfb_*.mp4 2>/dev/null | wc -l)
if [ "$NB" -lt 128 ]; then
  log "ORCH2: cfb at $NB/128 — resuming once."
  TAG=cfb KLAT=201 N=128 OUTD=$ROW/finals "$PY" -u "$ROW/cf_finals.py" >> "$ROW/finals_cfb.log" 2>&1
  NB=$(ls "$ROW/finals"/cfb_*.mp4 2>/dev/null | wc -l)
fi
log "ORCH2: cfb settled ($NB/128). Scoring."
TAG=cfb N=128 "$PY" -u "$ROW/cf_score.py" >> "$ROW/score_cfb.log" 2>&1; log "SCORE cfb: $(grep RESULT "$ROW/score_cfb.log" | tail -1)"
TAG=cfc N=128 "$PY" -u "$ROW/cf_score.py" >> "$ROW/score_cfc.log" 2>&1; log "SCORE cfc: $(grep RESULT "$ROW/score_cfc.log" | tail -1)"
"$PY" -u "$ROW/cf_pulse_prog.py" >> "$ROW/pulse_prog.log" 2>&1; log "PULSE/PROG: $(tail -1 "$ROW/pulse_prog.log")"
log "ORCH2 COMPLETE — CF row finished. Next: eyes-on sanity strip, paper row, evacuation."
