#!/bin/bash
# ORCH3: post-finals scoring chain — pulse/prog (fixed invocation) -> VBench 6-dim
# -> semantic 2-dim -> ranked pair thumb sheets. Disk-chained, resumable-ish
# (VBench table.json accumulates; rerunning a done tag re-scores, so guard on table keys).
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
set -u
ROW="${CF_ROW:-$REPO/cf_external_base}"
SF="${SF_REPO:-$REPO/self_forcing}"
DC="$REPO"
PY=${PYTHON:-python}
cd "$ROW"  # neutral cwd for VBench (NOT Self-Forcing/)

log() { echo "- $(date -u '+%Y-%m-%d %H:%M') UTC — $1" >> "$ROW/STATUS.md"; }

# Stage A: expose finals to the canonical scorers (idempotent)
for f in "$ROW"/finals/cf{b,c}_p*.mp4; do
  ln -sf "$f" "$SF/wan_cache/finals128/$(basename "$f")"
done
log "ORCH3 armed: pulse/prog -> 6dim -> semantic -> thumbs (symlinks refreshed)"

# Stage B: pulse/prog with proper TAG + argv (orch2 bug: called with no args)
for tag in cfb cfc; do
  TAG=$tag LAGS=12,84 "$PY" -u "$ROW/cf_pulse_prog.py" "$ROW"/finals/${tag}_p*.mp4 \
    >> "$ROW/pulse_prog_${tag}.log" 2>&1
  log "PULSE/PROG $tag: $(grep RESULT "$ROW/pulse_prog_${tag}.log" | tail -1)"
done

# Stage C: official VBench 6-dim (sequential; ~2-3h per tag)
for tag in cfb cfc; do
  "$PY" -u "$DC/scripts/score_official_6dim.py" --tag $tag >> "$ROW/vbench6_${tag}.log" 2>&1
  log "VBENCH6 $tag: $(grep -c RESULT "$ROW/vbench6_${tag}.log")/6 dims done; $(grep RESULT "$ROW/vbench6_${tag}.log" | tr '\n' ' | ')"
done

# Stage D: semantic 2-dim
for tag in cfb cfc; do
  "$PY" -u "$DC/scripts/score_official_semantic.py" --tag $tag >> "$ROW/vbsem_${tag}.log" 2>&1
  log "SEMANTIC $tag: $(grep RESULT "$ROW/vbsem_${tag}.log" | tr '\n' ' | ')"
done

# Stage E: ranked pair thumb sheets
"$PY" -u "$ROW/cf_thumbs.py" >> "$ROW/thumbs.log" 2>&1
log "THUMBS: $(grep RESULT "$ROW/thumbs.log" | tail -1)"

log "ORCH3 COMPLETE — all CF-row scoring done. Next: paper pass + buffer matrix (full-KV/sink)."
