#!/bin/bash
# ORCH4: CF-base inference-robustness matrix (Joonghyuk 8/26) — waits for ORCH3
# COMPLETE, then runs 16-prompt 50s cfb subsets under alternative inference configs:
#   cfbfull   ATTN=201 (naive full KV cache, no eviction over the whole horizon)
#   cfbsink3  SINK=3   (sliding window + first-chunk attention sink)
#   cfbsink10 SINK=10  (sliding window + half-window sink, Deep-Forcing-style budget)
# Our-protocol row = finals/cfb_p000..015 (already generated). Authors' README defers
# to Self-Forcing inference (verified 8/27) — no author-recommended config exists.
set -u
ROW="/localhome/local-wenqingw/projs/benchmarking/cf_row"
PY="/localhome/local-wenqingw/miniconda3/envs/df-gb300/bin/python"
cd /localhome/local-wenqingw/projs/Self-Forcing

log() { echo "- $(date -u '+%Y-%m-%d %H:%M') UTC — $1" >> "$ROW/STATUS.md"; }

until grep -q "ORCH3 COMPLETE" "$ROW/STATUS.md"; do sleep 300; done
log "ORCH4 start: robustness matrix (cfbfull ATTN=201, cfbsink3, cfbsink10; 16 prompts x 50s each)"
mkdir -p "$ROW/robust"

run_cfg() {  # $1 tag  $2 extra env as KEY=VAL
  local tag=$1 kv=$2
  env $kv TAG=$tag N=16 KLAT=201 OUTD="$ROW/robust" \
    "$PY" -u "$ROW/cf_finals.py" >> "$ROW/robust_${tag}.log" 2>&1
  local nvid
  nvid=$(ls "$ROW"/robust/${tag}_p*.mp4 2>/dev/null | wc -l)
  if [ "$nvid" -lt 16 ]; then log "ORCH4 STOP: $tag only $nvid/16 videos (see robust_${tag}.log)"; exit 1; fi
  OUTD="$ROW/robust" TAG=$tag "$PY" -u "$ROW/cf_score.py" >> "$ROW/robust_${tag}.log" 2>&1
  log "ROBUST $tag: $(grep RESULT "$ROW/robust_${tag}.log" | tail -1)"
}

run_cfg cfbfull   "ATTN=201"
run_cfg cfbsink3  "SINK=3"
run_cfg cfbsink10 "SINK=10"

log "ORCH4 COMPLETE — robustness matrix done. Compare vs finals cfb pids 0-15 (our protocol)."
