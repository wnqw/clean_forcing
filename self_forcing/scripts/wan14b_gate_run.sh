#!/usr/bin/env bash
# Detached driver for the Wan2.1-T2V-14B viability gate (gate only, NO training).
# Stages are resumable; every GPU stage goes through drift_correction's safe_run.sh
# (slot locks, driver-health probe, thread caps). Queues (retries) while slots are busy.
cd /localhome/local-wenqingw/projs/Self-Forcing || exit 1
SAFE=/localhome/local-wenqingw/projs/drift_correction/scripts/safe_run.sh
PY=/localhome/local-wenqingw/miniconda3/envs/df-gb300/bin/python

run_stage() {
  local name=$1; shift
  local tries=0
  while true; do
    echo "=== [$(date '+%F %T')] stage $name attempt $((tries + 1)) ==="
    "$SAFE" "$PY" -u "$@"
    local st=$?
    [ $st -eq 0 ] && { echo "=== stage $name OK ==="; return 0; }
    if [ $st -eq 3 ] || [ $st -eq 4 ]; then   # safe_run REFUSED: load / slots busy -> queue
      tries=$((tries + 1))
      [ $tries -ge 240 ] && { echo "stage $name: gave up waiting for a slot"; return 1; }
      sleep 120
      continue
    fi
    echo "=== stage $name FAILED status $st ==="
    return $st
  done
}

run_stage refs wan14b_gen_refs.py &&
run_stage rollouts wan14b_rollouts.py &&
run_stage gate wan14b_gate_estimate.py
echo "=== [$(date '+%F %T')] driver exit $? ==="
