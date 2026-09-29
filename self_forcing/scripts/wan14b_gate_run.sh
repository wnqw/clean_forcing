#!/usr/bin/env bash
# Detached driver for the Wan2.1-T2V-14B viability gate (gate only, NO training).
# Stages are resumable; every GPU stage goes through an optional safe_run.sh wrapper (SAFE_RUN)
# (slot locks, driver-health probe, thread caps). Queues (retries) while slots are busy.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${SF_REPO:-$REPO/self_forcing}" || exit 1
SAFE=${SAFE_RUN:-}   # optional GPU-slot wrapper from the original cluster; unset = run directly
PY=${PYTHON:-python}

run_stage() {
  local name=$1; shift
  local tries=0
  while true; do
    echo "=== [$(date '+%F %T')] stage $name attempt $((tries + 1)) ==="
    ${SAFE:+"$SAFE"} "$PY" -u "$@"
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
