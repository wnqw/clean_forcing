#!/usr/bin/env bash
# Stage launcher for the Wan2.1-14B reduced-protocol scale-up.
#   wan14b_stage.sh <stage-name> <command...>
# Waits until the GPU is actually FREE (no compute processes AND no safe_run slots) per the
# owner's coordination rule, then runs the command through safe_run.sh (slot lock, health
# probe, thread caps). Retries on safe_run REFUSED. Caller detaches via setsid.
cd "${WORKDIR:-/localhome/local-wenqingw/projs/Self-Forcing}" || exit 1
SAFE=/localhome/local-wenqingw/projs/drift_correction/scripts/safe_run.sh
NAME=$1; shift

wait_gpu_free() {
  while true; do
    local nprocs nslots
    nprocs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)
    nslots=$(ls /tmp/gpu_slots/*.pid 2>/dev/null | wc -l)
    [ "$nprocs" -eq 0 ] && [ "$nslots" -eq 0 ] && return 0
    echo "[$(date '+%F %T')] $NAME: GPU busy ($nprocs procs, $nslots slots) - waiting"
    sleep 300
  done
}

tries=0
while true; do
  wait_gpu_free
  echo "=== [$(date '+%F %T')] stage $NAME attempt $((tries + 1)) ==="
  "$SAFE" "$@"
  st=$?
  [ $st -eq 0 ] && { echo "=== [$(date '+%F %T')] stage $NAME OK ==="; exit 0; }
  if [ $st -eq 3 ] || [ $st -eq 4 ]; then
    tries=$((tries + 1))
    [ $tries -ge 500 ] && { echo "stage $NAME: gave up waiting"; exit 1; }
    sleep 120
    continue
  fi
  echo "=== [$(date '+%F %T')] stage $NAME FAILED status $st ==="
  exit $st
done
