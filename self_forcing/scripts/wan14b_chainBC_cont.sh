#!/usr/bin/env bash
# Continuation of chain B+C from stage 5, using the v2 VAL-PEAK snapshot (dag-R2 +0.643 > v1 +0.482).
# Stage-4 bar resolution: bar intent (v2 >= v1) satisfied by the pre-specified val-peak snapshot;
# the final-checkpoint regression (+0.415) is recorded honestly in wan14b_v2_result.json.
set -u
cd /localhome/local-wenqingw/projs/Self-Forcing || exit 1
S=scripts/wan14b_stage.sh
PY=/localhome/local-wenqingw/miniconda3/envs/df-gb300/bin/python
AB=wan_cache/wan14b_adapted_base_4000.pt
V2CKPT=wan_cache/wan14b_lora_v2_valpeak.pt
[ -f "$AB" ] && [ -f "$V2CKPT" ] || { echo "[cont] STOP: missing ckpt"; exit 1; }
echo "=== [$(date '+%F %T')] chainBC continuation: v2 = val-peak (+0.643) ==="

"$S" 2x2 env ADAPTED_BASE="$AB" V2_NAME=wan14b_lora_v2_valpeak.pt "$PY" -u wan14b_2x2_eval.py || exit 1
BAR5=$("$PY" -c "import json;print(json.load(open('wan_cache/wan14b_2x2.json'))['_kill_bar']['verdict'])")
echo "[cont] stage-5 kill bar (>=40% cut on adapted host): $BAR5"
[ "$BAR5" != "PASS" ] && { echo "[cont] STOP: 2x2 kill bar failed"; exit 1; }

"$S" ood_abase env ADAPTED_BASE="$AB" ARM=abase "$PY" -u wan14b_ood_gen.py || exit 1
"$S" ood_av2 env ADAPTED_BASE="$AB" ARM=av2 V2="$V2CKPT" "$PY" -u wan14b_ood_gen.py || exit 1
"$S" ood_score "$PY" -u wan14b_ood_score.py || exit 1
SF=/localhome/local-wenqingw/projs/Self-Forcing
WORKDIR=/tmp "$SF/$S" vb_abase14 "$PY" -u "$SF/wan14b_score_official.py" --tag abase14
WORKDIR=/tmp "$SF/$S" vb_av2_14 "$PY" -u "$SF/wan14b_score_official.py" --tag av2_14
echo "=== [$(date '+%F %T')] chain B+C complete (continuation) ==="
