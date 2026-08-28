#!/usr/bin/env bash
# 14B scale-up chains B+C, auto-gated on the pre-stated kill bars (stop-and-report on failure).
#   B: round-0 pairs -> v1 (bar: val R2 >= 0.15) -> DAgger pairs -> v2 (bar: dag-R2 >= v1)
#      -> in-domain 2x2 (bar: >= 40% sat-drift cut on adapted host)
#   C: OOD 32-prompt subset (2 arms, 50 s) -> MUSIQ/posthoc scoring -> official VBench rows
# Waits for chain A's stage-2 kill-bar JSON (unseeded check) and its PASS + deploy-select.
set -u
cd /localhome/local-wenqingw/projs/Self-Forcing || exit 1
S=scripts/wan14b_stage.sh
PY=/localhome/local-wenqingw/miniconda3/envs/df-gb300/bin/python
J=wan_cache/wan14b_unseeded.json

echo "[chainBC] waiting for stage-2 kill-bar result ($J)"
until [ -f "$J" ]; do sleep 600; done

VERDICT=$("$PY" -c "import json;print(json.load(open('$J'))['_kill_bar']['verdict'])")
SELECT=$("$PY" -c "import json;print(json.load(open('$J'))['_kill_bar']['deploy_select'])")
echo "[chainBC] stage-2 kill bar: $VERDICT | deploy-select: $SELECT"
[ "$VERDICT" != "PASS" ] && { echo "[chainBC] STOP: stage-2 kill bar failed"; exit 1; }
AB=wan_cache/wan14b_adapted_base_${SELECT}.pt
[ -f "$AB" ] || { echo "[chainBC] STOP: $AB missing"; exit 1; }

# ---- chain B ----
"$S" pairs0 env ADAPTED_BASE="$AB" OUT=wan_cache/wan14b_pairs_synth.pt "$PY" -u wan14b_build_pairs.py || exit 1
"$S" train_v1 env ADAPTED_BASE="$AB" "$PY" -u wan14b_train_v1.py || exit 1
V1V=$("$PY" -c "import json;d=json.load(open('wan_cache/wan14b_v1_result.json'));print(d['verdict'])")
V1R=$("$PY" -c "import json;d=json.load(open('wan_cache/wan14b_v1_result.json'));print(d['final_val_r2'])")
echo "[chainBC] stage-3 kill bar (v1 val R2 $V1R >= 0.15): $V1V"
[ "$V1V" != "PASS" ] && { echo "[chainBC] STOP: v1 kill bar failed"; exit 1; }

"$S" dagger env ADAPTED_BASE="$AB" CORRECTOR=wan_cache/wan14b_lora_v1.pt \
    OUT=wan_cache/wan14b_pairs_dagger.pt "$PY" -u wan14b_build_pairs.py || exit 1
"$S" train_v2 env ADAPTED_BASE="$AB" V1_VAL_R2="$V1R" "$PY" -u wan14b_train_v2.py || exit 1
V2V=$("$PY" -c "import json;print(json.load(open('wan_cache/wan14b_v2_result.json'))['verdict'])")
echo "[chainBC] stage-4 kill bar (v2 dag-R2 >= v1): $V2V"
[ "$V2V" != "PASS" ] && { echo "[chainBC] STOP: v2 kill bar failed"; exit 1; }

"$S" 2x2 env ADAPTED_BASE="$AB" "$PY" -u wan14b_2x2_eval.py || exit 1
BAR5=$("$PY" -c "import json;print(json.load(open('wan_cache/wan14b_2x2.json'))['_kill_bar']['verdict'])")
echo "[chainBC] stage-5 kill bar (>=40% cut on adapted host): $BAR5"
[ "$BAR5" != "PASS" ] && { echo "[chainBC] STOP: 2x2 kill bar failed"; exit 1; }

# ---- chain C ----
"$S" ood_abase env ADAPTED_BASE="$AB" ARM=abase "$PY" -u wan14b_ood_gen.py || exit 1
"$S" ood_av2 env ADAPTED_BASE="$AB" ARM=av2 "$PY" -u wan14b_ood_gen.py || exit 1
"$S" ood_score "$PY" -u wan14b_ood_score.py || exit 1
SF=/localhome/local-wenqingw/projs/Self-Forcing
WORKDIR=/tmp "$SF/$S" vb_abase14 "$PY" -u "$SF/wan14b_score_official.py" --tag abase14
WORKDIR=/tmp "$SF/$S" vb_av2_14 "$PY" -u "$SF/wan14b_score_official.py" --tag av2_14
echo "=== [$(date '+%F %T')] chain B+C complete ==="
