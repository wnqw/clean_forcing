#!/usr/bin/env bash
# 14B scale-up chain A: refs(300) -> causal adaptation (6K) -> stage-2 kill-bar check.
# Each stage goes through wan14b_stage.sh (waits for a FREE GPU, safe_run slot lock).
set -u
cd /localhome/local-wenqingw/projs/Self-Forcing || exit 1
S=scripts/wan14b_stage.sh
PY=/localhome/local-wenqingw/miniconda3/envs/df-gb300/bin/python

"$S" refs300 env NCLIPS=300 "$PY" -u wan14b_gen_refs.py &&
"$S" adapt "$PY" -u wan14b_train_adapt.py &&
"$S" unseeded "$PY" -u wan14b_unseeded_check.py
echo "=== [$(date '+%F %T')] chain A exit $? ==="
