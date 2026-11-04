#!/usr/bin/env bash
# Detached runner: MLP v2 (3x300 iters) + baselines + latency + score table.
# Log: experiments/logs/v2.log
cd /home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest || exit 1
exec > experiments/logs/v2.log 2>&1
echo "=== start $(date -Is) ==="
export MLC_HOME=/home/olorin/ml-contest-data/materials
PY=/tmp/mlc-venv/bin/python
[ -x "$PY" ] || { echo "=== STEP FAIL: venv missing ==="; exit 1; }
"$PY" experiments/02_mlp_v2_train.py || { echo "=== STEP FAIL train ==="; exit 1; }
echo "=== DONE OK $(date -Is) ==="