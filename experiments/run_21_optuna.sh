#!/usr/bin/env bash
# Detached: 21 = widened Optuna study (14-dim, 50 trials, ~4 h).
cd /home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest || exit 1
exec > experiments/logs/optuna.log 2>&1
echo "=== start $(date -Is) ==="
export MLC_HOME=/home/olorin/ml-contest-data/materials
PY=/tmp/mlc-venv/bin/python
[ -x "$PY" ] || { echo "=== STEP FAIL: venv missing ==="; exit 1; }
"$PY" experiments/20_optuna.py || { echo "=== STEP FAIL ==="; exit 1; }
echo "=== DONE OK $(date -Is) ==="