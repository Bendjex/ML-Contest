#!/usr/bin/env bash
# Detached: 17 = HPO stage 2 (winner combos, ~30-40 min).
cd /home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest || exit 1
exec > experiments/logs/hpo_stage2.log 2>&1
echo "=== start $(date -Is) ==="
export MLC_HOME=/home/olorin/ml-contest-data/materials
PY=/tmp/mlc-venv/bin/python
[ -x "$PY" ] || { echo "=== STEP FAIL: venv missing ==="; exit 1; }
"$PY" experiments/17_hpo_stage2.py || { echo "=== STEP FAIL ==="; exit 1; }
echo "=== DONE OK $(date -Is) ==="