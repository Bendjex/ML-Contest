#!/usr/bin/env bash
# Detached: 15 = Debug Harness-Diskrepanz (model.py vs numpy vs ORT auf demo+val).
cd /home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest || exit 1
exec > experiments/logs/debug_harness.log 2>&1
echo "=== start $(date -Is) ==="
export MLC_HOME=/home/olorin/ml-contest-data/materials
PY=/tmp/mlc-venv/bin/python
"$PY" experiments/15_debug_harness.py || { echo "=== STEP FAIL ==="; exit 1; }
echo "=== DONE OK $(date -Is) ==="