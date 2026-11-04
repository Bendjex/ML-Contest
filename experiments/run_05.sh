#!/usr/bin/env bash
# Detached: 05 = 1D-CNN (torch CPU) -> numpy export + latency + confusion matrix.
cd /home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest || exit 1
exec > experiments/logs/cnn.log 2>&1
echo "=== start $(date -Is) ==="
export MLC_HOME=/home/olorin/ml-contest-data/materials
PY=/tmp/mlc-venv/bin/python
[ -x "$PY" ] || { echo "=== STEP FAIL: venv missing ==="; exit 1; }
"$PY" -c "import torch" 2>/dev/null || { echo "=== STEP FAIL: torch missing ==="; exit 1; }
"$PY" experiments/05_cnn.py || { echo "=== STEP FAIL ==="; exit 1; }
echo "=== DONE OK $(date -Is) ==="