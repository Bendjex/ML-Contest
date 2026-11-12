#!/usr/bin/env bash
# Detached: 06 = CNN numpy-path latency + per-class score projection (weights from 05).
cd /home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest || exit 1
exec > experiments/logs/cnn_latency.log 2>&1
echo "=== start $(date -Is) ==="
export MLC_HOME=/home/olorin/ml-contest-data/materials
PY=/tmp/mlc-venv/bin/python
[ -x "$PY" ] || { echo "=== STEP FAIL: venv missing ==="; exit 1; }
"$PY" experiments/06_cnn_latency.py || { echo "=== STEP FAIL ==="; exit 1; }
echo "=== DONE OK $(date -Is) ==="