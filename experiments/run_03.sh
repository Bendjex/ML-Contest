#!/usr/bin/env bash
# Detached runner: 03 = baseline refit on features + final latency table + score projection.
# Log: experiments/logs/latency_final.log
cd /home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest || exit 1
exec > experiments/logs/latency_final.log 2>&1
echo "=== start $(date -Is) ==="
export MLC_HOME=/home/olorin/ml-contest-data/materials
PY=/tmp/mlc-venv/bin/python
[ -x "$PY" ] || { echo "=== STEP FAIL: venv missing ==="; exit 1; }
"$PY" experiments/03_latency_fix.py || { echo "=== STEP FAIL ==="; exit 1; }
echo "=== DONE OK $(date -Is) ==="