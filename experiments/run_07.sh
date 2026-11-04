#!/usr/bin/env bash
# Detached: 07 = EDA plots (class dist, meta, examples, spectra, acf, nan, std, confusion).
cd /home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest || exit 1
exec > experiments/logs/eda.log 2>&1
echo "=== start $(date -Is) ==="
export MLC_HOME=/home/olorin/ml-contest-data/materials
PY=/tmp/mlc-venv/bin/python
[ -x "$PY" ] || { echo "=== STEP FAIL: venv missing ==="; exit 1; }
"$PY" -c "import matplotlib" 2>/dev/null || "$PY" -m pip install -q matplotlib || { echo "=== STEP FAIL matplotlib ==="; exit 1; }
"$PY" experiments/07_eda.py || { echo "=== STEP FAIL ==="; exit 1; }
echo "=== DONE OK $(date -Is) ==="