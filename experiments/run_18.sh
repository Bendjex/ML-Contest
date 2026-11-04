#!/usr/bin/env bash
# Detached: 18 = final export (cnn_v8.onnx) + OFFICIAL harness proof runs.
cd /home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest || exit 1
exec > experiments/logs/final_v8.log 2>&1
echo "=== start $(date -Is) ==="
export MLC_HOME=/home/olorin/ml-contest-data/materials
PY=/tmp/mlc-venv/bin/python
MODEL=/home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest/submission/model.py
[ -x "$PY" ] || { echo "=== STEP FAIL: venv missing ==="; exit 1; }

echo "--- export + verify ---"
"$PY" experiments/18_final_export.py || { echo "=== STEP FAIL export ==="; exit 1; }

echo "--- official harness: test_demo (1 run, honest eval-like) ---"
cd /home/olorin/ml-contest-data/materials || exit 1
"$PY" harness.py evaluate --model "$MODEL" --set n_runs=1 --source file:data/test_demo.npz || { echo "=== STEP FAIL demo ==="; exit 1; }

echo "--- official harness: train_100k (3 runs, comparison to v6 78.16) ---"
"$PY" harness.py evaluate --model "$MODEL" || { echo "=== STEP FAIL train ==="; exit 1; }
echo "=== DONE OK $(date -Is) ==="