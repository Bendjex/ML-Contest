#!/usr/bin/env bash
# Detached: 22 = official harness on the final optuna model.
cd /home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest || exit 1
exec > experiments/logs/final_harness.log 2>&1
echo "=== start $(date -Is) ==="
MODEL=/home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest/submission/model.py
PY=/tmp/mlc-venv/bin/python
cd /home/olorin/ml-contest-data/materials || exit 1
echo "--- harness: test_demo (1 run) ---"
"$PY" harness.py evaluate --model "$MODEL" --set n_runs=1 --source file:data/test_demo.npz || { echo "=== FAIL demo ==="; exit 1; }
echo "--- harness: train_100k (3 runs) ---"
"$PY" harness.py evaluate --model "$MODEL" || { echo "=== FAIL train ==="; exit 1; }
echo "=== DONE OK $(date -Is) ==="
