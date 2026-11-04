#!/usr/bin/env bash
# Detached: 14 = OFFIZIELLER harness.py evaluate mit submission/model.py (ONNX-Modell).
# 1) demo test set, 1 run (400 Samples -> schneller Beweis der Mechanik)
# 2) train_100k, 3 runs (Default; Stream endet bei 100k Samples)
cd /home/olorin/ml-contest-data/materials || exit 1
exec > /home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest/experiments/logs/harness_eval.log 2>&1
echo "=== start $(date -Is) ==="
PY=/tmp/mlc-venv/bin/python
MODEL=/home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest/submission/model.py
echo "--- official harness: test_demo (1 run) ---"
"$PY" harness.py evaluate --model "$MODEL" --set n_runs=1 --source file:data/test_demo.npz || { echo "=== STEP FAIL demo ==="; exit 1; }
echo "--- official harness: train_100k (3 runs) ---"
"$PY" harness.py evaluate --model "$MODEL" || { echo "=== STEP FAIL train ==="; exit 1; }
echo "=== DONE OK $(date -Is) ==="