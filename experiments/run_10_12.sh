#!/usr/bin/env bash
# Detached: 10 = Latenz-Profiler (Phasen + torch + ONNX) und 12 = flaches 2D-Diagramm.
cd /home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest || exit 1
exec > experiments/logs/profile.log 2>&1
echo "=== start $(date -Is) ==="
export MLC_HOME=/home/olorin/ml-contest-data/materials
PY=/tmp/mlc-venv/bin/python
[ -x "$PY" ] || { echo "=== STEP FAIL: venv missing ==="; exit 1; }
"$PY" -c "import onnxruntime" 2>/dev/null || "$PY" -m pip install -q onnxruntime || echo "WARN: onnxruntime install failed ( continuing ohne ORT )"
echo "--- viz 12 (flach) ---"
"$PY" experiments/12_viz_flat.py || echo "WARN: viz failed"
echo "--- profiler 10 ---"
"$PY" experiments/10_latency_profile.py || { echo "=== STEP FAIL profiler ==="; exit 1; }
echo "=== DONE OK $(date -Is) ==="