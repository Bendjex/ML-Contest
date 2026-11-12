#!/usr/bin/env bash
# Detached: 20 = Optuna study — waits for stage 3 (max 3 h), then takes over all 6 cores.
cd /home/olorin/.openclaw/workspace/projects/gdg-uam-ml-contest || exit 1
exec > experiments/logs/optuna.log 2>&1
echo "=== start (waiting for stage 3) $(date -Is) ==="
export MLC_HOME=/home/olorin/ml-contest-data/materials
PY=/tmp/mlc-venv/bin/python
[ -x "$PY" ] || { echo "=== STEP FAIL: venv missing ==="; exit 1; }
"$PY" -c "import optuna" 2>/dev/null || "$PY" -m pip install -q optuna || { echo "=== STEP FAIL optuna ==="; exit 1; }

# wait for stage 3 to finish (up to 3 h)
for i in $(seq 1 360); do
  grep -q "ALL DONE" experiments/logs/hpo_stage3.log 2>/dev/null && break
  sleep 30
done
if ! grep -q "ALL DONE" experiments/logs/hpo_stage3.log 2>/dev/null; then
  echo "WARN: stage 3 not finished after 3h — starting optuna anyway (CPU shared)"
fi
echo "=== stage 3 done, launching optuna $(date -Is) ==="
"$PY" experiments/20_optuna.py || { echo "=== STEP FAIL optuna ==="; exit 1; }
echo "=== DONE OK $(date -Is) ==="