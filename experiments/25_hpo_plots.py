"""25: Clean matplotlib HPO plots (no plotly dependency).
Generates: optimization history, parallel coordinates, param importance.
Output: experiments/plots/hpo/*.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots" / "hpo"
PLOTS.mkdir(exist_ok=True, parents=True)

storage = f"sqlite:///{HERE / 'optuna' / 'study.db'}"
study = optuna.load_study(study_name="cnn_hpo", storage=storage)
print(f"study: {len(study.trials)} trials, best={study.best_trial.value:.1f}", flush=True)

completed = [t for t in study.trials if t.state.name == "COMPLETE"]
pruned = [t for t in study.trials if t.state.name == "PRUNED"]

# ============ 1: Optimization History ============
fig, ax = plt.subplots(figsize=(10, 4.5))
if pruned:
    pp = [t.number for t in pruned if t.value is not None]
    pv = [t.value for t in pruned if t.value is not None]
    ax.scatter(pp, pv, alpha=0.35, c="orange", s=25, label=f"pruned ({len(pruned)})")
if completed:
    vn = [t.number for t in completed]
    vv = [t.value for t in completed]
    ax.scatter(vn, vv, c="steelblue", s=40, zorder=3, label=f"completed ({len(completed)})")
    best_val = max(vv)
    ax.axhline(best_val, ls="--", c="red", lw=1.2, label=f"best = {best_val:.1f}")
    running = []
    cur = -np.inf
    for _, v in sorted(zip(vn, vv)):
        cur = max(cur, v)
        running.append(cur)
    ax.plot(sorted(vn), running, c="darkblue", lw=1.5, alpha=0.7, label="running best")
ax.set_xlabel("Trial number")
ax.set_ylabel("Combined objective (0.5 x demo + 0.5 x val)")
ax.set_title("Optuna Optimization History — CNN HPO (TPE, 25 trials)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(str(PLOTS / "optimization_history.png"), dpi=150)
plt.close(fig)
print("[ok] optimization_history.png", flush=True)

# ============ 2: Parallel Coordinates ============
all_keys = sorted(completed[0].params.keys())
n_dims = len(all_keys)
fig, ax = plt.subplots(figsize=(max(14, n_dims * 1.2), 6))
cmap = plt.cm.viridis
vals = [t.value for t in completed]
vmin, vmax = min(vals), max(vals)

for t in completed:
    pts = []
    for j, k in enumerate(all_keys):
        raw = t.params[k]
        col = [tt.params[k] for tt in completed]
        lo, hi = min(col), max(col)
        if hi == lo:
            nv = 0.5
        elif isinstance(raw, str):
            unique_vals = list(set(col))
            nv = unique_vals.index(raw) / max(len(unique_vals) - 1, 1)
        else:
            nv = (raw - lo) / (hi - lo)
        pts.append((j, nv))
    norm_val = (t.value - vmin) / max(vmax - vmin, 1e-9)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=cmap(norm_val), lw=0.8, alpha=0.6)

ax.set_xlim(-0.5, n_dims - 0.5)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(range(n_dims))
ax.set_xticklabels(all_keys, rotation=35, ha="right", fontsize=7.5)
ax.set_title("Parallel Coordinates — each line = one trial, color = objective (viridis = higher)")
ax.set_ylabel("Normalized parameter value")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
fig.colorbar(sm, ax=ax, label="combined objective", shrink=0.8)
ax.grid(alpha=0.25, axis="x")
fig.tight_layout()
fig.savefig(str(PLOTS / "parallel_coordinates.png"), dpi=150)
plt.close(fig)
print("[ok] parallel_coordinates.png", flush=True)

# ============ 3: Parameter Importance ============
from scipy import stats

importances = []
for k in all_keys:
    col_raw = [t.params[k] for t in completed]
    if isinstance(col_raw[0], str):
        unique = list(set(col_raw))
        col = [unique.index(v) for v in col_raw]
    else:
        col = col_raw
    if len(set(col)) < 2:
        importances.append((k, 0.0))
        continue
    r, p = stats.pearsonr(col, vals)
    importances.append((k, abs(r)))

importances.sort(key=lambda x: x[1], reverse=True)
fig, ax = plt.subplots(figsize=(8, 5))
names = [k for k, _ in importances]
imps = [v for _, v in importances]
colors = plt.cm.RdYlGn([v / max(imps) if max(imps) > 0 else 0.5 for v in imps])
bars = ax.barh(names, imps, color=colors, edgecolor="k", lw=0.5)
ax.set_xlabel("|Pearson r| with combined objective")
ax.set_title("Hyperparameter Importance (correlation with objective)")
ax.grid(alpha=0.3, axis="x")
for bar, v in zip(bars, imps):
    ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
            va="center", fontsize=8)
fig.tight_layout()
fig.savefig(str(PLOTS / "param_importances.png"), dpi=150)
plt.close(fig)
print("[ok] param_importances.png", flush=True)

# ============ 4: Best trial info ============
b = study.best_trial
print(f"\nBest trial: #{b.number}, combined={b.value:.1f}", flush=True)
print(f"  demo={b.user_attrs['score_demo']:.1f} val={b.user_attrs['score_val']:.1f}", flush=True)
print(f"  val_acc={b.user_attrs['val_acc']:.3f} ort={b.user_attrs['ort_us']:.0f}us", flush=True)
print(f"  params: {b.params}", flush=True)
print("=== ALL HPO PLOTS DONE ===", flush=True)