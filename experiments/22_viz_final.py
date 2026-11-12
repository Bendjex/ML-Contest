"""22: Flat 2D architecture diagram for the FINAL Optuna model (trial 20).

Style: same as 12_cnn_pass_flat.png but with the final architecture:
  conv1: 4→24, k=7, avg_pool 4      -> (24, 256) -> (24, 64)
  conv2: 24→48, k=13, avg_pool 4    -> (48, 64) -> (48, 16)
  conv3: 48→72, k=9, dil=2          -> (72, 16)
  global mean+max+std                 -> (216,)
  + meta [sr, group, level]          -> (221,)
  fc 221→8                           -> logits

Width = time (sqrt-scaled), Height = channels (sqrt-scaled).
English labels throughout.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

stages = [
    dict(op="Input\n(raw)",                    shape="(4, 256)",   L=256, C=4,   color="#8da2c4"),
    dict(op="Conv1 +BN +ReLU\n24 filters k=7",  shape="(24, 256)",  L=256, C=24,  color="#e08a4e"),
    dict(op="AvgPool ÷4",                       shape="(24, 64)",   L=64,  C=24,  color="#f2c088"),
    dict(op="Conv2 +BN +ReLU\n48 filters k=13", shape="(48, 64)",   L=64,  C=48,  color="#e08a4e"),
    dict(op="AvgPool ÷4",                       shape="(48, 16)",   L=16,  C=48,  color="#f2c088"),
    dict(op="Conv3 +BN +ReLU\n72 filters k=9 d=2", shape="(72, 16)", L=16,  C=72, color="#e08a4e"),
    dict(op="Global Pool\nmean+max+std\n+meta(5)", shape="(221,)",   L=1,   C=72, color="#7fae7f"),
    dict(op="FC 221→8\n→ argmax",              shape="(8,)",       L=1,   C=8,   color="#9b86c6"),
]

def w_of(L):
    return 1.6 if L == 1 else 0.9 + 3.3 * (L ** 0.5 / 16.0)

def h_of(C):
    return max(0.55, 2.6 * (C / 72.0) ** 0.5)

fig, ax = plt.subplots(figsize=(15.5, 5.4))
X0, gap = 0.9, 0.95
x = X0
yc = 2.55

for st in stages:
    w, h = w_of(st["L"]), h_of(st["C"])
    y = yc - h / 2
    ax.add_patch(Rectangle((x, y), w, h, facecolor=st["color"], edgecolor="k", lw=1.0, alpha=0.95))
    n_lines = min(st["C"], 12)
    for i in range(1, n_lines):
        yl = y + h * i / n_lines
        ax.plot([x + 0.06, x + w - 0.06], [yl, yl], color="k", lw=0.25, alpha=0.4)
    ax.text(x + w / 2, y + h + 0.28, st["op"], ha="center", va="bottom",
            fontsize=8.6, fontweight="bold")
    ax.text(x + w / 2, y - 0.18, st["shape"], ha="center", va="top",
            fontsize=10, family="monospace", fontweight="bold")
    if x > X0:
        ax.add_patch(FancyArrowPatch((x - gap + 0.06, yc), (x - 0.06, yc),
                                     arrowstyle="-|>", mutation_scale=15, lw=1.6, color="#333"))
    x += w + gap

# Time axis under input
ax.add_patch(FancyArrowPatch((X0, 0.35), (X0 + w_of(256), 0.35), arrowstyle="-|>",
                             mutation_scale=13, lw=1.2, color="#c44d52"))
ax.text(X0 + w_of(256) / 2, 0.22, "Time axis: 256 steps (t = 0 … 255)",
        ha="center", fontsize=8.5, color="#c44d52")
ax.annotate("Each filter sees 7 consecutive\nsteps across ALL 4 channels",
            xy=(X0 + 1.0, yc + h_of(4) / 2 + 0.15), xytext=(2.4, 5.15),
            fontsize=8.5, color="#c44d52",
            arrowprops=dict(arrowstyle="->", color="#c44d52", lw=1.2))

ax.text(X0, 5.7, "1D-CNN Final Architecture (Optuna trial 20) — width = time (√-scaled), height = channels (√-scaled)",
        fontsize=12, fontweight="bold")
ax.text(X0, 5.35, "Time: 256 → 64 → 16 → global  |  Channels: 4 → 24 → 48 → 72  |  Meta features: sample_rate + group + level → FC input  |  196 KB ONNX",
        fontsize=9, color="#555")
ax.set_xlim(0, x)
ax.set_ylim(0, 6.0)
ax.axis("off")
fig.tight_layout()
fig.savefig(PLOTS / "23_final_architecture.png", bbox_inches="tight")
plt.close(fig)
print("[ok] experiments/plots/23_final_architecture.png", flush=True)