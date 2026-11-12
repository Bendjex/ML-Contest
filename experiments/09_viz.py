"""09: CNN-pass visualization (classic box diagram) + what-one-sample-looks-like.

Writes:
  experiments/plots/10_cnn_pass.png       the v5 architecture pass with boxes+shapes
  experiments/plots/11_data_sample.png   two raw samples (class 2 vs 6), 4 channels each
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch

MLC_HOME = Path(os.environ.get("MLC_HOME", "/home/olorin/ml-contest-data/materials"))
HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
sys.path.insert(0, str(MLC_HOME))

from competition_api import load_dataset  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

# ---------------------------------------------------------- 1) CNN pass diagram
stages = [
    dict(name="Input\n(roh)", shape="(4, 256)", w=3.2, h=1.1, color="#4c72b0", fc="#4c72b0"),
    dict(name="Conv1\n+BN +ReLU", shape="(16, 256)", w=3.2, h=2.2, color="#dd8452"),
    dict(name="MaxPool 4", shape="(16, 64)", w=2.0, h=2.2, color="#f5c08a"),
    dict(name="Conv2\n+BN +ReLU", shape="(32, 64)", w=2.0, h=3.0, color="#dd8452"),
    dict(name="MaxPool 4", shape="(32, 16)", w=1.3, h=3.0, color="#f5c08a"),
    dict(name="Conv3\n+BN +ReLU", shape="(64, 16)", w=1.4, h=3.8, color="#dd8452"),
    dict(name="Global\navg+max+std*", shape="(192,)*", w=0.55, h=3.8, color="#55a868"),
    dict(name="FC\n128\u21928", shape="(8,)", w=0.5, h=1.1, color="#8172b2"),
]
ops = [
    "Conv: 16 Filter, k=9, pad=4\n(16\u00b74\u00b79+16 = 592 par)",
    "\u00f7 4 auf der Zeitachse",
    "Conv: 32 Filter, k=9, pad=4\n(32\u00b716\u00b79+32 = 4.640 par)",
    "\u00f7 4 auf der Zeitachse",
    "Conv: 64 Filter, k=5\u2192k=9 dil=2*\n\u00fcber die ganze Serie",
    "mean + max + std\nje Kanal \u00fcber Zeit",
    "Matmul (192\u21928 = 1.544 par)*",
    "argmax \u2192 Klasse 0..7",
]

fig, ax = plt.subplots(figsize=(15.5, 5.6))
DEPTH = 0.45
yc = 3.1  # vertical center of boxes
x = 0.6
gap = 1.75
for i, st in enumerate(stages):
    w, h = st["w"], st["h"]
    x0, y0 = x, yc - h / 2
    # 3D cuboid: front, top, right
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=st["color"], edgecolor="k",
                            lw=1.0, alpha=0.92, zorder=3))
    ax.add_patch(Polygon([[x0, y0 + h], [x0 + DEPTH, y0 + h + DEPTH],
                          [x0 + w + DEPTH, y0 + h + DEPTH], [x0 + w, y0 + h]],
                         facecolor=st["color"], edgecolor="k", lw=1.0, alpha=0.65, zorder=2))
    ax.add_patch(Polygon([[x0 + w, y0], [x0 + w + DEPTH, y0 + DEPTH],
                          [x0 + w + DEPTH, y0 + h + DEPTH], [x0 + w, y0 + h]],
                         facecolor=st["color"], edgecolor="k", lw=1.0, alpha=0.75, zorder=2))
    ax.text(x0 + w / 2, y0 - 0.42, st["shape"], ha="center", va="top",
            fontsize=9.5, fontweight="bold", family="monospace")
    ax.text(x0 + w / 2, y0 + h / 2, st["name"], ha="center", va="center",
            fontsize=8.5, color="white" if st["color"] != "#4c72b0" else "white",
            fontweight="bold", zorder=4)
    if i < len(ops):
        ax.add_patch(FancyArrowPatch((x0 + w + 0.08, yc), (x0 + w + gap - 0.28, yc),
                                     arrowstyle="-|>", mutation_scale=16,
                                     lw=1.6, color="#333", zorder=1))
        ax.text(x0 + w + gap / 2, yc + 0.62, ops[i], ha="center", va="bottom",
                fontsize=8.2, color="#222")
    x += w + gap

ax.annotate("1 Filter sieht hier: 9 aufeinanderfolgende\nZeitschritte aus ALLEN 4 Kan\u00e4len (4\u00d79 = 36 Werte)",
            xy=(stages[0]["w"] + 0.6 + 1.3, yc - stages[0]["h"] / 2 - 0.55),
            xytext=(0.5, 0.45), fontsize=8.5, color="#c44d52",
            arrowprops=dict(arrowstyle="->", color="#c44d52", lw=1.3))
ax.text(0.6, 5.75, "1D-CNN Pass durchs Netz (v5 gemessen, v6-* = laufende \u00c4nderung):  (4, 256) \u2192 (8,)",
        fontsize=12, fontweight="bold")
ax.text(0.6, 5.35, "Kan\u00e4le wachsen 4\u219216\u219232\u219264 (Gemisch aus allen vorigen), Zeitachse schrumpft 256\u219264\u219216 (Pooling)  |  16.568 Parameter gesamt (v5)",
        fontsize=9.5, color="#444")
ax.set_xlim(0, x + 0.3)
ax.set_ylim(0, 6.3)
ax.axis("off")
fig.tight_layout()
fig.savefig(PLOTS / "10_cnn_pass.png", bbox_inches="tight")
plt.close(fig)
print("[1/2] cnn pass diagram ok", flush=True)

# ---------------------------------------------------------- 2) raw data samples
ds = load_dataset(MLC_HOME / "data" / "train_100k.npz")
y = np.asarray(ds.labels)
rng = np.random.default_rng(3)
i2 = int(rng.choice(np.where(y == 2)[0]))
i6 = int(rng.choice(np.where(y == 6)[0]))

fig, axes = plt.subplots(4, 2, figsize=(12.5, 7.2), sharex=True, sharey=True)
ch_names = ["Kanal 0", "Kanal 1", "Kanal 2", "Kanal 3"]
ch_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
for col, (idx, cls) in enumerate([(i2, 2), (i6, 6)]):
    s = np.asarray(ds.series[idx])
    meta = ds.meta_at(idx)
    nan_any = np.isnan(s)
    for ch in range(4):
        ax_ = axes[ch, col]
        ax_.plot(np.arange(256), s[ch], lw=0.75, color=ch_colors[ch])
        m = nan_any[ch]
        if m.any():
            ax_.scatter(np.where(m)[0], s[ch][~np.isnan(s[ch])][0:0].tolist() or [0] * int(m.sum()),
                        marker="x", s=16, color="k")
        ax_.set_ylabel("Rohwert", fontsize=8)
        ax_.set_title(f"{ch_names[ch]}  |  {cls=}, sample_rate={meta['sample_rate']:.0f}, "
                      f"group={meta['group']}, level={meta['level']:+.2f}"
                      if ch == 0 else ch_names[ch], fontsize=8.5, loc="left")
    axes[3, col].set_xlabel("Zeitschritt t (0 \u2026 255)  \u2014  die ZEITACHSE", fontsize=9)
fig.suptitle("So sieht EIN Sample aus: 4 Kan\u00e4le \u00d7 256 Zeitschritte  (links: Klasse 2 glatt, rechts: Klasse 6 tr\u00e4ge)",
             fontsize=11.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(PLOTS / "11_data_sample.png", bbox_inches="tight")
plt.close(fig)
print("[2/2] data sample ok", flush=True)
print("=== ALL DONE ===", flush=True)