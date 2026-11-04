"""12: Flaches 2D-Architektur-Diagramm (keine 3D-Tiefe) — Breite=Zeit, Höhe=Kanäle.

Ehrlich skaliert: Breiten √-skaliert über die Zeitschritte (256 vs 16 nicht 16:1
sondern 4:1), Höhen √-skaliert über Kanäle. Keine isometrische Deko-Tiefe:
pro Sample ist alles 2D — (Kanäle, Zeit).
Schreibt experiments/plots/12_cnn_pass_flat.png
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
    dict(op="Input (roh)",              shape="(4, 256)",  L=256, C=4,  color="#8da2c4"),
    dict(op="Conv1 +BN +ReLU",          shape="(16, 256)", L=256, C=16, color="#e08a4e"),
    dict(op="MaxPool ÷4",               shape="(16, 64)",  L=64,  C=16, color="#f2c088"),
    dict(op="Conv2 +BN +ReLU",          shape="(32, 64)",  L=64,  C=32, color="#e08a4e"),
    dict(op="MaxPool ÷4",               shape="(32, 16)",  L=16,  C=32, color="#f2c088"),
    dict(op="Conv3 +BN +ReLU (dil=2)",  shape="(64, 16)",  L=16,  C=64, color="#e08a4e"),
    dict(op="Global pool mean+max+std", shape="(192,)",    L=1,   C=192, color="#7fae7f"),
    dict(op="FC → argmax",              shape="(8,)",      L=1,   C=8,   color="#9b86c6"),
]

def w_of(L):   # sqrt-skalierte Breite
    return 1.6 if L == 1 else 0.9 + 3.3 * (L ** 0.5 / 16.0)

def h_of(C):   # sqrt-skalierte Höhe
    return max(0.55, 2.6 * (C / 64.0) ** 0.5)

fig, ax = plt.subplots(figsize=(15.5, 5.0))
X0, gap = 0.9, 0.95
x = X0
yc = 2.55  # Grundlinie der Boxen (Unterkante)
for st in stages:
    w, h = w_of(st["L"]), h_of(st["C"])
    y = yc - h / 2
    ax.add_patch(Rectangle((x, y), w, h, facecolor=st["color"], edgecolor="k", lw=1.0, alpha=0.95))
    n_lines = min(st["C"], 12)
    for i in range(1, n_lines):  # Kanal-Linien andeuten
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

# Zeitachse unter dem Input annotieren
ax.add_patch(FancyArrowPatch((X0, 0.35), (X0 + w_of(256), 0.35), arrowstyle="-|>",
                             mutation_scale=13, lw=1.2, color="#c44d52"))
ax.text(X0 + w_of(256) / 2, 0.22, "Zeitachse: 256 Schritte  (t = 0 … 255)",
        ha="center", fontsize=8.5, color="#c44d52")
ax.annotate("1 Filter = 9 aufeinanderfolgende\nZeitschritte × ALLE 4 Kanäle",
            xy=(X0 + 1.0, yc + h_of(4) / 2 + 0.15), xytext=(2.4, 5.05),
            fontsize=8.5, color="#c44d52",
            arrowprops=dict(arrowstyle="->", color="#c44d52", lw=1.2))

ax.text(X0, 5.6, "1D-CNN v6 — flach:  Breite = Zeit (√-skaliert),  Höhe = Kanäle (√-skaliert)  ·  pro Sample 2D: (Kanäle, Zeit)",
        fontsize=12, fontweight="bold")
ax.text(X0, 5.25, "Zeit: 256 → 64 → 16 → global   ·   Kanäle: 4 → 16 → 32 → 64   ·   Breiten/Höhen nicht linear, sonst wäre Conv3 unsichtbar",
        fontsize=9, color="#555")
ax.set_xlim(0, x)
ax.set_ylim(0, 5.9)
ax.axis("off")
fig.tight_layout()
fig.savefig(PLOTS / "12_cnn_pass_flat.png", bbox_inches="tight")
print("[ok] experiments/plots/12_cnn_pass_flat.png", flush=True)