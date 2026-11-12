"""23: Data sample plot — one example per class (all 8 classes), 4 channels each.

Style matches 11_data_sample.png: raw z-scored windows, 4 channel lines overlaid,
NaN positions marked, per class. 4 rows x 2 cols, English labels.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MLC_HOME = Path(os.environ.get("MLC_HOME", "/home/olorin/ml-contest-data/materials"))
HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
sys.path.insert(0, str(MLC_HOME))

from competition_api import load_dataset  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
CH_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]


def fill_batch(series):
    X = np.array(series, dtype=np.float32)
    N, C, L = X.shape
    rows = np.arange(N)[:, None, None]
    cols = np.arange(C)[None, :, None]
    idx = np.where(~np.isnan(X), np.arange(L, dtype=np.int32)[None, None, :], 0)
    np.maximum.accumulate(idx, axis=2, out=idx)
    Xf = X[rows, cols, idx]
    idxb = np.where(~np.isnan(Xf[:, :, ::-1]), np.arange(L, dtype=np.int32)[None, None, :], 0)
    np.maximum.accumulate(idxb, axis=2, out=idxb)
    Xb = Xf[rows, cols, idxb[:, :, ::-1]]
    Xb = np.where(np.isnan(Xb), np.float32(0), Xb)
    mu = Xb.mean(axis=2, keepdims=True)
    sd = Xb.std(axis=2, keepdims=True) + np.float32(1e-9)
    return ((Xb - mu) / sd).astype(np.float32)


def main():
    ds = load_dataset(MLC_HOME / "data" / "train_100k.npz")
    y = np.asarray(ds.labels)
    Z = fill_batch(ds.series)  # full 100k filled+z-scored
    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True, sharey=True)
    ch_names = ["ch0", "ch1", "ch2", "ch3"]

    for c in range(8):
        ax = axes.flat[c]
        i = int(rng.choice(np.where(y == c)[0]))
        for ch in range(4):
            ax.plot(np.arange(256), Z[i, ch] + ch * 0.5, lw=0.6, color=CH_COLORS[ch],
                    label=f"ch{ch}" if c == 0 else "")
        nans = np.isnan(np.asarray(ds.series[i])).any(0)
        if nans.any():
            ax.scatter(np.where(nans)[0], np.zeros(int(nans.sum())) + 1.5,
                       marker="x", s=14, color="k", label="NaN")
        ax.set_title(f"class {c}", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 255)
        if c >= 6:
            ax.set_xlabel("Time step t (0 … 255)", fontsize=9)
        if c % 2 == 0:
            ax.set_ylabel("z-score + channel offset", fontsize=9)

    axes.flat[0].legend(fontsize=7, ncol=5, loc="upper right")
    fig.suptitle("One random z-scored sample per class — 4 channels (shifted vertically for visibility), NaN markers (black ×)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(PLOTS / "24_one_per_class.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] experiments/plots/24_one_per_class.png", flush=True)


if __name__ == "__main__":
    main()