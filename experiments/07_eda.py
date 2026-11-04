"""07: EDA — visual data analysis of the contest dataset (train_100k + test_demo).

Writes 8 PNGs to experiments/plots/:
  01 class distribution (train imbalanced vs demo balanced)
  02 meta per class (sample_rate share, group share, level hist)
  03 one example window per class (4 channels, z-scored, NaN marked)
  04 mean power spectrum per class (log)
  05 mean autocorrelation per class (lags 1..64)
  06 per-sample NaN fraction + per class
  07 per-class per-channel raw std
  08 CNN val confusion matrix (from models/cnn1_weights.npz)

Run: bash experiments/run_07.sh   Log: experiments/logs/eda.log
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MLC_HOME = Path(os.environ.get("MLC_HOME", "/home/olorin/ml-contest-data/materials"))
HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
PLOTS = HERE / "plots"
sys.path.insert(0, str(MLC_HOME))

from competition_api import load_dataset  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.titlesize": 10,
                     "axes.grid": True, "grid.alpha": 0.3})
CH_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
CLS_COLORS = plt.cm.tab10(np.linspace(0, 0.78, 8))


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
    return np.where(np.isnan(Xb), np.float32(0), Xb)


def z_batch(Xb):
    mu = Xb.mean(2, keepdims=True)
    sd = Xb.std(2, keepdims=True) + np.float32(1e-9)
    return ((Xb - mu) / sd).astype(np.float32)


def forward_np(x, w):
    def conv1d_np(x, W, b, pad):
        xp = np.pad(x, ((0, 0), (pad, pad)))
        sw = np.lib.stride_tricks.sliding_window_view(xp, W.shape[2], axis=1)
        cols = sw.transpose(0, 2, 1).reshape(-1, sw.shape[1])
        return W.reshape(W.shape[0], -1) @ cols + b[:, None]
    def maxpool_np(x, p):
        C, L = x.shape
        return x[:, :(L // p) * p].reshape(C, L // p, p).max(-1)
    x = np.maximum(conv1d_np(x, w["c1_W"], w["c1_b"], 4), 0)
    x = maxpool_np(x, 4)
    x = np.maximum(conv1d_np(x, w["c2_W"], w["c2_b"], 4), 0)
    x = maxpool_np(x, 4)
    x = np.maximum(conv1d_np(x, w["c3_W"], w["c3_b"], 2), 0)
    pooled = np.concatenate([x.mean(axis=1), x.max(axis=1)])
    return w["fc_W"] @ pooled + w["fc_b"]


def main():
    PLOTS.mkdir(exist_ok=True, parents=True)
    t0 = time.perf_counter()
    ds = load_dataset(MLC_HOME / "data" / "train_100k.npz")
    demo = load_dataset(MLC_HOME / "data" / "test_demo.npz")
    y = np.asarray(ds.labels)
    print(f"train: {ds.series.shape} {ds.series.dtype} | demo: {demo.series.shape}", flush=True)

    rng = np.random.default_rng(7)

    # ---------------- 01 class distribution ----------------
    counts = np.bincount(y, minlength=8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].bar(range(8), counts, color=CLS_COLORS)
    for c, v in enumerate(counts):
        axes[0].annotate(f"{v:,}", (c, v), ha="center", va="bottom", fontsize=8)
    axes[0].set(title="train_100k: samples per class (MILDLY imbalanced, 1.62x)",
                xlabel="class", ylabel="count", xticks=range(8))
    dc = np.bincount(np.asarray(demo.labels), minlength=8)
    axes[1].bar(range(8), dc, color=CLS_COLORS)
    axes[1].set(title="test_demo: 400 samples, EXACTLY balanced (eval-like)",
                xlabel="class", ylabel="count", xticks=range(8), ylim=(0, 60))
    fig.tight_layout()
    fig.savefig(PLOTS / "01_class_distribution.png")
    plt.close(fig)
    print(f"[1/8] class distribution ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---------------- 02 meta ----------------
    sr = np.asarray(ds.meta["sample_rate"], float)
    grp = np.asarray(ds.meta["group"])
    lev = np.asarray(ds.meta["level"], float)
    print(f"meta: sample_rate {sorted(set(sr.tolist()))}, group {sorted(set(grp.tolist()))}, "
          f"level [{lev.min():.2f}, {lev.max():.2f}]", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    sr2 = [float(np.mean(sr[y == c] == 2.0)) for c in range(8)]
    axes[0].bar(range(8), sr2, color=CLS_COLORS)
    axes[0].set(title="sample_rate = 2 share per class (~40% everywhere)", xlabel="class",
                ylabel="share", ylim=(0, 0.7), xticks=range(8))
    w = 0.27
    for gi, g in enumerate(("A", "B", "C")):
        shares = [float(np.mean(grp[y == c] == g)) for c in range(8)]
        axes[1].bar(np.arange(8) + (gi - 1) * w, shares, width=w, label=f"group {g}")
    axes[1].set(title="group share per class (~uniform, nuisance)", xlabel="class",
                ylabel="share", ylim=(0, 0.7), xticks=range(8))
    axes[1].legend(fontsize=8)
    for c in range(8):
        axes[2].hist(lev[y == c], bins=60, range=(-7, 7), histtype="step",
                      color=CLS_COLORS[c], lw=1)
    axes[2].set(title="level distribution per class (~N(0,2), nuisance)",
                xlabel="level", ylabel="samples/bin")
    fig.tight_layout()
    fig.savefig(PLOTS / "02_meta_overview.png")
    plt.close(fig)
    print(f"[2/8] meta overview ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---------------- 03 examples ----------------
    Xb = fill_batch(ds.series)
    Z = z_batch(Xb)
    fig, axes = plt.subplots(4, 2, figsize=(11, 12.5), sharex=True)
    for c in range(8):
        ax = axes.flat[c]
        i = int(rng.choice(np.where(y == c)[0]))
        for ch in range(4):
            ax.plot(Z[i, ch], lw=0.7, color=CH_COLORS[ch], label=f"ch{ch}")
        nans = np.isnan(np.asarray(ds.series[i])).any(0)
        if nans.any():
            ax.scatter(np.where(nans)[0], np.zeros(int(nans.sum())), marker="x",
                       s=14, color="k", label="NaN")
        acc = np.mean(np.argmax([0]) * 0)  # placeholder no-op
        ax.set(title=f"class {c}  (sample {i}, z-scored per channel)", xticks=[])
    axes.flat[0].legend(fontsize=7, ncol=5)
    fig.suptitle("one random example per class", y=0.995)
    fig.tight_layout()
    fig.savefig(PLOTS / "03_examples_per_class.png")
    plt.close(fig)
    print(f"[3/8] examples ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---------------- 04 spectra + 05 acf (subsample) ----------------
    sub = np.concatenate([rng.choice(np.where(y == c)[0], 2000, replace=False)
                          for c in range(8)])
    Zs, ys = Z[sub], y[sub]
    P = np.abs(np.fft.rfft(Zs, axis=2)) ** 2
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for c in range(8):
        p = P[ys == c].mean(axis=(0, 1))
        ax.plot(range(1, 65), p[1:65] + 1e-4, lw=1.1, color=CLS_COLORS[c], label=f"class {c}")
    ax.set(yscale="log",
           title="mean power spectrum per class (z-scored; classes separate at LOW freqs)",
           xlabel="k = cycles per 256-step window (k=13 ~ 0.05 cyc/step)",
           ylabel="mean |FFT|^2 (log)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS / "04_mean_power_spectrum.png")
    plt.close(fig)
    print(f"[4/8] spectra ({time.perf_counter()-t0:.0f}s)", flush=True)

    lags = np.arange(1, 65)
    acf = np.zeros((8, len(lags)))
    for c in range(8):
        z = Zs[ys == c]
        for li, lag in enumerate(lags):
            acf[c, li] = np.mean(z[:, :, :-lag] * z[:, :, lag:])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for c in range(8):
        ax.plot(lags, acf[c], lw=1.1, color=CLS_COLORS[c], label=f"class {c}")
    ax.set(title="mean autocorrelation per class: class 2 slow, 1 low, 6/7 high at lag 8-16",
           xlabel="lag (steps)", ylabel="acf")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS / "05_autocorrelation.png")
    plt.close(fig)
    print(f"[5/8] acf ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---------------- 06 NaN ----------------
    nanfrac = np.isnan(ds.series).mean(axis=(1, 2))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    axes[0].hist(nanfrac, bins=80, color="#1f77b4")
    axes[0].set(title=f"per-sample NaN fraction (mean {nanfrac.mean():.4f}, max {nanfrac.max():.3f})",
                xlabel="NaN fraction of the 4x256 window", ylabel="samples")
    percls = [float(nanfrac[y == c].mean()) for c in range(8)]
    axes[1].bar(range(8), percls, color=CLS_COLORS)
    axes[1].set(title="mean NaN fraction per class (uniform ~1.3-1.7%)",
                 xlabel="class", ylabel="mean frac", ylim=(0, 0.03), xticks=range(8))
    fig.tight_layout()
    fig.savefig(PLOTS / "06_nan_fraction.png")
    plt.close(fig)
    print(f"[6/8] nan ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---------------- 07 raw scale per class/channel ----------------
    stds = np.array([Xb[y == c].std(axis=(0, 2)) for c in range(8)])  # (8, 4)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.2
    for ch in range(4):
        ax.bar(np.arange(8) + (ch - 1.5) * w, stds[:, ch], width=w,
               color=CH_COLORS[ch], label=f"ch{ch}")
    ax.set(title="raw per-channel std per class (level shifts included)",
           xlabel="class", ylabel="std", xticks=range(8))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "07_raw_std_per_class.png")
    plt.close(fig)
    print(f"[7/8] raw std ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---------------- 08 confusion (CNN numpy path, val split) ----------------
    wts = {k: v for k, v in np.load(MODELS / "cnn1_weights.npz").items()}
    order = np.random.default_rng(0).permutation(len(y))
    val = order[:20_000]
    Xv, yv = Z[val], y[val]
    preds = np.empty(len(val), dtype=np.int64)
    t1 = time.perf_counter()
    for i in range(len(val)):
        preds[i] = int(np.argmax(forward_np(Xv[i], wts)))
    print(f"forward 20k: {time.perf_counter()-t1:.0f}s", flush=True)
    acc = float((preds == yv).mean())
    cm = np.zeros((8, 8), dtype=np.int64)
    np.add.at(cm, (yv, preds), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(8):
        for j in range(8):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black")
    ax.set(title=f"CNN val confusion (n=20k, acc={acc:.3f}) rows=true, cols=pred",
           xlabel="predicted", ylabel="true", xticks=range(8), yticks=range(8))
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(PLOTS / "08_confusion_matrix.png")
    plt.close(fig)
    print(f"[8/8] confusion done ({time.perf_counter()-t0:.0f}s)", flush=True)

    print("\nTEXT SUMMARY", flush=True)
    print(f"  series: {ds.series.shape} {ds.series.dtype}, ~{nanfrac.mean()*100:.1f}% NaN", flush=True)
    print(f"  classes: {[int(v) for v in counts]} (ratio {counts.max()/counts.min():.2f}x)", flush=True)
    print(f"  CNN val acc: {acc:.4f}", flush=True)
    print(f"  weak pairs (confusions >200): "
          f"{[(int(cm[i, j]), i, j) for i in range(8) for j in range(8) if cm[i, j] > 200 and i != j]}",
          flush=True)
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()