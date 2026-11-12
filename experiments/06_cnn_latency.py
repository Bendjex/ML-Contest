"""06: CNN latency + exact per-class score projection (training done in 05).

05 crashed at the final latency bench (len(s) bug, now fixed); the trained weights
are saved (models/cnn1_weights.npz) and the numpy export was verified (max diff
4.3e-06 vs torch). This script:

 1. reloads the weights, recomputes the val confusion matrix through the NUMBA-free
    numpy path (exact per-class accs),
 2. measures harness-style latency on the demo set (one call per sample,
    single-threaded),
 3. projects T_k = sqrt(n_k)*(acc_k - 0.33*err_k)*acc_k^2 and the geomean score.

Log: experiments/logs/cnn_latency.log
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

MLC_HOME = Path(os.environ.get("MLC_HOME", "/home/olorin/ml-contest-data/materials"))
HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
sys.path.insert(0, str(MLC_HOME))

from competition_api import load_dataset  # noqa: E402


def prep_batch(series: np.ndarray) -> np.ndarray:
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


def prep_one(series: np.ndarray) -> np.ndarray:
    x = np.array(series, dtype=np.float32)
    t = np.arange(x.shape[1])
    for ch in x:
        m = np.isnan(ch)
        if m.all():
            ch[:] = 0.0
        elif m.any():
            ch[m] = np.interp(t[m].astype(np.float64), t[~m], ch[~m].astype(np.float64))
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + np.float32(1e-9)
    return ((x - mu) / sd).astype(np.float32)


def conv1d_np(x, W, b, pad):
    xp = np.pad(x, ((0, 0), (pad, pad)))
    sw = np.lib.stride_tricks.sliding_window_view(xp, W.shape[2], axis=1)
    cols = sw.transpose(0, 2, 1).reshape(-1, sw.shape[1])
    return W.reshape(W.shape[0], -1) @ cols + b[:, None]


def maxpool_np(x, p):
    C, L = x.shape
    return x[:, :(L // p) * p].reshape(C, L // p, p).max(-1)


def forward_np(x, w):
    x = np.maximum(conv1d_np(x, w["c1_W"], w["c1_b"], 4), 0)
    x = maxpool_np(x, 4)
    x = np.maximum(conv1d_np(x, w["c2_W"], w["c2_b"], 4), 0)
    x = maxpool_np(x, 4)
    x = np.maximum(conv1d_np(x, w["c3_W"], w["c3_b"], 2), 0)
    pooled = np.concatenate([x.mean(axis=1), x.max(axis=1)])
    return w["fc_W"] @ pooled + w["fc_b"]


def main():
    w = {k: v for k, v in np.load(MODELS / "cnn1_weights.npz").items()}

    ds = load_dataset(MLC_HOME / "data" / "train_100k.npz")
    y = np.asarray(ds.labels, dtype=np.int64)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    val = order[:20_000]
    yv = y[val]
    t0 = time.perf_counter()
    Xv = prep_batch(ds.series[val])
    print(f"prep val: {Xv.shape} in {time.perf_counter()-t0:.0f}s", flush=True)

    # batched numpy forward over val (vectorized over samples)
    preds = np.empty(len(val), dtype=np.int64)
    t0 = time.perf_counter()
    for i in range(len(val)):
        preds[i] = int(np.argmax(forward_np(Xv[i], w)))
    dt = time.perf_counter() - t0
    acc = float((preds == yv).mean())
    print(f"numpy-path val_acc={acc:.4f} ({dt:.0f}s for 20k, {dt/len(val)*1e6:.0f} us/sample unthrottled)",
          flush=True)

    cm = np.zeros((8, 8), dtype=np.int64)
    np.add.at(cm, (yv, preds), 1)
    print("confusion (rows=true, cols=pred):", flush=True)
    for c in range(8):
        print("  " + " ".join(f"{v:5d}" for v in cm[c]), f" | class {c}", flush=True)
    per_class = [float((preds[yv == c] == c).mean()) for c in range(8)]
    print("per-class acc:", ["%.3f" % a for a in per_class], flush=True)

    # ---- harness-style latency, one call at a time, single thread ----
    from threadpoolctl import threadpool_limits
    demo = load_dataset(MLC_HOME / "data" / "test_demo.npz")
    samples = [demo[i] for i in range(len(demo))]

    def lat_of(fn):
        for s in samples[:20]:
            fn(s)
        t0 = time.perf_counter()
        for s in samples:
            fn(s)
        return (time.perf_counter() - t0) / len(samples)

    with threadpool_limits(limits=1):
        lat = lat_of(lambda s: int(np.argmax(forward_np(prep_one(s.series), w))))
        lat_fwd = lat_of(lambda s: forward_np(prep_one(s.series), w))

    print(f"\nLATENCY: {lat*1e6:.0f} us/call (fwd alone {lat_fwd*1e6:.0f} us)", flush=True)

    n_total = 60.0 / lat
    n_k = n_total / 8.0
    print(f"n(60 s) ~ {n_total:,.0f} | n_k ~ {n_k:,.0f}", flush=True)

    print(f"\nSCORE PROJECTION per class (alpha=0.5, p=2, lambda=0.33):", flush=True)
    print(f"{'class':>5} {'acc':>6} {'T_k':>8}", flush=True)
    Ts = []
    for c, a in enumerate(per_class):
        e = 1.0 - a
        T = (n_k ** 0.5) * (a - 0.33 * e) * a ** 2
        Ts.append(max(T, 0.01))
        print(f"{c:>5} {a:>6.3f} {T:>8.1f}", flush=True)
    gm = math.exp(sum(math.log(t) for t in Ts) / 8.0)
    print(f"\nGEOMEAN SCORE ~ {gm:.1f}  (tree d14: ~22, GBM: ~10)", flush=True)
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()