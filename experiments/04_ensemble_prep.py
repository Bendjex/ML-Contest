"""04: Tree ensembles (RF/ET) + rate-aware preprocessing — Nils' two questions.

A) Ensemble idea: Random Forest / ExtraTrees on the same features as the single tree.
   Pay per-tree traversal latency (single-thread!), hope for +8-10pp accuracy.
   Reference point: GBM = 300 iterations x 8 classes = 2400 shallow trees = 31 ms/call.
B) Preprocessing idea: rate-aware features. meta sample_rate (1/2 Hz) means series are
   sampled at different physical rates -> ACF lags in STEPS, FFT bands in ARRAY bins
   and roughness per STEP are all rate-confounded. Fix: ACF at lags in SECONDS
   (step = lag_s * sr), spectra interpolated onto a fixed HZ grid, diff over ~1 s.
   Plus: group one-hot (baseline drops it), per-channel NaN fractions,
   cross-channel correlations at +/- 4 s, channel means (pre-standardization).
   Diagnostic first: dominant frequency per class x sample_rate — if fdom(sr=2) ~=
   2 x fdom(sr=1), the generator produces the same processes at different rates
   and alignment must help.

Everything on the fixed 20k val split (rng seed 0). Latency measured harness-style:
one predict call per demo sample, single-threaded.

Log: experiments/logs/ensemble_prep.log
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

MLC_HOME = Path(os.environ.get("MLC_HOME", "/home/olorin/ml-contest-data/materials"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(MLC_HOME))
sys.path.insert(0, str(MLC_HOME / "examples"))

from competition_api import load_dataset  # noqa: E402
from baseline_model import features as base_features  # noqa: E402

HZ_GRID = np.linspace(0.0, 0.45, 33)
ACF_SECONDS = (1, 2, 4, 8, 16)
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def fill_interp(series: np.ndarray) -> np.ndarray:
    out = np.array(series, dtype=np.float64)
    t = np.arange(out.shape[1])
    for ch in out:
        m = np.isnan(ch)
        if m.all():
            ch[:] = 0.0
        elif m.any():
            ch[m] = np.interp(t[m], t[~m], ch[~m])
    return out


def feats_upg(series: np.ndarray, sr: float, group: str) -> np.ndarray:
    raw = fill_interp(series)
    z = (raw - raw.mean(1, keepdims=True)) / (raw.std(1, keepdims=True) + 1e-9)
    C, L = z.shape
    parts = [
        raw.mean(1), raw.std(1),                                   # 8 level/scale
        np.percentile(z, (5, 25, 50, 75, 95), axis=1).T.ravel(),    # 20 quantiles
    ]
    acf = []
    for lag_s in ACF_SECONDS:
        step = max(1, int(round(lag_s * sr)))
        acf.append(np.mean(z[:, :-step] * z[:, step:], axis=1) if step < L else np.zeros(C))
    parts.append(np.stack(acf, 1).ravel())                          # 20 acf in seconds
    P = np.abs(np.fft.rfft(z, axis=1)) ** 2
    f = np.fft.rfftfreq(L, d=1.0 / sr)                              # Hz
    logs = np.log1p(np.stack([np.interp(HZ_GRID, f, P[c]) for c in range(C)]))
    bands = np.array_split(logs, 8, axis=1)
    parts.append(np.stack([b.sum(1) for b in bands], 1).ravel())     # 32 spectral in Hz
    step = max(1, int(round(sr)))
    d1 = z[:, :-step] - z[:, step:]
    parts.append(np.array([d1.std(1), np.abs(d1).mean(1)]).ravel())  # 8 roughness per ~1s
    lag = max(1, int(round(4 * sr)))
    cc = []
    for i, j in PAIRS:
        cc.append(np.mean(z[i] * z[j]))
        if lag < L:
            cc.append(np.mean(z[i, :-lag] * z[j, lag:]))
            cc.append(np.mean(z[i, lag:] * z[j, :-lag]))
        else:
            cc += [0.0, 0.0]
    parts.append(np.array(cc))                                      # 18 lagged cross-corr
    parts.append(np.array([group == "A", group == "B", group == "C"], float))  # 3
    parts.append(np.isnan(series).mean(1))                          # 4 per-channel NaN
    parts.append(np.array([np.isnan(series).mean()]))               # 1 global NaN
    return np.concatenate([np.asarray(p, float).ravel() for p in parts])


def dominant_hz(z: np.ndarray, sr: float) -> float:
    P = np.abs(np.fft.rfft(z, axis=1)) ** 2
    f = np.fft.rfftfreq(z.shape[1], d=1.0 / sr)
    return float(f[1:][np.argmax(P[:, 1:].mean(0))])


def lat_of(fn, samples, warm=20):
    for s in samples[:warm]:
        fn(s)
    t0 = time.perf_counter()
    for s in samples:
        fn(s)
    return (time.perf_counter() - t0) / len(samples)


def main():
    ds = load_dataset(MLC_HOME / "data" / "train_100k.npz")
    y = np.asarray(ds.labels)
    sr = np.asarray(ds.meta["sample_rate"], dtype=float)
    grp = np.asarray(ds.meta["group"])

    t0 = time.perf_counter()
    F_base = np.stack([base_features(s) for s in ds])
    print(f"baseline features: {F_base.shape} in {time.perf_counter()-t0:.0f}s", flush=True)
    t0 = time.perf_counter()
    F_upg = np.stack([feats_upg(ds.series[i], sr[i], grp[i]) for i in range(len(ds))])
    print(f"upgraded features: {F_upg.shape} in {time.perf_counter()-t0:.0f}s", flush=True)

    # ---- diagnostic: rate confound ----
    print("\nDIAGNOSTIC dominant frequency (Hz) per class x sample_rate:", flush=True)
    rng0 = np.random.default_rng(1)
    sub = np.concatenate([rng0.choice(np.where(y == c)[0], 400, replace=False) for c in range(8)])
    for c in range(8):
        row = []
        for s in (1.0, 2.0):
            idx = [i for i in sub if y[i] == c and sr[i] == s][:150]
            doms = [dominant_hz(fill_interp(ds.series[i]), sr[i]) for i in idx]
            row.append(np.mean(doms))
        print(f"  class {c}: sr=1 {row[0]:.3f} Hz | sr=2 {row[1]:.3f} Hz | ratio {row[1]/max(row[0],1e-9):.2f}", flush=True)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    n_val = 20_000
    val, tr = order[:n_val], order[n_val:]
    yv = y[val]

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    def evaluate(name, mk, F):
        t0 = time.perf_counter()
        m = mk().fit(F[tr], y[tr])
        fit_t = time.perf_counter() - t0
        pred = m.predict(F[val])
        acc = float((pred == yv).mean())
        worst = min(float((pred[yv == c] == c).mean()) for c in range(8))
        print(f"{name:34s} val_acc={acc:.3f} worst_class={worst:.3f} fit={fit_t:.0f}s", flush=True)
        return m, acc

    results = {}
    for tag, F in (("base", F_base), ("upg", F_upg)):
        print(f"\n=== feature set: {tag} ===", flush=True)
        results[f"tree_{tag}"] = evaluate(f"tree d14 [{tag}]",
            lambda: DecisionTreeClassifier(max_depth=14, random_state=0), F)
        results[f"rf_{tag}"] = evaluate(f"RandomForest(100,d14) [{tag}]",
            lambda: RandomForestClassifier(n_estimators=100, max_depth=14,
                                           max_features="sqrt", n_jobs=1,
                                           class_weight="balanced_subsample", random_state=0), F)
        results[f"et_{tag}"] = evaluate(f"ExtraTrees(100,d16) [{tag}]",
            lambda: ExtraTreesClassifier(n_estimators=100, max_depth=16,
                                         max_features="sqrt", n_jobs=1,
                                         class_weight="balanced", random_state=0), F)
        results[f"gbm_{tag}"] = evaluate(f"HistGBM(300,bal) [{tag}]",
            lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                   class_weight="balanced", random_state=0), F)

    # ---- latency, harness style, single thread ----
    demo = load_dataset(MLC_HOME / "data" / "test_demo.npz")
    samples = [demo[i] for i in range(len(demo))]

    def score_proj(acc, lat, budget=60.0, k=8):
        n_k = budget / lat / k
        err = 1 - acc
        return n_k ** 0.5 * (acc - 0.33 * err) * acc ** 2

    print("\nLATENCY (single-thread, one call at a time) + SCORE PROJECTION:", flush=True)
    with threadpool_limits(limits=1):
        lat_fb = lat_of(lambda s: base_features(s), samples)
        lat_fu = lat_of(lambda s: feats_upg(s.series, s.meta["sample_rate"], s.meta["group"]), samples)
        print(f"  feature cost: base {lat_fb*1e6:.0f} us | upgraded {lat_fu*1e6:.0f} us", flush=True)
        for tag, lat_feat in (("base", lat_fb), ("upg", lat_fu)):
            for mname in ("tree", "rf", "et", "gbm"):
                model, acc = results[f"{mname}_{tag}"]
                tot = lat_of(lambda s, m=model: int(m.predict(
                    (base_features(s) if tag == "base" else
                     feats_upg(s.series, s.meta["sample_rate"], s.meta["group"])
                    )[np.newaxis, :])[0]), samples)
                print(f"  {mname}({tag}): {tot*1e6:7.0f} us/call acc={acc:.3f} "
                      f"-> score ~ {score_proj(acc, tot):.1f}", flush=True)

    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()