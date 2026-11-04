"""MLP v2 + baseline retraining: the full comparison in one detached run.

Trains (3 seeds x 300 iters) raw-window MLPs, rebuilds tree/GBM baselines,
evaluates the 3-seed ensemble, and measures harness-style latency for every
path (sklearn predict, pure-numpy forward, baseline features+tree/GBM).

Run:  bash experiments/run_v2.sh   (detached; log: experiments/logs/v2.log)
Deps: /tmp/mlc-venv (numpy, scikit-learn, threadpoolctl), MLC_HOME materials.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

MLC_HOME = Path(os.environ.get("MLC_HOME", "/home/olorin/ml-contest-data/materials"))
HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
sys.path.insert(0, str(MLC_HOME))
sys.path.insert(0, str(MLC_HOME / "examples"))

from competition_api import load_dataset  # noqa: E402

MAX_ITER = 300
SEEDS = (0, 1, 2)


# --------------------------------------------------------------- preprocessing
def prep_batch(series: np.ndarray) -> np.ndarray:
    """(N,C,L) -> (N, C*L) float32: NaN ffill/bfill (all-NaN->0), per-channel z-score."""
    X = np.array(series, dtype=np.float32)
    N, C, L = X.shape
    rows = np.arange(N)[:, None, None]
    cols = np.arange(C)[None, :, None]
    idx = np.where(~np.isnan(X), np.arange(L, dtype=np.int32)[None, None, :], 0)
    np.maximum.accumulate(idx, axis=2, out=idx)
    Xf = X[rows, cols, idx]                                    # forward fill
    idxb = np.where(~np.isnan(Xf[:, :, ::-1]), np.arange(L, dtype=np.int32)[None, None, :], 0)
    np.maximum.accumulate(idxb, axis=2, out=idxb)
    Xb = Xf[rows, cols, idxb[:, :, ::-1]]                       # backward fill
    Xb = np.where(np.isnan(Xb), np.float32(0), Xb)              # all-NaN channel -> 0
    mu = Xb.mean(axis=2, keepdims=True)
    sd = Xb.std(axis=2, keepdims=True) + np.float32(1e-9)
    return (((Xb - mu) / sd).astype(np.float32)).reshape(N, C * L)


def prep_one(series: np.ndarray) -> np.ndarray:
    """predict()-path, one (C,L) sample: NaN linear-interp, per-channel z-score."""
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
    return ((x - mu) / sd).ravel()


# --------------------------------------------------------------- numpy forward
def forward(x: np.ndarray, coefs, inters) -> np.ndarray:
    a = x
    last = len(coefs) - 1
    for i, (w, b) in enumerate(zip(coefs, inters)):
        a = a @ w + b
        if i < last:
            np.maximum(a, 0.0, out=a)
    return a


# --------------------------------------------------------------- scoring
def score_projection(acc: float, lat_s: float, budget_s: float = 60.0, k: int = 8) -> float:
    """T_k = sqrt(n_k) * (acc - 0.33*err) * acc^2, balanced classes, geomean = T_k."""
    n = budget_s / lat_s
    n_k = n / k
    err = 1.0 - acc
    return float(n_k ** 0.5 * (acc - 0.33 * err) * acc ** 2)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(MLC_HOME / "data" / "train_100k.npz")
    y = np.asarray(ds.labels)
    t0 = time.perf_counter()
    X = prep_batch(ds.series)
    print(f"prep_batch: {X.shape} in {time.perf_counter()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    n_val = 20_000
    val, tr = order[:n_val], order[n_val:]
    yv = y[val]

    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    # ---------------- MLPs ----------------
    nets, accs = {}, {}
    for seed in SEEDS:
        t0 = time.perf_counter()
        net = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=MAX_ITER,
                            random_state=seed, early_stopping=False)
        net.fit(X[tr], y[tr])
        fit_t = time.perf_counter() - t0
        accs[seed] = float((net.predict(X[val]) == yv).mean())
        converged = len(net.loss_curve_) < MAX_ITER
        print(f"mlp s{seed}: val_acc={accs[seed]:.3f} iters={len(net.loss_curve_)} "
              f"{'CONVERGED' if converged else 'MAX-ITERS'} fit={fit_t:.0f}s", flush=True)
        nets[seed] = net
        with open(MODELS / f"mlp300_s{seed}.pkl", "wb") as fh:
            pickle.dump(net, fh)

    # numpy weights for the fast inference path
    W = {}
    for seed in SEEDS:
        for i, (c, b) in enumerate(zip(nets[seed].coefs_, nets[seed].intercepts_)):
            W[f"c{seed}_{i}"] = np.asarray(c, dtype=np.float32)
            W[f"b{seed}_{i}"] = np.asarray(b, dtype=np.float32)
    np.savez(MODELS / "mlp300_weights.npz", **W)

    # ---------------- 3-seed ensemble ----------------
    proba = np.mean([nets[s].predict_proba(X[val]) for s in SEEDS], axis=0)
    ens_pred = np.argmax(proba, axis=1)
    ens_acc = float((ens_pred == yv).mean())
    per_class = [float((ens_pred[yv == c] == c).mean()) for c in range(8)]
    print(f"ENSEMBLE(3x mlp300): val_acc={ens_acc:.3f} per_class="
          f"{['%.2f' % a for a in per_class]}", flush=True)

    # ---------------- baselines (same split) ----------------
    tree = DecisionTreeClassifier(max_depth=14, random_state=0).fit(X[tr], y[tr])
    tree_acc = float((tree.predict(X[val]) == yv).mean())
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=0).fit(X[tr], y[tr])
    gbm_acc = float((gbm.predict(X[val]) == yv).mean())
    with open(MODELS / "tree_d14.pkl", "wb") as fh:
        pickle.dump(tree, fh)
    with open(MODELS / "gbm300.pkl", "wb") as fh:
        pickle.dump(gbm, fh)
    print(f"baselines: tree_d14 val_acc={tree_acc:.3f} | gbm300 val_acc={gbm_acc:.3f}", flush=True)

    # ---------------- harness-style latency (single-thread, one call at a time) ----
    demo = load_dataset(MLC_HOME / "data" / "test_demo.npz")
    n = len(demo)
    samples = [demo[i] for i in range(n)]
    from baseline_model import features as base_features  # noqa: E402

    ws = {k: v for k, v in np.load(MODELS / "mlp300_weights.npz").items()}
    coefs = {s: [ws[f"c{s}_0"], ws[f"c{s}_1"], ws[f"c{s}_2"]] for s in SEEDS}
    inters = {s: [ws[f"b{s}_0"], ws[f"b{s}_1"], ws[f"b{s}_2"]] for s in SEEDS}

    def lat_of(fn):
        for s in samples[:20]:
            fn(s)
        t0 = time.perf_counter()
        for s in samples:
            fn(s)
        return (time.perf_counter() - t0) / n

    with threadpool_limits(limits=1):
        lat_prep = lat_of(lambda s: prep_one(s.series))
        lat_mlp_np = lat_of(lambda s: int(np.argmax(forward(prep_one(s.series), coefs[0], inters[0]))))
        def ens_fwd(s):
            v = prep_one(s.series)
            out = forward(v, coefs[0], inters[0])
            for sd_ in SEEDS[1:]:
                out += forward(v, coefs[sd_], inters[sd_])
            return int(np.argmax(out))
        lat_ens_np = lat_of(ens_fwd)
        lat_sk = lat_of(lambda s: int(nets[0].predict(prep_one(s.series)[None, :])[0]))
        lat_tree = lat_of(lambda s: int(tree.predict(base_features(s)[np.newaxis, :])[0]))
        lat_gbm = lat_of(lambda s: int(gbm.predict(base_features(s)[np.newaxis, :])[0]))

    print(f"\nLATENCY (single-thread, one call at a time):", flush=True)
    print(f"  prep_one            {lat_prep*1e6:8.0f} us", flush=True)
    print(f"  mlp numpy forward   {lat_mlp_np*1e6:8.0f} us  (incl prep)", flush=True)
    print(f"  mlp ensemble(3) np  {lat_ens_np*1e6:8.0f} us  (incl prep)", flush=True)
    print(f"  mlp sklearn predict  {lat_sk*1e6:8.0f} us  (incl prep)", flush=True)
    print(f"  tree d14 + features  {lat_tree*1e6:8.0f} us", flush=True)
    print(f"  gbm300 + features    {lat_gbm*1e6:8.0f} us", flush=True)

    # ---------------- score projections ----------------
    print(f"\nSCORE PROJECTIONS (T_k ~ sqrt(n_k)*(acc-0.33*err)*acc^2, balanced):", flush=True)
    rows = [
        ("mlp s0 (numpy)", accs[0], lat_mlp_np),
        ("mlp ensemble3 np", ens_acc, lat_ens_np),
        ("tree d14", tree_acc, lat_tree),
        ("gbm300", gbm_acc, lat_gbm),
    ]
    for name, acc, lat in rows:
        n_off = 60.0 / lat
        print(f"  {name:18s} acc={acc:.3f} lat={lat*1e6:7.0f}us n~{n_off:>9,.0f} "
              f"-> score ~ {score_projection(acc, lat):.1f}", flush=True)

    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()