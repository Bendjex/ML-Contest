"""Train a single decision tree on the baseline feature vector, with a depth sweep.

    /tmp/mlc-venv/bin/python examples/tree_train.py

Extracts the baseline features once, then:
* depth sweep for a single DecisionTreeClassifier (val accuracy),
* trains the HistGradientBoostingClassifier baseline on the same split,
* measures per-sample latency exactly like the harness will (one call at a time),
* saves tree_model.pkl and gbm_model.pkl next to this file.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

from baseline_model import features  # noqa: E402
from competition_api import Sample, load_dataset  # noqa: E402


def per_call_latency(model, ds, n=500) -> float:
    """Mean seconds per predict(x) call, one sample at a time, like the harness."""
    samples = [ds[i] for i in range(n)]
    for s in samples:  # warm caches once
        model.predict(s)
    t0 = time.perf_counter()
    for s in samples:
        model.predict(s)
    return (time.perf_counter() - t0) / n


def main():
    ds = load_dataset(REPO / "data" / "train_100k.npz")
    y = np.asarray(ds.labels)

    t0 = time.perf_counter()
    X = np.stack([features(s) for s in ds])
    dt = time.perf_counter() - t0
    print(f"features: {X.shape[1]} dims x {len(ds)} samples | "
          f"{dt:.0f}s total | {dt/len(ds)*1e6:.0f} us/sample", flush=True)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    n_val = 20_000
    val, tr = order[:n_val], order[n_val:]
    yv = y[val]

    # ---------------- single decision tree: depth sweep ----------------
    from sklearn.tree import DecisionTreeClassifier
    best_acc, best_depth = -1.0, None
    for depth in (6, 10, 14, 20, 28):
        t0 = time.perf_counter()
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X[tr], y[tr])
        fit_t = time.perf_counter() - t0
        acc = float((tree.predict(X[val]) == yv).mean())
        print(f"tree depth={depth:>2}: val_acc={acc:.3f} leaves={tree.get_n_leaves():>6} "
              f"fit={fit_t:.0f}s", flush=True)
        if acc > best_acc:
            best_acc, best_depth = acc, depth
    print(f"BEST TREE: depth={best_depth} val_acc={best_acc:.3f}", flush=True)
    tree = DecisionTreeClassifier(max_depth=best_depth, random_state=0).fit(X, y)
    with open(Path(__file__).with_name("tree_model.pkl"), "wb") as f:
        pickle.dump(tree, f)

    # ---------------- gradient boosting (the baseline, same features) --------
    from sklearn.ensemble import HistGradientBoostingClassifier
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
    t0 = time.perf_counter()
    gbm.fit(X[tr], y[tr])
    print(f"gbm fit: {time.perf_counter()-t0:.0f}s", flush=True)
    gbm_acc = float((gbm.predict(X[val]) == yv).mean())
    print(f"GBM val_acc={gbm_acc:.3f}", flush=True)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(X, y)
    with open(Path(__file__).with_name("gbm_model.pkl"), "wb") as f:
        pickle.dump(gbm, f)

    # ---------------- latency, one call at a time (harness-style) ------------
    demo = load_dataset(REPO / "data" / "test_demo.npz")

    class TreeModel:
        def __init__(self, m): self.m = m
        def predict(self, x): return int(self.m.predict(features(x)[np.newaxis, :])[0])

    class GbmModel:
        def __init__(self, m): self.m = m
        def predict(self, x): return int(self.m.predict(features(x)[np.newaxis, :])[0])

    lat_tree = per_call_latency(TreeModel(tree), demo)
    lat_gbm = per_call_latency(GbmModel(gbm), demo)
    print(f"LATENCY tree: {lat_tree*1e6:.0f} us/sample", flush=True)
    print(f"LATENCY gbm:  {lat_gbm*1e6:.0f} us/sample", flush=True)

    # ---------------- score projection: T_k ~ sqrt(n)*acc^3 ------------------
    # official eval: 60s budget, stream ~ infinite -> n ~ budget/latency (capped by stream)
    for name, acc, lat in (("tree", best_acc, lat_tree), ("gbm", gbm_acc, lat_gbm)):
        n_official = min(60.0 / lat, 10_000_000)
        print(f"SCORE PROJECTION {name}: val_acc={acc:.3f} latency={lat*1e6:.0f}us "
              f"-> n_official~{n_official:,.0f}", flush=True)

    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()