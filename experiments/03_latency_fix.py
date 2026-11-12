"""v2 postmortem fix + final latency/score table.

v2 (02_mlp_v2_train.py) crashed in the latency benchmark: tree/GBM had been fitted on the
RAW 1024-dim window, then fed the 101-dim baseline feature vector (ValueError). The
baselines' v2 accuracies (21.3% / 71.0%) are therefore meaningless; the v1 numbers on the
101-dim features (tree d14 63.8%, GBM 82.0%) stand.

This script (detached via run_03.sh):
 1. refits tree d14 + GBM 300 on the 101-dim baseline features (same split as v1/v2),
 2. reloads the v2 MLPs (models/mlp300_s{0,1,2}.pkl + models/mlp300_weights.npz),
 3. re-measures every latency path harness-style: one predict call per sample, demo set,
    single-threaded (threadpool_limits), warmup calls excluded,
 4. prints the final score projection (T_k = sqrt(n_k)*(acc-0.33*err)*acc^2, balanced).

Log: experiments/logs/latency_final.log
"""
from __future__ import annotations

import importlib
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
sys.path.insert(0, str(HERE))

from competition_api import load_dataset  # noqa: E402
from baseline_model import features as base_features  # noqa: E402

m2 = importlib.import_module("02_mlp_v2_train")  # prep_batch/prep_one/forward/score_projection


def main():
    ds = load_dataset(MLC_HOME / "data" / "train_100k.npz")
    y = np.asarray(ds.labels)
    t0 = time.perf_counter()
    X = m2.prep_batch(ds.series)
    print(f"prep_batch(raw window): {X.shape} in {time.perf_counter()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    n_val = 20_000
    val, tr = order[:n_val], order[n_val:]
    yv = y[val]

    t0 = time.perf_counter()
    F = np.stack([base_features(s) for s in ds])
    print(f"baseline features: {F.shape} in {time.perf_counter()-t0:.0f}s", flush=True)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    tree = DecisionTreeClassifier(max_depth=14, random_state=0).fit(F[tr], y[tr])
    tree_acc = float((tree.predict(F[val]) == yv).mean())
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=0).fit(F[tr], y[tr])
    gbm_acc = float((gbm.predict(F[val]) == yv).mean())
    print(f"baselines (101 feats): tree_d14 {tree_acc:.3f} | gbm300 {gbm_acc:.3f}", flush=True)
    with open(MODELS / "tree_d14_feats.pkl", "wb") as fh:
        pickle.dump(tree, fh)
    with open(MODELS / "gbm300_feats.pkl", "wb") as fh:
        pickle.dump(gbm, fh)

    nets = []
    for s in (0, 1, 2):
        with open(MODELS / f"mlp300_s{s}.pkl", "rb") as fh:
            nets.append(pickle.load(fh))
    mlp_accs = [float((n.predict(X[val]) == yv).mean()) for n in nets]
    proba = np.mean([n.predict_proba(X[val]) for n in nets], axis=0)
    ens_acc = float((np.argmax(proba, axis=1) == yv).mean())
    print(f"mlp val_accs: {[f'{a:.3f}' for a in mlp_accs]} | ensemble(3) {ens_acc:.3f}", flush=True)

    ws = dict(np.load(MODELS / "mlp300_weights.npz"))
    SEEDS = (0, 1, 2)
    coefs = {s: [ws[f"c{s}_0"], ws[f"c{s}_1"], ws[f"c{s}_2"]] for s in SEEDS}
    inters = {s: [ws[f"b{s}_0"], ws[f"b{s}_1"], ws[f"b{s}_2"]] for s in SEEDS}

    demo = load_dataset(MLC_HOME / "data" / "test_demo.npz")
    n = len(demo)
    samples = [demo[i] for i in range(n)]

    def lat_of(fn):
        for s in samples[:20]:
            fn(s)
        t0 = time.perf_counter()
        for s in samples:
            fn(s)
        return (time.perf_counter() - t0) / n

    with threadpool_limits(limits=1):
        lat_prep = lat_of(lambda s: m2.prep_one(s.series))

        def mlp_fwd(s):
            return int(np.argmax(m2.forward(m2.prep_one(s.series), coefs[0], inters[0])))

        lat_mlp_np = lat_of(mlp_fwd)

        def ens_fwd(s):
            v = m2.prep_one(s.series)
            out = m2.forward(v, coefs[0], inters[0])
            for sd in (1, 2):
                out += m2.forward(v, coefs[sd], inters[sd])
            return int(np.argmax(out))

        lat_ens_np = lat_of(ens_fwd)
        lat_sk = lat_of(lambda s: int(nets[0].predict(m2.prep_one(s.series)[None, :])[0]))
        lat_tree = lat_of(lambda s: int(tree.predict(base_features(s)[np.newaxis, :])[0]))
        lat_gbm = lat_of(lambda s: int(gbm.predict(base_features(s)[np.newaxis, :])[0]))

    print("\nLATENCY (single-thread, one call at a time, demo n=400):", flush=True)
    for name, lat in (("prep_one", lat_prep),
                      ("mlp numpy fwd", lat_mlp_np),
                      ("ensemble(3) numpy", lat_ens_np),
                      ("mlp sklearn predict", lat_sk),
                      ("tree d14 + feats", lat_tree),
                      ("gbm300 + feats", lat_gbm)):
        print(f"  {name:20s} {lat*1e6:8.0f} us   n(60s) ~ {60/lat:>10,.0f}", flush=True)

    print("\nSCORE PROJECTION (T_k = sqrt(n_k)*(acc-0.33*err)*acc^2, balanced, geomean):", flush=True)
    rows = [("mlp s0 numpy", mlp_accs[0], lat_mlp_np),
            ("ensemble(3) numpy", ens_acc, lat_ens_np),
            ("tree d14 feats", tree_acc, lat_tree),
            ("gbm300 feats", gbm_acc, lat_gbm)]
    for name, acc, lat in rows:
        print(f"  {name:20s} acc={acc:.3f} lat={lat*1e6:7.0f}us "
              f"-> score ~ {m2.score_projection(acc, lat):8.1f}", flush=True)

    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()