"""Latency benchmark (harness-style, one call at a time) for tree vs GBM.
Models are already pickled by tree_train.py; only re-runs the latency part.
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
from competition_api import load_dataset  # noqa: E402


def bench(name, path):
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    demo = load_dataset(REPO / "data" / "test_demo.npz")
    n = len(demo)
    samples = [demo[i] for i in range(n)]
    for s in samples:
        int(model.predict(features(s)[np.newaxis, :])[0])  # warm
    t0 = time.perf_counter()
    for s in samples:
        int(model.predict(features(s)[np.newaxis, :])[0])
    lat = (time.perf_counter() - t0) / n
    # split: feature cost alone
    t0 = time.perf_counter()
    for s in samples:
        features(s)
    feat_lat = (time.perf_counter() - t0) / n
    print(f"{name}: total={lat*1e6:.0f} us/call | features={feat_lat*1e6:.0f} us | "
          f"model={1e6*(lat-feat_lat):.0f} us | -> n_official(60s) ~ {60/lat:,.0f}",
          flush=True)


print("=== LATENCY BENCHMARK (harness-style) ===", flush=True)
bench("tree depth14", REPO / "examples" / "tree_model.pkl")
bench("gbm(300)", REPO / "examples" / "gbm_model.pkl")
print("=== DONE ===", flush=True)