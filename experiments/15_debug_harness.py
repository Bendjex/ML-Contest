"""15: DEBUG — warum liefert model.py durch den offiziellen Harness nur ~30-40% acc?

Verdächtige: (a) model.py _prep (np.interp) unterscheidet sich von prep_batch
(ffill/bfill) — aber nur bei Rand-NaNs, sollte ~keinen Einfluss haben;
(b) ONNX/ORT-Verhalten im Worker; (c) Sample-Übergabe im Harness.

Test: submission Model end-to-end auf test_demo (400) + auf 2k train-val-Samples,
gegen numpy forward_v6 mit prep_batch. Plus prep-Differenzvergleich.
Log: experiments/logs/debug_harness.log
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

MLC_HOME = Path(os.environ.get("MLC_HOME", "/home/olorin/ml-contest-data/materials"))
HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
SUBMISSION = HERE.parent / "submission"
sys.path.insert(0, str(MLC_HOME))

from competition_api import load_dataset  # noqa: E402


def prep_batch(series):
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


def conv1d_np(x, W, b, pad, dil=1):
    C_out, C_in, K = W.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    L_out = xp.shape[1] - dil * (K - 1)
    t = np.arange(L_out)[:, None]
    j = dil * np.arange(K)[None, :]
    cols = xp[:, t + j]
    cols = cols.transpose(1, 0, 2).reshape(L_out, C_in * K)
    return W.reshape(C_out, C_in * K) @ cols.T + b[:, None]


def maxpool_np(x, p):
    C, L = x.shape
    return x[:, :(L // p) * p].reshape(C, L // p, p).max(-1)


def forward_v6(x, w):
    x = np.maximum(conv1d_np(x, w["c1_W"], w["c1_b"], 4), 0)
    x = maxpool_np(x, 4)
    x = np.maximum(conv1d_np(x, w["c2_W"], w["c2_b"], 4), 0)
    x = maxpool_np(x, 4)
    x = np.maximum(conv1d_np(x, w["c3_W"], w["c3_b"], 8, 2), 0)
    pooled = np.concatenate([x.mean(1), x.max(1), x.std(1)])
    return w["fc_W"] @ pooled + w["fc_b"]


def main():
    # ---- submission model importieren (exakt wie der Harness: aus der Datei) ----
    spec = importlib.util.spec_from_file_location("submodel", SUBMISSION / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.Model()
    print("[ok] submission Model geladen:", mod.ONNX_FILE, flush=True)

    w = {k: v for k, v in np.load(MODELS / "cnn2_weights.npz").items()}

    # ---- Test A: test_demo (400), so wie der Harness es sieht ----
    demo = load_dataset(MLC_HOME / "data" / "test_demo.npz")
    yd = np.asarray(demo.labels)
    preds = np.array([model.predict(demo[j]) for j in range(len(demo))])
    acc_model = float((preds == yd).mean())
    print(f"\n[A] model.py auf test_demo: acc={acc_model:.3f}", flush=True)
    print(f"    pred-Verteilung: {np.bincount(preds, minlength=8).tolist()}", flush=True)
    print(f"    true-Verteilung: {np.bincount(yd, minlength=8).tolist()}", flush=True)

    # numpy referenz auf denselben Samples (mit prep_batch)
    Xd = prep_batch(demo.series)
    ref = np.array([int(np.argmax(forward_v6(Xd[j], w))) for j in range(len(demo))])
    acc_ref = float((ref == yd).mean())
    print(f"[B] numpy+prep_batch auf test_demo: acc={acc_ref:.3f}", flush=True)

    # ---- Test C: prep-Differenz _prep vs prep_batch ----
    diffs = []
    for j in range(len(demo)):
        a = model._prep(demo[j].series)
        b = prep_batch(demo.series[j:j + 1])[0]
        diffs.append(float(np.abs(a - b).max()))
    print(f"[C] prep max|diff|: mean={np.mean(diffs):.3f} max={np.max(diffs):.3f}", flush=True)

    # ---- Test D: model.py predictions vs numpy auf demo — wo weichen sie ab? ----
    dis = int((preds != ref).sum())
    print(f"[D] Abweichungen model.py vs numpy: {dis}/{len(demo)}", flush=True)

    # ---- Test E: model.py auf 2000 train-val Samples ----
    ds = load_dataset(MLC_HOME / "data" / "train_100k.npz")
    y = np.asarray(ds.labels)
    val = np.random.default_rng(0).permutation(len(y))[:2000]
    pv = np.array([model.predict(ds[int(j)]) for j in val])
    accv = float((pv == y[val]).mean())
    print(f"\n[E] model.py auf 2k train-val: acc={accv:.3f} (soll 0.90)", flush=True)
    print(f"    pred-Verteilung: {np.bincount(pv, minlength=8).tolist()}", flush=True)

    # ---- Test F: ORT-Sessions im selben Prozess, prep_batch-Input ----
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(mod.ONNX_FILE), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    Xv = prep_batch(ds.series[val])
    ortp = np.array([int(np.argmax(sess.run(None, {name: Xv[i][None]})[0][0])) for i in range(len(val))])
    print(f"[F] ORT+prep_batch auf 2k val: acc={float((ortp == y[val]).mean()):.3f}", flush=True)

    # ---- Test G: elementweise _prep vs ORT-Input direkt vergleichen (1 Sample durch den Graph) ----
    j = int(val[0])
    x1 = model._prep(ds[j].series)
    lg1 = sess.run(None, {name: x1[None]})[0][0]
    lg2 = forward_v6(prep_batch(ds.series[j:j + 1])[0], w)
    print(f"[G] logits |diff| _prep-vs-prep_batch Graph: {np.abs(lg1 - lg2).max():.3e}", flush=True)
    print("=== DEBUG DONE ===", flush=True)


if __name__ == "__main__":
    main()