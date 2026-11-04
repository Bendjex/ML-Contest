"""13: Echte v6-Gewichte -> ONNX (vollständiger Graph bis Logits) + Verifikation.

v6.5 (10_latency_profile) hat gezeigt: ONNX Runtime schlägt die numpy-im2col-Pipeline
6.7x (50 vs 335 us). Hier wird der Export mit den ECHTEN trainierten Gewichten
(models/cnn2_weights.npz, BN bereits gefaltet) gemacht:

  1. Torch-Modell ohne BN (Fold schon in den Gewichten), vollstaendiger Graph:
     conv1 -> pool -> conv2 -> pool -> conv3(dil=2) -> global mean/max/std -> fc -> logits
  2. Torch vs numpy-forward_v6 Verifikation (gleiche Mathematik, max diff)
  3. ONNX-Export (opset 17, statisch (1,4,256))
  4. ORT vs numpy auf 20k val: Genauigkeit muss identisch sein
  5. Harness-style Latenz (prep + ORT, single-thread) + Score-Projektion
  6. Kopiert onnx + model.py-Vorlage nach submission/

Log: experiments/logs/onnx_real.log
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

MLC_HOME = Path(os.environ.get("MLC_HOME", "/home/olorin/ml-contest-data/materials"))
HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
SUBMISSION = HERE.parent / "submission"
sys.path.insert(0, str(MLC_HOME))

from competition_api import load_dataset  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(1)


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


class CNNExport(nn.Module):
    """v6 ohne BN (Fold ist in den exportierten Gewichten), komplett bis Logits."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(4, 16, 9, padding=4)
        self.c2 = nn.Conv1d(16, 32, 9, padding=4)
        self.c3 = nn.Conv1d(32, 64, 9, padding=8, dilation=2)
        self.fc = nn.Linear(192, 8)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.c1(x)), 4)
        x = F.max_pool1d(F.relu(self.c2(x)), 4)
        x = F.relu(self.c3(x))
        x = torch.cat([x.mean(-1), x.amax(-1), x.std(-1, unbiased=False)], dim=1)
        return self.fc(x)


def main():
    w = {k: v for k, v in np.load(MODELS / "cnn2_weights.npz").items()}
    m = CNNExport()
    with torch.no_grad():
        m.c1.weight.copy_(torch.from_numpy(w["c1_W"]))
        m.c1.bias.copy_(torch.from_numpy(w["c1_b"]))
        m.c2.weight.copy_(torch.from_numpy(w["c2_W"]))
        m.c2.bias.copy_(torch.from_numpy(w["c2_b"]))
        m.c3.weight.copy_(torch.from_numpy(w["c3_W"]))
        m.c3.bias.copy_(torch.from_numpy(w["c3_b"]))
        m.fc.weight.copy_(torch.from_numpy(w["fc_W"]))
        m.fc.bias.copy_(torch.from_numpy(w["fc_b"]))
    m.eval()

    ds = load_dataset(MLC_HOME / "data" / "train_100k.npz")
    y = np.asarray(ds.labels, dtype=np.int64)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    val = order[:20_000]
    yv = y[val]
    X = prep_batch(ds.series)

    # 1) torch vs numpy (5 Samples)
    with torch.no_grad():
        tl = np.stack([m(torch.from_numpy(X[val[i]])[None]).numpy()[0] for i in range(5)])
    nl = np.stack([forward_v6(X[val[i]], w) for i in range(5)])
    print(f"torch vs numpy: |max diff| = {np.abs(tl - nl).max():.2e}", flush=True)

    # 2) ONNX-Export (kompletter Graph, statisch (1,4,256))
    SUBMISSION.mkdir(exist_ok=True, parents=True)
    onnx_path = SUBMISSION / "cnn_v6.onnx"
    xt = torch.from_numpy(X[val[0]])[None]
    with torch.no_grad():
        m(xt)
    torch.onnx.export(m, xt, str(onnx_path), input_names=["series"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    print(f"ONNX exportiert: {onnx_path} ({onnx_path.stat().st_size/1024:.0f} KB)", flush=True)

    # 3) ORT vs numpy auf 20k val
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    ort_pred = np.empty(len(val), dtype=np.int64)
    for i, j in enumerate(val):
        ort_pred[i] = int(np.argmax(sess.run(None, {name: X[j][None]})[0][0]))
    dt = time.perf_counter() - t0
    acc_ort = float((ort_pred == yv).mean())
    print(f"ORT val_acc = {acc_ort:.4f} (numpy war 0.9018) | 20k in {dt:.0f}s unthrottled",
          flush=True)

    np_pred = np.array([int(np.argmax(forward_v6(X[j], w))) for j in val[:2000]])
    ort_pred2k = ort_pred[:2000]
    print(f"Diskrepanz ORT vs numpy auf 2k: {(np_pred != ort_pred2k).sum()} Samples", flush=True)

    # 4) harness-style Latenz: prep + ORT, single thread
    from threadpoolctl import threadpool_limits
    demo = load_dataset(MLC_HOME / "data" / "test_demo.npz")
    samples = [demo[i] for i in range(len(demo))]

    def prep_one(series):
        x = np.array(series, dtype=np.float32)
        t = np.arange(x.shape[1])
        for ch in x:
            mm = np.isnan(ch)
            if mm.all():
                ch[:] = 0.0
            elif mm.any():
                ch[mm] = np.interp(t[mm], t[~mm], ch[~mm])
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + np.float32(1e-9)
        return ((x - mu) / sd).astype(np.float32)

    def full_call(s):
        return int(np.argmax(sess.run(None, {name: prep_one(s.series)[None]})[0][0]))

    with threadpool_limits(limits=1):
        for s in samples[:20]:
            full_call(s)
        t0 = time.perf_counter()
        for s in samples:
            full_call(s)
        lat = (time.perf_counter() - t0) / len(samples)
        t0 = time.perf_counter()
        for s in samples:
            prep_one(s.series)
        lat_prep = (time.perf_counter() - t0) / len(samples)

    print(f"\nLATENCY prep+ORT: {lat*1e6:.0f} us/call (prep {lat_prep*1e6:.0f} us)", flush=True)

    # 5) Score-Projektion mit per-class acc aus ORT (20k val)
    n_total = 60.0 / lat
    n_k = n_total / 8.0
    Ts = []
    print(f"n(60 s) ~ {n_total:,.0f} | n_k ~ {n_k:,.0f}", flush=True)
    for c in range(8):
        a = float((ort_pred[yv == c] == c).mean())
        e = 1.0 - a
        T = max((n_k ** 0.5) * (a - 0.33 * e) * a ** 2, 0.01)
        Ts.append(T)
        print(f"  class {c}: acc={a:.3f} T_k={T:.1f}", flush=True)
    gm = math.exp(sum(map(math.log, Ts)) / 8.0)
    print(f"\nGEOMEAN SCORE (ORT-Pfad, Projektion) ~ {gm:.1f}  (numpy: 82.4, tree: 22)", flush=True)
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()