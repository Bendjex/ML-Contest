"""21: Export the Optuna best trial (trial 20, combined 127.4) to ONNX.

Architecture (from optuna_best.npz weight shapes):
  conv1: 4→24, k=7, pad=3 (BN folded)        -> (24, 256)
  avg_pool 4                                  -> (24, 64)
  conv2: 24→48, k=13, pad=6 (BN folded)      -> (48, 64)
  avg_pool 4                                   -> (48, 16)
  conv3: 48→72, k=9, pad=8, dil=2 (BN folded) -> (72, 16)
  global mean+max+std                          -> (216,)
  + meta [sr, gA, gB, gC, level]              -> (221,)
  fc 221→8                                    -> logits

ONNX has TWO inputs: series (1,4,256) and meta (1,5).
"""
from __future__ import annotations

import os
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


def meta_matrix(ds, idx):
    sr = np.asarray(ds.meta["sample_rate"], dtype=np.float32)[idx]
    grp = np.asarray(ds.meta["group"])[idx]
    lev = np.asarray(ds.meta["level"], dtype=np.float32)[idx]
    M = np.zeros((len(idx), 5), dtype=np.float32)
    M[:, 0] = (sr - 1.5) / 0.5
    for j, g in enumerate("ABC"):
        M[:, 1 + j] = (grp == g)
    M[:, 4] = lev / 4.0
    return M


class CNNFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(4, 24, 7, padding=3)
        self.c2 = nn.Conv1d(24, 48, 13, padding=6)
        self.c3 = nn.Conv1d(48, 72, 9, padding=8, dilation=2)
        self.fc = nn.Linear(221, 8)

    def forward(self, x, meta):
        x = F.avg_pool1d(F.relu(self.c1(x)), 4)
        x = F.avg_pool1d(F.relu(self.c2(x)), 4)
        x = F.relu(self.c3(x))
        feats = torch.cat([x.mean(-1), x.amax(-1), x.std(-1, unbiased=False), meta], 1)
        return self.fc(feats)


def forward_np(x, w, meta_vec):
    def conv(x, W, b, pad, dil=1):
        C_out, C_in, K = W.shape
        xp = np.pad(x, ((0, 0), (pad, pad)))
        L_out = xp.shape[1] - dil * (K - 1)
        t = np.arange(L_out)[:, None]
        j = dil * np.arange(K)[None, :]
        cols = xp[:, t + j].transpose(1, 0, 2).reshape(L_out, C_in * K)
        return W.reshape(C_out, C_in * K) @ cols.T + b[:, None]

    def pavg(x, p):
        C, L = x.shape
        return x[:, :(L // p) * p].reshape(C, L // p, p).mean(-1)

    x = pavg(np.maximum(conv(x, w["c1_W"], w["c1_b"], 3), 0), 4)
    x = pavg(np.maximum(conv(x, w["c2_W"], w["c2_b"], 6), 0), 4)
    x = np.maximum(conv(x, w["c3_W"], w["c3_b"], 8, 2), 0)
    p = np.concatenate([x.mean(1), x.max(1), x.std(1), meta_vec])
    return w["fc_W"] @ p + w["fc_b"]


def main():
    w = {k: v for k, v in np.load(MODELS / "optuna_best.npz").items()}
    m = CNNFinal()
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
    val = order[:5_000]
    X = prep_batch(ds.series)
    Mval = meta_matrix(ds, val)

    # torch vs numpy verification
    with torch.no_grad():
        tl = np.stack([m(torch.from_numpy(X[j])[None], torch.from_numpy(Mval[i])[None]).numpy()[0]
                       for i, j in enumerate(val[:5])])
    nl = np.stack([forward_np(X[j], w, Mval[i]) for i, j in enumerate(val[:5])])
    print(f"torch vs numpy: |max diff| = {np.abs(tl - nl).max():.2e}", flush=True)

    # ONNX export (two inputs: series + meta)
    onnx_path = SUBMISSION / "cnn_final.onnx"
    xt = torch.from_numpy(X[val[0]])[None]
    mt = torch.from_numpy(Mval[0])[None]
    torch.onnx.export(m, (xt, mt), str(onnx_path),
                      input_names=["series", "meta"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    print(f"ONNX: {onnx_path} ({onnx_path.stat().st_size/1024:.0f} KB)", flush=True)

    # ORT verification + latency
    import onnxruntime as ort
    from threadpoolctl import threadpool_limits
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    demo = load_dataset(MLC_HOME / "data" / "test_demo.npz")
    Xd = prep_batch(demo.series)
    Md = meta_matrix(demo, np.arange(len(demo)))
    yd = np.asarray(demo.labels)

    ort_pred = np.array([int(np.argmax(sess.run(None, {"series": Xd[j][None], "meta": Md[j][None]})[0][0]))
                         for j in range(len(demo))])
    demo_acc = float((ort_pred == yd).mean())
    print(f"ORT demo acc: {demo_acc:.3f}", flush=True)

    def prep_one(series):
        x = np.array(series, dtype=np.float32)
        t = np.arange(x.shape[1], dtype=np.int32)
        rows = np.arange(x.shape[0])[:, None]
        idx = np.where(~np.isnan(x), t[None, :], 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        xf = x[rows, idx]
        idxb = np.where(~np.isnan(xf[:, ::-1]), t[None, :], 0)
        np.maximum.accumulate(idxb, axis=1, out=idxb)
        xb = xf[rows, idxb[:, ::-1]]
        xb = np.where(np.isnan(xb), np.float32(0), xb)
        mu = xb.mean(axis=1, keepdims=True)
        sd = xb.std(axis=1, keepdims=True) + np.float32(1e-9)
        return ((xb - mu) / sd).astype(np.float32)

    samples = [demo[i] for i in range(len(demo))]

    def meta_one(m):
        sr = float(m.get("sample_rate", 1.5))
        g = str(m.get("group", ""))
        lev = float(m.get("level", 0.0))
        return np.array([(sr - 1.5) / 0.5, float(g == "A"), float(g == "B"),
                         float(g == "C"), lev / 4.0], dtype=np.float32)

    with threadpool_limits(limits=1):
        for s in samples[:20]:
            sess.run(None, {"series": prep_one(s.series)[None], "meta": meta_one(s.meta)[None]})
        t0 = time.perf_counter()
        for s in samples:
            sess.run(None, {"series": prep_one(s.series)[None], "meta": meta_one(s.meta)[None]})
        lat = (time.perf_counter() - t0) / len(samples)
    print(f"prep+ORT latency: {lat*1e6:.0f} us/call", flush=True)

    # per-class on demo
    pc = [float((ort_pred[yd == c] == c).mean()) for c in range(8)]
    print(f"per-class demo: {['%.2f' % a for a in pc]}", flush=True)

    # 5k val acc
    ort_pred_v = np.array([int(np.argmax(sess.run(None, {"series": X[j][None], "meta": Mval[i][None]})[0][0]))
                           for i, j in enumerate(val)])
    val_acc = float((ort_pred_v == y[val]).mean())
    print(f"ORT 5k val acc: {val_acc:.4f}", flush=True)
    print("=== EXPORT DONE ===", flush=True)


if __name__ == "__main__":
    main()