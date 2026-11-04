"""18: Export stage-2 winner (avg_aug40_c6w2, avg_pool) to ONNX + official harness.

The winner differs from v6: avg_pool (not max_pool) in the two intermediate pools.
Loads models/hpo2_avg_aug40_c6w2.npz (BN already folded), builds the torch twin,
exports submission/cnn_v8.onnx, verifies vs numpy forward, prints latency.

run_18.sh then runs the OFFICIAL harness: test_demo (1 run, honest eval-like)
and train_100k (3 runs, comparison to v6's 78.16).
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


class CNNV8(nn.Module):
    """Stage-2 winner: avg_pool intermediate pooling, BN folded (plain convs)."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(4, 16, 9, padding=4)
        self.c2 = nn.Conv1d(16, 32, 9, padding=4)
        self.c3 = nn.Conv1d(32, 64, 9, padding=8, dilation=2)
        self.fc = nn.Linear(192, 8)

    def forward(self, x):
        x = F.avg_pool1d(F.relu(self.c1(x)), 4)
        x = F.avg_pool1d(F.relu(self.c2(x)), 4)
        x = F.relu(self.c3(x))
        return self.fc(torch.cat([x.mean(-1), x.amax(-1), x.std(-1, unbiased=False)], 1))


def forward_np(x, w):
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

    x = pavg(np.maximum(conv(x, w["c1_W"], w["c1_b"], 4), 0), 4)
    x = pavg(np.maximum(conv(x, w["c2_W"], w["c2_b"], 4), 0), 4)
    x = np.maximum(conv(x, w["c3_W"], w["c3_b"], 8, 2), 0)
    p = np.concatenate([x.mean(1), x.max(1), x.std(1)])
    return w["fc_W"] @ p + w["fc_b"]


def main():
    w = {k: v for k, v in np.load(MODELS / "hpo2_avg_aug40_c6w2.npz").items()}
    m = CNNV8()
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
    val = np.random.default_rng(0).permutation(len(y))[:5000]
    X = prep_batch(ds.series)

    with torch.no_grad():
        tl = np.stack([m(torch.from_numpy(X[j])[None]).numpy()[0] for j in val[:5]])
    nl = np.stack([forward_np(X[j], w) for j in val[:5]])
    print(f"torch vs numpy: |max diff| = {np.abs(tl - nl).max():.2e}", flush=True)

    onnx_path = SUBMISSION / "cnn_v8.onnx"
    xt = torch.from_numpy(X[val[0]])[None]
    torch.onnx.export(m, xt, str(onnx_path), input_names=["series"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    print(f"ONNX exportiert: {onnx_path} ({onnx_path.stat().st_size/1024:.0f} KB)", flush=True)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    ort_pred = np.array([int(np.argmax(sess.run(None, {name: X[j][None]})[0][0])) for j in val])
    np_pred = np.array([int(np.argmax(forward_np(X[j], w))) for j in val])
    print(f"ORT vs numpy auf 5k val: {(ort_pred != np_pred).sum()} Abweichungen", flush=True)
    print(f"ORT val_acc (5k): {float((ort_pred == y[val]).mean()):.4f}", flush=True)

    from threadpoolctl import threadpool_limits
    demo = load_dataset(MLC_HOME / "data" / "test_demo.npz")

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
    with threadpool_limits(limits=1):
        for s in samples[:20]:
            int(np.argmax(sess.run(None, {name: prep_one(s.series)[None]})[0][0]))
        t0 = time.perf_counter()
        for s in samples:
            int(np.argmax(sess.run(None, {name: prep_one(s.series)[None]})[0][0]))
        lat = (time.perf_counter() - t0) / len(samples)
    print(f"prep+ORT latency: {lat*1e6:.0f} us/call", flush=True)
    print("=== EXPORT DONE ===", flush=True)


if __name__ == "__main__":
    main()