"""05: 1D-CNN on the raw window (torch CPU) -> exported to numpy for submission.

Question: can a small conv net learn the temporal shape that handcrafted features
miss? Target: <=500 us/call, >=0.70 val acc (tree is 0.638 @ 681 us).

Architecture (deliberately numpy-exportable, no fancy ops):
  conv1: 4->16  k=9 pad=4, BN, ReLU, maxpool 4      -> 16x64
  conv2: 16->32 k=9 pad=4, BN, ReLU, maxpool 4      -> 32x16
  conv3: 32->64 k=5 pad=2, BN, ReLU                 -> 64x16
  global avg-pool + global max-pool (concat)        -> 128
  fc 128->8

Training: Adam lr=1e-3, batch 256, class weights (eval stream is balanced),
          NaN-like dropout augmentation (p=0.014 -> value 0 = channel mean),
          Gaussian noise sigma=0.05, early stop on val acc.

Export: BN folded into conv weights; npz with conv W/b, fc W/b.
Latency: numpy im2col forward, single-thread, harness-style one call per sample.
Plus: confusion matrix on val (degenerate-pair check), score projection.

Log: experiments/logs/cnn.log
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
sys.path.insert(0, str(MLC_HOME))
sys.path.insert(0, str(HERE))

from competition_api import load_dataset  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

EPOCHS = 30
BATCH = 256
LR = 1e-3
SEED = 0


def prep_batch(series: np.ndarray) -> np.ndarray:
    """(N,C,L) float32 -> (N,C,L) float32: NaN ffill/bfill (all-NaN->0), per-channel z."""
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


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(4, 16, 9, padding=4)
        self.b1 = nn.BatchNorm1d(16)
        self.c2 = nn.Conv1d(16, 32, 9, padding=4)
        self.b2 = nn.BatchNorm1d(32)
        self.c3 = nn.Conv1d(32, 64, 5, padding=2)
        self.b3 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(128, 8)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.b1(self.c1(x))), 4)
        x = F.max_pool1d(F.relu(self.b2(self.c2(x))), 4)
        x = F.relu(self.b3(self.c3(x)))
        x = torch.cat([x.mean(-1), x.amax(-1)], dim=1)
        return self.fc(x)


# ---------------------------------------------------------------- numpy export
def conv1d_np(x, W, b, pad):
    xp = np.pad(x, ((0, 0), (pad, pad)))
    sw = np.lib.stride_tricks.sliding_window_view(xp, W.shape[2], axis=1)
    cols = sw.transpose(0, 2, 1).reshape(-1, sw.shape[1])
    return W.reshape(W.shape[0], -1) @ cols + b[:, None]


def maxpool_np(x, p):
    C, L = x.shape
    return x[:, :(L // p) * p].reshape(C, L // p, p).max(-1)


def forward_np(x, w):
    """x: (4,256) float32 standardized -> logits (8,)"""
    x = np.maximum(conv1d_np(x, w["c1_W"], w["c1_b"], 4), 0)
    x = maxpool_np(x, 4)
    x = np.maximum(conv1d_np(x, w["c2_W"], w["c2_b"], 4), 0)
    x = maxpool_np(x, 4)
    x = np.maximum(conv1d_np(x, w["c3_W"], w["c3_b"], 2), 0)
    pooled = np.concatenate([x.mean(axis=1), x.max(axis=1)])
    return w["fc_W"] @ pooled + w["fc_b"]


def export(model: CNN) -> dict:
    w = {}
    for name, conv, bn in (("c1", model.c1, model.b1), ("c2", model.c2, model.b2), ("c3", model.c3, model.b3)):
        scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy().astype(np.float32)
        shift = (bn.bias - bn.running_mean * torch.sqrt(bn.running_var + bn.eps) / bn.weight * 0).detach()
        shift = (bn.bias - bn.running_mean * (bn.weight / torch.sqrt(bn.running_var + bn.eps))).detach().numpy().astype(np.float32)
        W = conv.weight.detach().numpy().astype(np.float32) * scale[:, None, None]
        b = conv.bias.detach().numpy().astype(np.float32) * scale + shift
        w[f"{name}_W"], w[f"{name}_b"] = W, b
    w["fc_W"] = model.fc.weight.detach().numpy().astype(np.float32)
    w["fc_b"] = model.fc.bias.detach().numpy().astype(np.float32)
    return w


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()}", flush=True)

    ds = load_dataset(MLC_HOME / "data" / "train_100k.npz")
    y = np.asarray(ds.labels, dtype=np.int64)
    t0 = time.perf_counter()
    X = prep_batch(ds.series)
    print(f"prep: {X.shape} in {time.perf_counter()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    n_val = 20_000
    val, tr = order[:n_val], order[n_val:]

    Xtr = torch.from_numpy(X[tr])
    ytr = torch.from_numpy(y[tr])
    Xva = torch.from_numpy(X[val])
    yva = torch.from_numpy(y[val])

    counts = np.bincount(y[tr], minlength=8)
    class_w = torch.tensor(len(y[tr]) / (8 * counts), dtype=torch.float32)
    print("class weights:", class_w.tolist(), flush=True)

    model = CNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(weight=class_w)
    n_train = len(tr)
    best_acc, best_state = -1.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        t0 = time.perf_counter()
        tot = 0.0
        for i in range(0, n_train, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], ytr[idx]
            # augmentation: NaN-like dropout + noise
            xb = xb * (torch.rand_like(xb) > 0.014).float()
            xb = xb + 0.05 * torch.randn_like(xb)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            pred = model(Xva).argmax(1).numpy()
        acc = float((pred == y[val]).mean())
        dt = time.perf_counter() - t0
        print(f"epoch {epoch:2d}: loss={tot/n_train:.4f} val_acc={acc:.4f} ({dt:.0f}s)", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xva).argmax(1).numpy()
    yv = y[val]
    print(f"\nBEST val_acc={best_acc:.4f}", flush=True)
    cm = np.zeros((8, 8), dtype=np.int64)
    np.add.at(cm, (yv, pred), 1)
    print("confusion matrix (rows=true, cols=pred):", flush=True)
    for c in range(8):
        print("  " + " ".join(f"{v:5d}" for v in cm[c]), f"   | class {c}", flush=True)
    per_class = [float((pred[yv == c] == c).mean()) for c in range(8)]
    print("per-class acc:", ["%.2f" % a for a in per_class], flush=True)

    w = export(model)
    np.savez(MODELS / "cnn1_weights.npz", **w)

    # sanity: numpy forward vs torch
    with torch.no_grad():
        t_logits = model(Xva[:5]).numpy()
    n_logits = np.stack([forward_np(X[val[i]], w) for i in range(5)])
    print(f"export check |max diff| = {np.abs(t_logits - n_logits).max():.2e}", flush=True)

    # numpy forward accuracy (should match)
    pred_np = np.array([int(np.argmax(forward_np(X[val[i]], w))) for i in range(len(val))])
    print(f"numpy-forward val_acc={float((pred_np == yv).mean()):.4f}", flush=True)

    # harness-style latency, single thread
    from threadpoolctl import threadpool_limits
    demo = load_dataset(MLC_HOME / "data" / "test_demo.npz")
    samples = [demo[i] for i in range(len(demo))]

    def prep_one(series):
        x = np.array(series, dtype=np.float32)
        t = np.arange(x.shape[1])
        for ch in x:
            m = np.isnan(ch)
            if m.all():
                ch[:] = 0.0
            elif m.any():
                ch[m] = np.interp(t[m], t[~m], ch[~m])
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + np.float32(1e-9)
        return ((x - mu) / sd).astype(np.float32)

    def lat_of(fn):
        for s in samples[:20]:
            fn(s)
        t0 = time.perf_counter()
        for s in samples:
            fn(s)
        return (time.perf_counter() - t0) / len(samples)

    with threadpool_limits(limits=1):
        lat_full = lat_of(lambda s: int(np.argmax(forward_np(prep_one(s.series), w))))
        lat_prep = lat_of(lambda s: prep_one(s.series))

    print(f"\nLATENCY numpy path: {lat_full*1e6:.0f} us/call (prep {lat_prep*1e6:.0f} us)", flush=True)
    n_off = 60.0 / lat_full
    acc = best_acc
    err = 1 - acc
    score = (n_off / 8) ** 0.5 * (acc - 0.33 * err) * acc ** 2
    print(f"SCORE PROJECTION: acc={acc:.3f} lat={lat_full*1e6:.0f}us n~{n_off:,.0f} -> score ~ {score:.1f}", flush=True)
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()