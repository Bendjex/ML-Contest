"""08: v6 CNN — targeted fixes for the 6<->7 and 0<->3 confusions (training + export).

Changes vs 05 (v5: val 0.890 @ 465 us, score ~81):
- conv3: k=5 pad=2  ->  k=9 pad=8 DILATION=2
    receptive field: 1+(9-1)*2 = 17 steps on the 32x16 stage = 17*16 = 272 input
    steps ~ the whole 256-window (v5 saw ~80). Targets: 6/7 (slow inertia dynamics).
- global pooling: [mean, max] (128) -> [mean, max, STD] (192); fc 192->8.
    std per channel captures roughness variance. Targets: 0/3.
- circular shift augmentation per sample (phase invariance).
- cosine LR schedule, 40 epochs (v5 was still improving at 30).
- error analysis: 6/7 & 0/3 confusions split by sample_rate, NaN stats of errors.
- v5 kept for comparison on the same val split (cnn1_weights.npz).
Export: models/cnn2_weights.npz. Log: experiments/logs/cnn_v6.log (via run_08.sh).
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

MLC_HOME = Path(os.environ.get("MLC_HOME", "/home/olorin/ml-contest-data/materials"))
HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
sys.path.insert(0, str(MLC_HOME))

from competition_api import load_dataset  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

EPOCHS = 40
BATCH = 256
LR = 1e-3
SEED = 0


def prep_batch(series: np.ndarray) -> np.ndarray:
    """(N,C,L) float32 -> NaN ffill/bfill (all-NaN->0) + per-channel z-score."""
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
        self.c3 = nn.Conv1d(32, 64, 9, padding=8, dilation=2)
        self.b3 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(192, 8)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.b1(self.c1(x))), 4)
        x = F.max_pool1d(F.relu(self.b2(self.c2(x))), 4)
        x = F.relu(self.b3(self.c3(x)))
        x = torch.cat([x.mean(-1), x.amax(-1), x.std(-1, unbiased=False)], dim=1)
        return self.fc(x)


# ---------------------------------------------------------------- numpy paths
def conv1d_np(x, W, b, pad, dil=1):
    """x (C_in,L), W (C_out,C_in,K) -> (C_out,L_out). Dilation via index gather."""
    C_out, C_in, K = W.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    L_out = xp.shape[1] - dil * (K - 1)
    t = np.arange(L_out)[:, None]
    j = dil * np.arange(K)[None, :]
    cols = xp[:, t + j]                              # (C_in, L_out, K)
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


def forward_v5(x, w):  # from 06, for comparison
    def conv(x, W, b, pad):
        xp = np.pad(x, ((0, 0), (pad, pad)))
        sw = np.lib.stride_tricks.sliding_window_view(xp, W.shape[2], axis=1)
        cols = sw.transpose(0, 2, 1).reshape(-1, sw.shape[1])
        return W.reshape(W.shape[0], -1) @ cols + b[:, None]
    x = np.maximum(conv(x, w["c1_W"], w["c1_b"], 4), 0)
    x = maxpool_np(x, 4)
    x = np.maximum(conv(x, w["c2_W"], w["c2_b"], 4), 0)
    x = maxpool_np(x, 4)
    x = np.maximum(conv(x, w["c3_W"], w["c3_b"], 2), 0)
    pooled = np.concatenate([x.mean(axis=1), x.max(axis=1)])
    return w["fc_W"] @ pooled + w["fc_b"]


def export(model):
    w = {}
    for name, conv, bn in (("c1", model.c1, model.b1), ("c2", model.c2, model.b2),
                           ("c3", model.c3, model.b3)):
        scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy().astype(np.float32)
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
    sr_all = np.asarray(ds.meta["sample_rate"], dtype=float)
    nanfrac = np.isnan(ds.series).mean(axis=(1, 2))
    t0 = time.perf_counter()
    X = prep_batch(ds.series)
    print(f"prep: {X.shape} in {time.perf_counter()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    n_val = 20_000
    val, tr = order[:n_val], order[n_val:]
    yv = y[val]
    sr_val = sr_all[val]
    nf_val = nanfrac[val]

    Xtr = torch.from_numpy(X[tr])
    ytr = torch.from_numpy(y[tr])
    Xva = torch.from_numpy(X[val])

    counts = np.bincount(y[tr], minlength=8)
    class_w = torch.tensor(len(y[tr]) / (8 * counts), dtype=torch.float32)
    model = CNN()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params: {n_par:,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss(weight=class_w)

    L = 256
    ar = torch.arange(L)
    best_acc, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(tr))
        t0 = time.perf_counter()
        tot = 0.0
        for i in range(0, len(tr), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], ytr[idx]
            # aug: circular shift (per sample) -> NaN dropout -> noise
            sh = torch.randint(0, L, (xb.shape[0],))
            gather_idx = ((ar[None, :] + sh[:, None]) % L).unsqueeze(1).expand(-1, 4, -1)
            xb = torch.gather(xb, 2, gather_idx)
            xb = xb * (torch.rand_like(xb) > 0.014).float()
            xb = xb + 0.05 * torch.randn_like(xb)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(Xva).argmax(1).numpy() == yv).mean())
        print(f"epoch {epoch:2d}: loss={tot/len(tr):.4f} val_acc={acc:.4f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_t = model(Xva).argmax(1).numpy()
    print(f"\nBEST torch val_acc={best_acc:.4f}", flush=True)

    w = export(model)
    np.savez(MODELS / "cnn2_weights.npz", **w)
    with torch.no_grad():
        t_logits = model(Xva[:5]).numpy()
    n_logits = np.stack([forward_v6(X[val[i]], w) for i in range(5)])
    print(f"export check v6 |max diff| = {np.abs(t_logits - n_logits).max():.2e}", flush=True)

    # numpy-path val accuracy
    t0 = time.perf_counter()
    pred = np.array([int(np.argmax(forward_v6(X[val[i]], w))) for i in range(len(val))])
    print(f"numpy-path val_acc={float((pred == yv).mean()):.4f} "
          f"({time.perf_counter()-t0:.0f}s for 20k)", flush=True)

    # v5 comparison on the same split
    w5 = {k: v for k, v in np.load(MODELS / "cnn1_weights.npz").items()}
    pred5 = np.array([int(np.argmax(forward_v5(X[val[i]], w5))) for i in range(len(val))])
    print(f"v5 comparison val_acc={float((pred5 == yv).mean()):.4f}", flush=True)

    # ---------------- confusion + error analysis ----------------
    for tag, p in (("v6", pred), ("v5", pred5)):
        cm = np.zeros((8, 8), dtype=np.int64)
        np.add.at(cm, (yv, p), 1)
        pc = [float((p[yv == c] == c).mean()) for c in range(8)]
        print(f"\n{tag} per-class acc: {['%.3f' % a for a in pc]}", flush=True)
        print(f"{tag} weak pairs (>200): "
              f"{[(int(cm[i, j]), i, j) for i in range(8) for j in range(8) if cm[i, j] > 200 and i != j]}",
              flush=True)

    print("\nERROR ANALYSIS v6 (6/7 + 0/3):", flush=True)
    for c, other in ((6, 7), (7, 6), (0, 3), (3, 0)):
        m_err = (yv == c) & (pred == other)
        m_ok = (yv == c) & (pred == c)
        if m_err.sum():
            print(f"  true {c} -> pred {other}: n={int(m_err.sum())} | "
                  f"sr=2 share err {float(np.mean(sr_val[m_err]==2)):.2f} vs ok {float(np.mean(sr_val[m_ok]==2)):.2f} | "
                  f"NaN err {float(nf_val[m_err].mean()):.4f} vs ok {float(nf_val[m_ok].mean()):.4f}",
                  flush=True)

    # ---------------- latency + score ----------------
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
        lat = lat_of(lambda s: int(np.argmax(forward_v6(prep_one(s.series), w))))
    acc_np = float((pred == yv).mean())
    n_total = 60.0 / lat
    n_k = n_total / 8.0
    print(f"\nLATENCY v6: {lat*1e6:.0f} us/call | n(60s) ~ {n_total:,.0f} | n_k ~ {n_k:,.0f}",
          flush=True)
    pc6 = [float((pred[yv == c] == c).mean()) for c in range(8)]
    Ts = []
    for c, a in enumerate(pc6):
        e = 1.0 - a
        T = (n_k ** 0.5) * (a - 0.33 * e) * a ** 2
        Ts.append(max(T, 0.01))
        print(f"  class {c}: acc={a:.3f} T_k={T:.1f}", flush=True)
    gm = math.exp(sum(math.log(t) for t in Ts) / 8.0)
    print(f"\nGEOMEAN SCORE v6 ~ {gm:.1f}  (v5: 81.3, tree: 22, GBM: 10)", flush=True)
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()