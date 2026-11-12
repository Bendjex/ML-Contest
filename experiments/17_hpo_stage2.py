"""17: HPO stage 2 — combine stage-1 winners + training-side levers.

Stage 1 result (16_hpo_stage1.log): avg_pool wins (74.8 vs 66.1 baseline);
k=15, wide, gelu, leaky, gp-variants all lose. Class 6 collapses on demo (0.36).

Stage 2 probes (all avg_pool unless noted):
  avg40          — 40 epochs (longer cosine schedule)
  avg_aug        — + demo-matched augmentation: noise sigma 0.05->0.10, NaN p 0.014->0.02
  avg_aug40      — both of the above
  avg_wide24     — channels 24-48-96 (stage-1 wide had best val_acc 0.904)
  avg_aug40_c6w2 — + class-6 weight x2 (class 6 is THE bottleneck on demo)

Objective identical to stage 1: official score formula on demo per-class acc,
n from calibrated ORT latency estimate. Log: experiments/logs/hpo_stage2.log
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

BATCH = 256
LR = 1e-3


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


class CNN(nn.Module):
    def __init__(self, ch=(16, 32, 64), k=(9, 9, 9), dil3=2, pool="avg", gp="mms"):
        super().__init__()
        self.ch, self.k, self.dil3, self.pool, self.gp = ch, k, dil3, pool, gp
        self.c1 = nn.Conv1d(4, ch[0], k[0], padding=k[0] // 2)
        self.b1 = nn.BatchNorm1d(ch[0])
        self.c2 = nn.Conv1d(ch[0], ch[1], k[1], padding=k[1] // 2)
        self.b2 = nn.BatchNorm1d(ch[1])
        self.c3 = nn.Conv1d(ch[1], ch[2], k[2], padding=(k[2] // 2) * dil3, dilation=dil3)
        self.b3 = nn.BatchNorm1d(ch[2])
        self.fc = nn.Linear(3 * ch[2], 8)

    def forward(self, x):
        pool = F.max_pool1d if self.pool == "max" else F.avg_pool1d
        x = pool(F.relu(self.b1(self.c1(x))), 4)
        x = pool(F.relu(self.b2(self.c2(x))), 4)
        x = F.relu(self.b3(self.c3(x)))
        return self.fc(torch.cat([x.mean(-1), x.amax(-1), x.std(-1, unbiased=False)], 1))


def export_np(model):
    w = {}
    for name, conv, bn in (("c1", model.c1, model.b1), ("c2", model.c2, model.b2),
                           ("c3", model.c3, model.b3)):
        scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy().astype(np.float32)
        shift = (bn.bias - bn.running_mean * (bn.weight / torch.sqrt(bn.running_var + bn.eps))).detach().numpy().astype(np.float32)
        w[f"{name}_W"] = conv.weight.detach().numpy().astype(np.float32) * scale[:, None, None]
        w[f"{name}_b"] = conv.bias.detach().numpy().astype(np.float32) * scale + shift
    w["fc_W"] = model.fc.weight.detach().numpy().astype(np.float32)
    w["fc_b"] = model.fc.bias.detach().numpy().astype(np.float32)
    return w


def forward_np(x, w, ch, k, dil3, pool):
    def conv(x, W, b, pad, dil=1):
        C_out, C_in, K = W.shape
        xp = np.pad(x, ((0, 0), (pad, pad)))
        L_out = xp.shape[1] - dil * (K - 1)
        t = np.arange(L_out)[:, None]
        j = dil * np.arange(K)[None, :]
        cols = xp[:, t + j].transpose(1, 0, 2).reshape(L_out, C_in * K)
        return W.reshape(C_out, C_in * K) @ cols.T + b[:, None]

    def pmax(x, p):
        C, L = x.shape
        return x[:, :(L // p) * p].reshape(C, L // p, p).max(-1)

    def pavg(x, p):
        C, L = x.shape
        return x[:, :(L // p) * p].reshape(C, L // p, p).mean(-1)

    poolf = pmax if pool == "max" else pavg
    x = poolf(np.maximum(conv(x, w["c1_W"], w["c1_b"], k[0] // 2), 0), 4)
    x = poolf(np.maximum(conv(x, w["c2_W"], w["c2_b"], k[1] // 2), 0), 4)
    x = np.maximum(conv(x, w["c3_W"], w["c3_b"], (k[2] // 2) * dil3, dil3), 0)
    p = np.concatenate([x.mean(1), x.max(1), x.std(1)])
    return w["fc_W"] @ p + w["fc_b"]


PROBES = [
    ("avg40",          dict(epochs=40)),
    ("avg_aug",        dict(epochs=20, sigma=0.10, nanp=0.02)),
    ("avg_aug40",      dict(epochs=40, sigma=0.10, nanp=0.02)),
    ("avg_wide24",     dict(epochs=20, ch=(24, 48, 96))),
    ("avg_aug40_c6w2", dict(epochs=40, sigma=0.10, nanp=0.02, c6w=2.0)),
]


def train_probe(cfg, split, seed=0):
    epochs = cfg.get("epochs", 20)
    sigma = cfg.get("sigma", 0.05)
    nanp = cfg.get("nanp", 0.014)
    c6w = cfg.get("c6w", 1.0)
    Xtr_t, ytr_t, Xva_t, yva = split
    torch.manual_seed(seed)
    m = CNN(ch=cfg.get("ch", (16, 32, 64)))
    counts = np.bincount(ytr_t.numpy(), minlength=8)
    cw_np = len(ytr_t) / (8 * counts) * np.where(np.arange(8) == 6, c6w, 1.0)
    cw = torch.tensor(cw_np, dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=cw)
    L = 256
    ar = torch.arange(L)
    best, state = -1.0, None
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(ytr_t))
        for i in range(0, len(ytr_t), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            sh = torch.randint(0, L, (xb.shape[0],))
            gi = ((ar[None, :] + sh[:, None]) % L).unsqueeze(1).expand(-1, 4, -1)
            xb = torch.gather(xb, 2, gi)
            xb = xb * (torch.rand_like(xb) > nanp).float()
            xb = xb + sigma * torch.randn_like(xb)
            opt.zero_grad()
            loss = lossf(m(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            acc = float((m(Xva_t).argmax(1).numpy() == yva).mean())
        if acc > best:
            best = acc
            state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    m.eval()
    return m, best


def main():
    torch.set_num_threads(6)
    ds = load_dataset(MLC_HOME / "data" / "train_100k.npz")
    demo = load_dataset(MLC_HOME / "data" / "test_demo.npz")
    y = np.asarray(ds.labels, dtype=np.int64)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    val, tr = order[:20_000], order[20_000:]
    X = prep_batch(ds.series)
    Xtr_t = torch.from_numpy(X[tr])
    ytr_t = torch.from_numpy(y[tr])
    Xva_t = torch.from_numpy(X[val])
    yva = y[val]
    yd = np.asarray(demo.labels)
    Xd = prep_batch(demo.series)

    from threadpoolctl import threadpool_limits

    print("=" * 78, flush=True)
    print("HPO STAGE 2 — winner combos + training levers (all avg_pool)", flush=True)
    print("objective = official score formula on demo per-class acc, calibrated ORT lat", flush=True)
    print("=" * 78, flush=True)
    print(f"\n{'probe':<16} {'val_acc':>7} {'demo_acc':>8} {'np_us':>6} {'ort_us':>7} "
          f"{'n(60s)':>9} {'score':>7}  per-class demo", flush=True)

    best_name, best_score = None, -1.0
    for name, cfg in PROBES:
        t0 = time.perf_counter()
        m, val_acc = train_probe(cfg, (Xtr_t, ytr_t, Xva_t, yva))
        t_train = time.perf_counter() - t0
        w = export_np(m)
        np.savez(MODELS / f"hpo2_{name}.npz", **w)
        with threadpool_limits(limits=1):
            preds_d = np.empty(len(demo), dtype=np.int64)
            t0 = time.perf_counter()
            for j in range(len(demo)):
                preds_d[j] = int(np.argmax(forward_np(Xd[j], w, m.ch, m.k, m.dil3, m.pool)))
            np_fwd = (time.perf_counter() - t0) / len(demo)
        demo_acc = float((preds_d == yd).mean())
        ort_est = 61e-6 + 0.24 * np_fwd
        n_total = 60.0 / ort_est
        n_k = n_total / 8
        pc = [float((preds_d[yd == c] == c).mean()) for c in range(8)]
        Ts = [max((n_k ** 0.5) * (a - 0.33 * (1 - a)) * a ** 2, 0.01) for a in pc]
        score = math.exp(sum(map(math.log, Ts)) / 8)
        if score > best_score:
            best_name, best_score = name, score
        print(f"{name:<16} {val_acc:>7.3f} {demo_acc:>8.3f} {np_fwd*1e6:>6.0f} {ort_est*1e6:>7.0f} "
              f"{n_total:>9,.0f} {score:>7.1f}  {['%.2f' % a for a in pc]}  ({t_train:.0f}s)",
              flush=True)

    print(f"\nBEST STAGE 2: {best_name} score {best_score:.1f} (stage1 best: avg_pool 74.8)", flush=True)
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()