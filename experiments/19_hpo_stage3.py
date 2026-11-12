"""19: HPO stage 3 — the untested levers (one-lever probes on the stage-2 winner).

Base = stage-2 winner avg_aug40_c6w2 (ch 16/32/64, k 9/9/9, dil3=2, avg_pool,
gp mean+max+std, 40 epochs, sigma 0.10, NaN p 0.02, class-6 weight x2).

Probes:
  meta      — concat [sr, group one-hot(3), level] to pooled features (FC 192->197).
              THE mechanistic lever: v6 error analysis showed the 6/7 confusion is
              rate-conditional (sr=2 overrepresented in BOTH directions). The net
              cannot infer sample_rate reliably from the window alone -> feed it.
  ep60      — 60 epochs (cosine T_max=60).
  c67w2     — class 6 AND 7 both x2 (stage 2 pulled 6: 0.36->0.54 but pushed 7: 0.68->0.54).
  dil3_4    — conv3 dilation 2 -> 4 (whole-window sparse RF).
  augheavy  — sigma 0.15, NaN p 0.03 (push demo-matched augmentation further).
  k1_7      — first kernel 9 -> 7 (latency lever: cheaper conv1).

Objective: official score formula on DEMO per-class acc (eval-like preset) with
calibrated ORT latency, PLUS train-val per-class score (stable, n~2500/class)
for noise control — demo n=50/class is noisy, rankings must agree.

Log: experiments/logs/hpo_stage3.log (via run_19.sh)
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


def meta_matrix(ds, idx):
    """[sr_scaled, group A/B/C one-hot, level_scaled] -> (n, 5) float32."""
    sr = np.asarray(ds.meta["sample_rate"], dtype=np.float32)[idx]
    grp = np.asarray(ds.meta["group"])[idx]
    lev = np.asarray(ds.meta["level"], dtype=np.float32)[idx]
    M = np.zeros((len(idx), 5), np.float32)
    M[:, 0] = (sr - 1.5) / 0.5
    for j, g in enumerate("ABC"):
        M[:, 1 + j] = (grp == g)
    M[:, 4] = lev / 4.0
    return M


class CNN(nn.Module):
    def __init__(self, ch=(16, 32, 64), k=(9, 9, 9), dil3=2, meta=False):
        super().__init__()
        self.ch, self.k, self.dil3, self.meta = ch, k, dil3, meta
        self.c1 = nn.Conv1d(4, ch[0], k[0], padding=k[0] // 2)
        self.b1 = nn.BatchNorm1d(ch[0])
        self.c2 = nn.Conv1d(ch[0], ch[1], k[1], padding=k[1] // 2)
        self.b2 = nn.BatchNorm1d(ch[1])
        self.c3 = nn.Conv1d(ch[1], ch[2], k[2], padding=(k[2] // 2) * dil3, dilation=dil3)
        self.b3 = nn.BatchNorm1d(ch[2])
        self.fc = nn.Linear(3 * ch[2] + (5 if meta else 0), 8)

    def forward(self, x, m=None):
        x = F.avg_pool1d(F.relu(self.b1(self.c1(x))), 4)
        x = F.avg_pool1d(F.relu(self.b2(self.c2(x))), 4)
        x = F.relu(self.b3(self.c3(x)))
        feats = torch.cat([x.mean(-1), x.amax(-1), x.std(-1, unbiased=False)], 1)
        if self.meta:
            feats = torch.cat([feats, m], 1)
        return self.fc(feats)


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


def forward_np(x, w, k, dil3, meta_vec=None):
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

    x = pavg(np.maximum(conv(x, w["c1_W"], w["c1_b"], k[0] // 2), 0), 4)
    x = pavg(np.maximum(conv(x, w["c2_W"], w["c2_b"], k[1] // 2), 0), 4)
    x = np.maximum(conv(x, w["c3_W"], w["c3_b"], (k[2] // 2) * dil3, dil3), 0)
    p = np.concatenate([x.mean(1), x.max(1), x.std(1)])
    if meta_vec is not None:
        p = np.concatenate([p, meta_vec])
    return w["fc_W"] @ p + w["fc_b"]


PROBES = [
    ("meta",     dict(meta=True)),
    ("ep60",     dict(epochs=60)),
    ("c67w2",    dict(c6w=2.0, c7w=2.0)),
    ("dil3_4",   dict(dil3=4)),
    ("augheavy", dict(sigma=0.15, nanp=0.03)),
    ("k1_7",     dict(k=(7, 9, 9))),
]


def train_probe(cfg, split, seed=0):
    epochs = cfg.get("epochs", 40)
    sigma = cfg.get("sigma", 0.10)
    nanp = cfg.get("nanp", 0.02)
    cw_np = np.ones(8)
    cw_np[6] = cfg.get("c6w", 2.0)
    cw_np[7] = cfg.get("c7w", 1.0)
    Xtr_t, ytr_t, Xva_t, yva, Mtr_t, Mva_t = split
    torch.manual_seed(seed)
    m = CNN(k=cfg.get("k", (9, 9, 9)), dil3=cfg.get("dil3", 2), meta=cfg.get("meta", False))
    counts = np.bincount(ytr_t.numpy(), minlength=8)
    cw = torch.tensor(len(ytr_t) / (8 * counts) * cw_np, dtype=torch.float32)
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
            loss = lossf(m(xb, Mtr_t[idx]), yb)
            loss.backward()
            opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            acc = float((m(Xva_t, Mva_t).argmax(1).numpy() == yva).mean())
        if acc > best:
            best = acc
            state = {kk: v.detach().clone() for kk, v in m.state_dict().items()}
    m.load_state_dict(state)
    m.eval()
    return m, best


def score_from(pc, lat_s):
    """pc: per-class accs -> geomean of official T_k with n from latency."""
    n_k = (60.0 / lat_s) / 8
    Ts = [max((n_k ** 0.5) * (a - 0.33 * (1 - a)) * a ** 2, 0.01) for a in pc]
    return math.exp(sum(map(math.log, Ts)) / 8)


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
    Mtr = meta_matrix(ds, tr)
    Mva = meta_matrix(ds, val)
    Mtr_t = torch.from_numpy(Mtr)
    Mva_t = torch.from_numpy(Mva)
    yd = np.asarray(demo.labels)
    Xd = prep_batch(demo.series)
    Md = meta_matrix(demo, np.arange(len(demo)))

    from threadpoolctl import threadpool_limits

    print("=" * 78, flush=True)
    print("HPO STAGE 3 — untested levers on avg_aug40_c6w2", flush=True)
    print("objective: demo score (official formula, calib. ORT lat) + val score (control)", flush=True)
    print("=" * 78, flush=True)
    print(f"\n{'probe':<10} {'val_acc':>7} {'demo_acc':>8} {'np_us':>6} {'ort_us':>7} "
          f"{'score_demo':>10} {'score_val':>9}  demo pc6/pc7", flush=True)

    best_name, best_score = None, -1.0
    for name, cfg in PROBES:
        t0 = time.perf_counter()
        m, val_acc = train_probe(cfg, (Xtr_t, ytr_t, Xva_t, yva, Mtr_t, Mva_t))
        t_train = time.perf_counter() - t0
        w = export_np(m)
        np.savez(MODELS / f"hpo3_{name}.npz", **w)
        meta_on = cfg.get("meta", False)

        with threadpool_limits(limits=1):
            preds_d = np.empty(len(demo), dtype=np.int64)
            t0 = time.perf_counter()
            for j in range(len(demo)):
                mv = Md[j] if meta_on else None
                preds_d[j] = int(np.argmax(forward_np(Xd[j], w, m.k, m.dil3, mv)))
            np_fwd = (time.perf_counter() - t0) / len(demo)
            preds_v = np.empty(2000, dtype=np.int64)
            t0 = time.perf_counter()
            for i, j in enumerate(val[:2000]):
                mv = Mva[i] if meta_on else None
                preds_v[i] = int(np.argmax(forward_np(X[j], w, m.k, m.dil3, mv)))
            np_fwd_v = (time.perf_counter() - t0) / 2000
        demo_acc = float((preds_d == yd).mean())
        ort_est = 61e-6 + 0.24 * np_fwd
        pc_d = [float((preds_d[yd == c] == c).mean()) for c in range(8)]
        pc_v = [float((preds_v[y[val[:2000]] == c] == c).mean()) for c in range(8)]
        s_demo = score_from(pc_d, ort_est)
        s_val = score_from(pc_v, 61e-6 + 0.24 * np_fwd_v)
        if s_demo + s_val > best_score:
            best_score = s_demo + s_val
            best_name = name
        print(f"{name:<10} {val_acc:>7.3f} {demo_acc:>8.3f} {np_fwd*1e6:>6.0f} {ort_est*1e6:>7.0f} "
              f"{s_demo:>10.1f} {s_val:>9.1f}  {pc_d[6]:.2f}/{pc_d[7]:.2f}  ({t_train:.0f}s)",
              flush=True)

    print(f"\nBEST STAGE 3 (by combined demo+val): {best_name} ({best_score/2:.1f} avg)", flush=True)
    print(f"Stage-2 winner reference: demo 81.5 / val ~93 (est)", flush=True)
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()