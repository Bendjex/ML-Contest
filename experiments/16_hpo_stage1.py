"""16: Distribution-shift diagnosis + HPO stage 1 (one-lever probes).

Stage A — shift diagnosis: why is demo (eval-like) acc 0.79 while train-val is 0.91?
Per-class fingerprint comparison: train subsample (2k/class) vs demo (50/class):
acf1/acf8, low-freq power, roughness, cross-channel corr, NaN fraction, level, sr share.

Stage B — HPO stage 1: one lever at a time, 20 epochs each (~3 min per probe):
  probes: baseline / k1=15 / k3=15 / wider / narrower / avg-pool / gelu / leaky /
  drop-std-gp / quantile-gp
Objective = OFFICIAL score formula: per-class acc measured on demo (eval-like
preset!), n_k = (60 s / estimated ORT latency) / 8. ORT latency estimated from
numpy forward via the v6 calibration factor (ORT_conv = 0.24 x numpy_fwd).
Also reports train-val acc (fit quality) so we see fit-vs-generalization per probe.

Log: experiments/logs/hpo_stage1.log
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

EPOCHS = 20
BATCH = 256
LR = 1e-3

FINGER_PRINTS = ("acf1", "acf8", "lowf", "rough", "xcorr", "nan", "level_std", "sr2")


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


def fingerprint(Xz, sr, lev, nanfrac):
    """Xz: (n,4,256) standardized. Returns dict of scalar fingerprint means."""
    z = Xz
    acf1 = np.mean(z[:, :, :-1] * z[:, :, 1:])
    acf8 = np.mean(z[:, :, :-8] * z[:, :, 8:])
    P = np.abs(np.fft.rfft(z, axis=2)) ** 2
    lowf = (P[:, :, 1:6].sum(-1) / (P.sum(-1) + 1e-9)).mean()
    rough = np.abs(np.diff(z, axis=2)).mean()
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    xcorr = np.mean([abs(np.mean(z[:, i] * z[:, j])) for i, j in pairs])
    return dict(acf1=acf1, acf8=acf8, lowf=lowf, rough=rough, xcorr=xcorr,
                nan=nanfrac, level_std=lev.std(), sr2=float(np.mean(np.asarray(sr) == 2.0)))


def stage_a(ds, demo):
    y = np.asarray(ds.labels)
    sr = np.asarray(ds.meta["sample_rate"], float)
    lev = np.asarray(ds.meta["level"], float)
    nanfrac = np.isnan(ds.series).mean(axis=(1, 2))
    X = prep_batch(ds.series)
    rng = np.random.default_rng(5)
    print("=" * 78, flush=True)
    print("STAGE A: SHIFT DIAGNOSIS  (train preset vs demo preset, per class)", flush=True)
    print("=" * 78, flush=True)
    yd = np.asarray(demo.labels)
    srd = np.asarray(demo.meta["sample_rate"], float)
    levd = np.asarray(demo.meta["level"], float)
    nanfd = np.isnan(demo.series).mean(axis=(1, 2))
    Xd = prep_batch(demo.series)
    hdr = f"{'cls':>3} | " + " | ".join(f"{k:>8}" for k in FINGER_PRINTS) + "  (train -> demo)"
    print(hdr, flush=True)
    for c in range(8):
        m_tr = y == c
        idx = rng.choice(np.where(m_tr)[0], 2000, replace=False)
        fp_tr = fingerprint(X[idx], sr[idx], lev[idx], nanfrac[idx].mean())
        m_de = yd == c
        fp_de = fingerprint(Xd[m_de], srd[m_de], levd[m_de], nanfd[m_de].mean())
        row = f"{c:>3} | " + " | ".join(
            f"{fp_tr[k]:+.3f}->{fp_de[k]:+.3f}"[:17].rjust(8 + 8 + 1)[:17] for k in FINGER_PRINTS)
        print(row, flush=True)
    print(f"\nglobal NaN: train {nanfrac.mean():.4f} vs demo {nanfd.mean():.4f}", flush=True)
    print(f"global sr=2 share: train {np.mean(sr==2):.3f} vs demo {np.mean(srd==2):.3f}", flush=True)
    print(f"global level std: train {lev.std():.2f} vs demo {levd.std():.2f}", flush=True)
    print("", flush=True)


# ---------------------------------------------------------------- model family
class CNN(nn.Module):
    def __init__(self, ch=(16, 32, 64), k=(9, 9, 9), dil3=2, act="relu",
                 pool="max", gp="mms"):
        super().__init__()
        self.ch, self.k, self.dil3, self.pool, self.gp = ch, k, dil3, pool, gp
        self.c1 = nn.Conv1d(4, ch[0], k[0], padding=k[0] // 2)
        self.b1 = nn.BatchNorm1d(ch[0])
        self.c2 = nn.Conv1d(ch[0], ch[1], k[1], padding=k[1] // 2)
        self.b2 = nn.BatchNorm1d(ch[1])
        self.c3 = nn.Conv1d(ch[1], ch[2], k[2], padding=(k[2] // 2) * dil3, dilation=dil3)
        self.b3 = nn.BatchNorm1d(ch[2])
        gpd = {"mms": 3, "mm": 2, "mmsq": 5}[gp]
        self.fc = nn.Linear(gpd * ch[2], 8)
        self.act = {"relu": F.relu, "gelu": F.gelu,
                    "leaky": lambda t: F.leaky_relu(t, 0.1)}[act]

    def forward(self, x):
        pool = F.max_pool1d if self.pool == "max" else F.avg_pool1d
        x = pool(self.act(self.b1(self.c1(x))), 4)
        x = pool(self.act(self.b2(self.c2(x))), 4)
        x = self.act(self.b3(self.c3(x)))
        if self.gp == "mms":
            x = torch.cat([x.mean(-1), x.amax(-1), x.std(-1, unbiased=False)], 1)
        elif self.gp == "mm":
            x = torch.cat([x.mean(-1), x.amax(-1)], 1)
        else:
            q = torch.quantile(x, torch.tensor([0.1, 0.9], device=x.device), dim=-1)
            x = torch.cat([x.mean(-1), x.amax(-1), x.std(-1, unbiased=False),
                            q[0], q[1]], 1)
        return self.fc(x)


def export_np(model: CNN):
    """Fold BN into convs, return weight dict + numpy forward closure params."""
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


def np_act(name, x):
    if name == "relu":
        return np.maximum(x, 0)
    if name == "gelu":
        return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
    return np.where(x > 0, x, np.float32(0.1) * x)


def forward_np(x, w, ch, k, dil3, act, pool, gp):
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
    x = poolf(np_act(act, conv(x, w["c1_W"], w["c1_b"], k[0] // 2)), 4)
    x = poolf(np_act(act, conv(x, w["c2_W"], w["c2_b"], k[1] // 2)), 4)
    x = np_act(act, conv(x, w["c3_W"], w["c3_b"], (k[2] // 2) * dil3, dil3))
    if gp == "mms":
        p = np.concatenate([x.mean(1), x.max(1), x.std(1)])
    elif gp == "mm":
        p = np.concatenate([x.mean(1), x.max(1)])
    else:
        srt = np.sort(x, axis=1)
        n = x.shape[1]
        q10 = srt[:, max(0, int(0.1 * (n - 1)))]
        q90 = srt[:, max(0, int(0.9 * (n - 1)))]
        p = np.concatenate([x.mean(1), x.max(1), x.std(1), q10, q90])
    return w["fc_W"] @ p + w["fc_b"]


PROBES = [
    ("baseline_v6", dict()),
    ("k1_15", dict(k=(15, 9, 9))),
    ("k3_15", dict(k=(9, 9, 15))),
    ("wide_24_48_96", dict(ch=(24, 48, 96))),
    ("narrow_12_24_48", dict(ch=(12, 24, 48))),
    ("avg_pool", dict(pool="avg")),
    ("gelu", dict(act="gelu")),
    ("leaky01", dict(act="leaky")),
    ("gp_mean_max", dict(gp="mm")),
    ("gp_plus_quantiles", dict(gp="mmsq")),
]


def train_probe(cfg, yv_split, seed=0):
    Xtr_t, ytr_t, Xva_t, yva = yv_split
    torch.manual_seed(seed)
    m = CNN(**cfg)
    counts = np.bincount(ytr_t.numpy(), minlength=8)
    cw = torch.tensor(len(ytr_t) / (8 * counts), dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss(weight=cw)
    L = 256
    ar = torch.arange(L)
    best, state = -1.0, None
    for ep in range(EPOCHS):
        m.train()
        perm = torch.randperm(len(ytr_t))
        for i in range(0, len(ytr_t), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            sh = torch.randint(0, L, (xb.shape[0],))
            gi = ((ar[None, :] + sh[:, None]) % L).unsqueeze(1).expand(-1, 4, -1)
            xb = torch.gather(xb, 2, gi)
            xb = xb * (torch.rand_like(xb) > 0.014).float()
            xb = xb + 0.05 * torch.randn_like(xb)
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

    stage_a(ds, demo)

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
    demo_samples = [demo[j] for j in range(len(demo))]

    from threadpoolctl import threadpool_limits

    print("=" * 78, flush=True)
    print("STAGE B: HPO STAGE 1 — one-lever probes", flush=True)
    print(f"epochs={EPOCHS}, objective = official score formula on DEMO per-class acc", flush=True)
    print("  (demo n=50/class → noisy!; train-val acc reported as fit reference)", flush=True)
    print("=" * 78, flush=True)
    print(f"\n{'probe':<18} {'val_acc':>7} {'demo_acc':>8} {'np_fwd_us':>9} "
          f"{'ort_est_us':>10} {'n(60s)':>9} {'score':>7}", flush=True)

    results = []
    for name, cfg in PROBES:
        t0 = time.perf_counter()
        m, val_acc = train_probe(cfg, (Xtr_t, ytr_t, Xva_t, yva))
        t_train = time.perf_counter() - t0
        w = export_np(m)
        ch, k = m.ch, m.k
        dil3, act, pool, gp = m.dil3, m.act.__name__ if hasattr(m.act, "__name__") else "relu", m.pool, m.gp
        act_n = "gelu" if "gelu" in name or cfg.get("act") == "gelu" else ("leaky" if cfg.get("act") == "leaky" else "relu")
        with threadpool_limits(limits=1):
            preds_d = np.empty(len(demo), dtype=np.int64)
            t0 = time.perf_counter()
            for j in range(len(demo)):
                xx = Xd[j]
                pr = None
                # single-sample numpy forward, prep excluded (measure separately)
                preds_d[j] = int(np.argmax(forward_np(xx, w, ch, k, dil3, act_n, pool, gp)))
            np_fwd = (time.perf_counter() - t0) / len(demo)
        demo_acc = float((preds_d == yd).mean())
        ort_est = 61e-6 + 0.24 * np_fwd          # calibrated on v6 (61us prep + ORT≈0.24x numpy)
        n_total = 60.0 / ort_est
        n_k = n_total / 8
        pc = [float((preds_d[yd == c] == c).mean()) for c in range(8)]
        Ts = [max((n_k ** 0.5) * (a - 0.33 * (1 - a)) * a ** 2, 0.01) for a in pc]
        score = math.exp(sum(map(math.log, Ts)) / 8)
        results.append((name, val_acc, demo_acc, np_fwd, ort_est, score, t_train))
        print(f"{name:<18} {val_acc:>7.3f} {demo_acc:>8.3f} {np_fwd*1e6:>9.0f} "
              f"{ort_est*1e6:>10.0f} {n_total:>9,.0f} {score:>7.1f}   ({t_train:.0f}s train)",
              flush=True)
        np.savez(MODELS / f"hpo_{name}.npz", **w)

    print("\nPER-CLASS demo acc of best probe:", flush=True)
    best = max(results, key=lambda r: r[5])
    print(f"  {best[0]}: ", flush=True)
    # recompute per-class for the best probe
    w = {k: v for k, v in np.load(MODELS / f"hpo_{best[0]}.npz").items()}
    cfg = dict(PROBES)[best[0]]
    m = CNN(**cfg)
    act_n = cfg.get("act", "relu")
    preds_d = np.array([int(np.argmax(forward_np(Xd[j], w, m.ch, m.k, m.dil3, act_n, m.pool, m.gp)))
                        for j in range(len(demo))])
    for c in range(8):
        a = float((preds_d[yd == c] == c).mean())
        print(f"    class {c}: {a:.3f}", flush=True)
    print("\n=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()