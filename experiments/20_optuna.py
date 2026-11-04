"""20: Optuna HPO study — TPE sampler + median pruning on the CNN design space.

Fixed from stage 1-3 findings: avg_pool intermediate pooling, global mean+max+std,
meta features concatenated (stage-3 winner: demo score 87.3 vs 81.5!), ffill/bfill prep.

Search space (10 dims):
  ch    — channel triple (16,32,64) / (24,48,96) / (12,24,48) / (20,40,80) / (16,32,48) / (24,48,72)
  k1    — 7, 9, 11 (stage 1 showed k1=15 loses)
  k3    — 9, 11, 13 (dil3 spreads it; stage 1 showed k3=15 loses)
  dil3  — 2, 3, 4
  epochs — 25..60 (cosine T_max=epochs)
  lr    — log 5e-4..2e-3
  sigma — 0.05..0.18 (noise aug)
  nanp  — 0.008..0.03 (NaN-dropout aug)
  c6w   — 1.0..3.0 (class-6 weight)
  c7w   — 0.7..2.5 (class-7 weight)

Pruning: per-epoch val_acc reported; MedianPruner(n_warmup=8, interval=2).

Objective: 0.5*score_demo + 0.5*score_val — official formula per class,
n_k from calibrated ORT latency (61us prep + 0.24x numpy fwd).
Demo (n=50/class) = honest target, val (n=250/class) = noise control.

Every COMPLETE trial saves models/optuna_trial{n}.npz (no study-closure bugs);
best trial copied to models/optuna_best.npz at the end.
Storage: sqlite at experiments/optuna/study.db (resumable — rerun adds trials).

Log: experiments/logs/optuna.log (via run_20_after_stage3.sh)
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
OPTUNA_DIR = HERE / "optuna"
sys.path.insert(0, str(MLC_HOME))

from competition_api import load_dataset  # noqa: E402

import optuna  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

BATCH = 256
N_TRIALS = 40

# --------------------------------------------------------------- data (loaded once)
class Data:
    pass


D = Data()


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


class CNN(nn.Module):
    def __init__(self, ch, k, dil3):
        super().__init__()
        self.ch, self.k, self.dil3 = ch, k, dil3
        self.c1 = nn.Conv1d(4, ch[0], k[0], padding=k[0] // 2)
        self.b1 = nn.BatchNorm1d(ch[0])
        self.c2 = nn.Conv1d(ch[0], ch[1], k[1], padding=k[1] // 2)
        self.b2 = nn.BatchNorm1d(ch[1])
        self.c3 = nn.Conv1d(ch[1], ch[2], k[2], padding=(k[2] // 2) * dil3, dilation=dil3)
        self.b3 = nn.BatchNorm1d(ch[2])
        self.fc = nn.Linear(3 * ch[2] + 5, 8)

    def forward(self, x, m):
        x = F.avg_pool1d(F.relu(self.b1(self.c1(x))), 4)
        x = F.avg_pool1d(F.relu(self.b2(self.c2(x))), 4)
        x = F.relu(self.b3(self.c3(x)))
        feats = torch.cat([x.mean(-1), x.amax(-1), x.std(-1, unbiased=False), m], 1)
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


def forward_np(x, w, k, dil3, meta_vec):
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
    p = np.concatenate([x.mean(1), x.max(1), x.std(1), meta_vec])
    return w["fc_W"] @ p + w["fc_b"]


def score_from(pc, lat_s):
    n_k = (60.0 / lat_s) / 8
    Ts = [max((n_k ** 0.5) * (a - 0.33 * (1 - a)) * a ** 2, 0.01) for a in pc]
    return math.exp(sum(map(math.log, Ts)) / 8)


def objective(trial):
    ch = trial.suggest_categorical("ch", [(16, 32, 64), (24, 48, 96), (12, 24, 48),
                                          (20, 40, 80), (16, 32, 48), (24, 48, 72)])
    k1 = trial.suggest_categorical("k1", [7, 9, 11])
    k3 = trial.suggest_categorical("k3", [9, 11, 13])
    dil3 = trial.suggest_categorical("dil3", [2, 3, 4])
    epochs = trial.suggest_int("epochs", 25, 60)
    lr = trial.suggest_float("lr", 5e-4, 2e-3, log=True)
    sigma = trial.suggest_float("sigma", 0.05, 0.18)
    nanp = trial.suggest_float("nanp", 0.008, 0.03)
    c6w = trial.suggest_float("c6w", 1.0, 3.0)
    c7w = trial.suggest_float("c7w", 0.7, 2.5)
    k = (k1, 9, k3)

    torch.manual_seed(trial.number)
    m = CNN(ch, k, dil3)
    cw_np = np.ones(8, dtype=np.float64)
    cw_np[6] = c6w
    cw_np[7] = c7w
    base = len(D.ytr) / (8.0 * D.counts)          # balanced base weights
    cw = torch.tensor((base * cw_np).astype(np.float32))

    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=cw)
    L = 256
    ar = torch.arange(L)
    best, state = -1.0, None
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(D.ytr))
        for i in range(0, len(D.ytr), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = D.Xtr[idx], D.ytr[idx]
            sh = torch.randint(0, L, (xb.shape[0],))
            gi = ((ar[None, :] + sh[:, None]) % L).unsqueeze(1).expand(-1, 4, -1)
            xb = torch.gather(xb, 2, gi)
            xb = xb * (torch.rand_like(xb) > nanp).float()
            xb = xb + sigma * torch.randn_like(xb)
            opt.zero_grad()
            loss = lossf(m(xb, D.Mtr[idx]), yb)
            loss.backward()
            opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            acc = float((m(D.Xva, D.Mva).argmax(1).numpy() == D.yva_np).mean())
        trial.report(acc, ep)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if acc > best:
            best = acc
            state = {kk: v.detach().clone() for kk, v in m.state_dict().items()}
    m.load_state_dict(state)
    m.eval()

    w = export_np(m)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        preds_d = np.empty(len(D.yd), dtype=np.int64)
        t0 = time.perf_counter()
        for j in range(len(D.yd)):
            preds_d[j] = int(np.argmax(forward_np(D.Xd[j], w, m.k, m.dil3, D.Md[j])))
        np_fwd = (time.perf_counter() - t0) / len(D.yd)
        preds_v = np.empty(D.n_val_eval, dtype=np.int64)
        for i in range(D.n_val_eval):
            j = int(D.val_idx[i])
            preds_v[i] = int(np.argmax(forward_np(D.Xnp[j], w, m.k, m.dil3, D.Mnp_val[i])))
    ort_est = 61e-6 + 0.24 * np_fwd
    pc_d = [float((preds_d[D.yd == c] == c).mean()) for c in range(8)]
    pc_v = [float((preds_v[D.yv_eval == c] == c).mean()) for c in range(8)]
    s_demo = score_from(pc_d, ort_est)
    s_val = score_from(pc_v, ort_est)
    combined = 0.5 * s_demo + 0.5 * s_val

    trial.set_user_attr("val_acc", best)
    trial.set_user_attr("demo_acc", float((preds_d == D.yd).mean()))
    trial.set_user_attr("score_demo", s_demo)
    trial.set_user_attr("score_val", s_val)
    trial.set_user_attr("ort_us", ort_est * 1e6)
    trial.set_user_attr("pc6_pc7", [pc_d[6], pc_d[7]])
    np.savez(MODELS / f"optuna_trial{trial.number}.npz", **w)

    print(f"[trial {trial.number}] combined={combined:.1f} demo={s_demo:.1f} val={s_val:.1f} "
          f"val_acc={best:.3f} ort={ort_est*1e6:.0f}us pc6/7={pc_d[6]:.2f}/{pc_d[7]:.2f} "
          f"({time.strftime('%H:%M')})", flush=True)
    return combined


def main():
    torch.set_num_threads(6)
    ds = load_dataset(MLC_HOME / "data" / "train_100k.npz")
    demo = load_dataset(MLC_HOME / "data" / "test_demo.npz")
    y = np.asarray(ds.labels, dtype=np.int64)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    val, tr = order[:20_000], order[20_000:]
    X = prep_batch(ds.series)

    D.Xnp = X                                   # full numpy (absolute indexing)
    D.Xtr = torch.from_numpy(X[tr])
    D.ytr = torch.from_numpy(y[tr])
    D.Mtr = torch.from_numpy(meta_matrix(ds, tr))
    D.counts = np.bincount(y[tr], minlength=8)
    D.Xva = torch.from_numpy(X[val])
    D.yva_np = y[val]
    D.Mva = torch.from_numpy(meta_matrix(ds, val))
    D.Mnp_val = meta_matrix(ds, val)           # numpy meta for forward_np
    D.val_idx = np.arange(2000)
    D.n_val_eval = 2000
    D.yv_eval = y[val[:2000]]
    D.yd = np.asarray(demo.labels)
    D.Xd = prep_batch(demo.series)
    D.Md = meta_matrix(demo, np.arange(len(demo)))

    OPTUNA_DIR.mkdir(exist_ok=True, parents=True)
    storage = f"sqlite:///{OPTUNA_DIR / 'study.db'}"
    study = optuna.create_study(
        study_name="cnn_hpo", direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8, interval_steps=2),
        storage=storage, load_if_exists=True,
    )
    n_done = len([t for t in study.trials if t.state.is_finished()])
    print(f"study cnn_hpo: {n_done} finished trials in db, running {N_TRIALS} more", flush=True)
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

    b = study.best_trial
    src = MODELS / f"optuna_trial{b.number}.npz"
    if src.exists():
        shutil.copy(src, MODELS / "optuna_best.npz")
    print("\nBEST TRIAL:", flush=True)
    print(f"  combined={b.value:.1f} demo={b.user_attrs['score_demo']:.1f} "
          f"val={b.user_attrs['score_val']:.1f} val_acc={b.user_attrs['val_acc']:.3f} "
          f"ort={b.user_attrs['ort_us']:.0f}us pc6/7={b.user_attrs['pc6_pc7']}", flush=True)
    print(f"  params: {b.params}", flush=True)
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()