"""10: Latenz-Profiler — wo geht die Inferenzzeit hin? + ONNX/onnxruntime/torch-Vergleich.

Profilet die v6-numpy-Pipeline Phase für Phase (prep, pad, view, matmul, relu,
pool, fc) auf 400 Demo-Samples, single-thread. Vergleicht dann:
  - unsere numpy-im2col-Pipeline
  - torch Conv1d (mit set_num_threads(1), zufällige Gewichte — Timing-äquivalent)
  - onnxruntime (ONNX-Export desselben Modells, intra_op=1)

Exportiert zusätzlich experiments/onnx/cnn_v6.onnx (für Netron-Visualisierung).

Log: experiments/logs/profile.log
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
ONNX_DIR = HERE / "onnx"
sys.path.insert(0, str(MLC_HOME))

from competition_api import load_dataset  # noqa: E402


def prep_one(series):
    x = np.array(series, dtype=np.float32)
    t = np.arange(x.shape[1])
    for ch in x:
        m = np.isnan(ch)
        if m.all():
            ch[:] = 0.0
        elif m.any():
            ch[m] = np.interp(t[m].astype(np.float64), t[~m], ch[~m].astype(np.float64))
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + np.float32(1e-9)
    return ((x - mu) / sd).astype(np.float32)


def phases(w, s):  # s = prepared (4,256) sample
    """Run the v6 pipeline phase by phase, timing each. Returns dict of seconds."""
    out = {}

    def conv(x, W, b, pad, dil=1):
        xp = np.pad(x, ((0, 0), (pad, pad)))
        sw = np.lib.stride_tricks.sliding_window_view(xp, W.shape[2], axis=1)
        cols = sw.transpose(0, 2, 1).reshape(-1, sw.shape[1])
        return W.reshape(W.shape[0], -1) @ cols + b[:, None]

    t = {}

    def timed(name, fn):
        t0 = time.perf_counter()
        r = fn()
        t[name] = t.get(name, 0.0) + (time.perf_counter() - t0)
        return r

    x = s
    x = timed("conv1(pad+view+matmul)", lambda: np.maximum(conv(x, w["c1_W"], w["c1_b"], 4), 0))
    x = timed("pool1", lambda: x[:, :64].reshape(16, 16, 4).max(-1))
    x = timed("conv2", lambda: np.maximum(conv(x, w["c2_W"], w["c2_b"], 4), 0))
    x = timed("pool2", lambda: x[:, :16].reshape(32, 4, 4).max(-1))
    x = timed("conv3(dil=2)", lambda: np.maximum(conv(x, w["c3_W"], w["c3_b"], 8, 2), 0))
    pooled = timed("global pool", lambda: np.concatenate([x.mean(1), x.max(1), x.std(1)]))
    logits = timed("fc", lambda: w["fc_W"] @ pooled + w["fc_b"])
    timed("argmax", lambda: int(np.argmax(logits)))
    return t


def main():
    w = {k: v for k, v in np.load(MODELS / "cnn2_weights.npz").items()}
    demo = load_dataset(MLC_HOME / "data" / "test_demo.npz")
    samples = [demo[i] for i in range(len(demo))]
    prepared = [prep_one(s.series) for s in samples]

    # warm
    for s in prepared[:20]:
        phases(w, s)

    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        # prep alone
        t0 = time.perf_counter()
        for s in samples:
            prep_one(s.series)
        prep_us = (time.perf_counter() - t0) / len(samples) * 1e6
        # phases
        agg = {}
        t0 = time.perf_counter()
        for s in prepared:
            for k, v in phases(w, s).items():
                agg[k] = agg.get(k, 0.0) + v
        total = time.perf_counter() - t0
    print(f"PREP (NaN-interp + z-score): {prep_us:7.0f} us", flush=True)
    print(f"FORWARD Phasen (Mittelwert über {len(samples)} Samples):", flush=True)
    s_ = 0.0
    for k, v in agg.items():
        us = v / len(samples) * 1e6
        s_ += us
        print(f"  {k:28s} {us:7.1f} us", flush=True)
    print(f"  {'SUMME forward':28s} {s_:7.1f} us", flush=True)
    print(f"  {'GESAMT (prep+fwd)':28s} {prep_us + s_:7.1f} us", flush=True)

    # ---------------- torch comparison ----------------
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(1)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv1d(4, 16, 9, padding=4)
            self.c2 = nn.Conv1d(16, 32, 9, padding=4)
            self.c3 = nn.Conv1d(32, 64, 9, padding=8, dilation=2)

        def forward(self, x):
            x = F.max_pool1d(F.relu(self.c1(x)), 4)
            x = F.max_pool1d(F.relu(self.c2(x)), 4)
            x = F.relu(self.c3(x))
            return torch.cat([x.mean(-1), x.amax(-1), x.std(-1, unbiased=False)], 1)

    m = CNN().eval()
    xt = torch.from_numpy(prepared[0])[None]
    with torch.no_grad():
        m(xt)  # warm
        with threadpool_limits(limits=1):
            t0 = time.perf_counter()
            for s in prepared:
                m(torch.from_numpy(s)[None])
            torch_us = (time.perf_counter() - t0) / len(prepared) * 1e6
    print(f"\nTORCH conv (single-thread): {torch_us:7.0f} us  ({prep_us + torch_us:.0f} inkl prep)", flush=True)

    # ---------------- ONNX export + onnxruntime ----------------
    ONNX_DIR.mkdir(exist_ok=True)
    onnx_path = ONNX_DIR / "cnn_v6.onnx"
    try:
        torch.onnx.export(m, xt, str(onnx_path), input_names=["series"], output_names=["pooled"],
                          opset_version=17, dynamo=False)
        print(f"\nONNX exportiert: {onnx_path}", flush=True)
    except Exception as e:
        print(f"\nONNX-Export fehlgeschlagen: {e}", flush=True)

    try:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])
        inp = {"series": prepared[0][None]}
        sess.run(None, inp)  # warm
        with threadpool_limits(limits=1):
            t0 = time.perf_counter()
            for s in prepared:
                sess.run(None, {"series": s[None]})
            ort_us = (time.perf_counter() - t0) / len(prepared) * 1e6
        print(f"ONNXRUNTIME (1 thread):    {ort_us:7.0f} us  ({prep_us + ort_us:.0f} inkl prep)", flush=True)
    except ImportError:
        print("\nonnxruntime nicht installiert (pip install onnxruntime)", flush=True)

    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()