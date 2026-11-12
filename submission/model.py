"""Final submission model: 1D-CNN (Optuna Trial 20) via ONNX Runtime single-thread (with NumPy fallback)."""
from __future__ import annotations

from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ONNX_CANDIDATES = [
    HERE / "cnn_final.onnx",
    HERE.parent / "cnn_final.onnx",
    HERE.parent / "submission" / "cnn_final.onnx",
]
NPZ_CANDIDATES = [
    HERE / "optuna_best.npz",
    HERE.parent / "optuna_best.npz",
    HERE.parent / "experiments" / "models" / "optuna_best.npz",
]


class Model:
    def __init__(self) -> None:
        self.sess = None
        self.ONNX_FILE = None
        for p in ONNX_CANDIDATES:
            if p.exists():
                self.ONNX_FILE = p
                break

        if self.ONNX_FILE is not None:
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.sess = ort.InferenceSession(
                    str(self.ONNX_FILE),
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
                self.weights = None
                return
            except ImportError:
                self.sess = None

        npz_file = None
        for p in NPZ_CANDIDATES:
            if p.exists():
                npz_file = p
                break
        if npz_file is None:
            raise FileNotFoundError(
                "Neither cnn_final.onnx (with onnxruntime) nor optuna_best.npz found."
            )
        self.weights = {k: v for k, v in np.load(npz_file).items()}

    @staticmethod
    def _prep(series: np.ndarray) -> np.ndarray:
        """(4, 256) float32: exact vectorized ffill + bfill + per-channel z-score."""
        x = np.array(series, dtype=np.float32)
        t = np.arange(x.shape[1], dtype=np.int32)
        rows = np.arange(x.shape[0])[:, None]

        idx = np.where(~np.isnan(x), t[None, :], 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        xf = x[rows, idx]

        idxb = np.where(~np.isnan(xf[:, ::-1]), t[None, :], 0)
        np.maximum.accumulate(idxb, axis=1, out=idxb)
        xb = xf[rows, idxb[:, ::-1]]

        xb = np.where(np.isnan(xb), np.float32(0.0), xb)

        mu = xb.mean(axis=1, keepdims=True)
        sd = xb.std(axis=1, keepdims=True) + np.float32(1e-9)
        return ((xb - mu) / sd).astype(np.float32)

    @staticmethod
    def _meta(meta: dict | None) -> np.ndarray:
        """Extract and normalize [sample_rate, group_A, group_B, group_C, level] -> (5,) float32."""
        if not meta:
            return np.zeros(5, dtype=np.float32)
        sr = float(meta.get("sample_rate", 1.5))
        g = str(meta.get("group", ""))
        lev = float(meta.get("level", 0.0))
        return np.array(
            [
                (sr - 1.5) / 0.5,
                float(g == "A"),
                float(g == "B"),
                float(g == "C"),
                lev / 4.0,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _forward_np(x: np.ndarray, w: dict, meta_vec: np.ndarray) -> np.ndarray:
        def conv(inp, W, b, pad, dil=1):
            C_out, C_in, K = W.shape
            xp = np.pad(inp, ((0, 0), (pad, pad)))
            L_out = xp.shape[1] - dil * (K - 1)
            t = np.arange(L_out)[:, None]
            j = dil * np.arange(K)[None, :]
            cols = xp[:, t + j].transpose(1, 0, 2).reshape(L_out, C_in * K)
            return W.reshape(C_out, C_in * K) @ cols.T + b[:, None]

        def pavg(inp, p):
            C, L = inp.shape
            return inp[:, :(L // p) * p].reshape(C, L // p, p).mean(-1)

        x = pavg(np.maximum(conv(x, w["c1_W"], w["c1_b"], 3), 0), 4)
        x = pavg(np.maximum(conv(x, w["c2_W"], w["c2_b"], 6), 0), 4)
        x = np.maximum(conv(x, w["c3_W"], w["c3_b"], 8, 2), 0)
        p = np.concatenate([x.mean(1), x.max(1), x.std(1), meta_vec])
        return w["fc_W"] @ p + w["fc_b"]

    def predict(self, x) -> int:
        s = self._prep(x.series)
        m = self._meta(getattr(x, "meta", None))
        if self.sess is not None:
            logits = self.sess.run(None, {"series": s[None, :, :], "meta": m[None, :]})[0][0]
            return int(np.argmax(logits))
        return int(np.argmax(self._forward_np(s, self.weights, m)))
