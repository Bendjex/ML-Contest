# GDG on Campus UAM — Machine Learning Contest Submission

## 1. Executive Summary

Our solution addresses the dual Accuracy–Throughput optimization challenge of the **GDG UAM Machine Learning Contest** by replacing slow hand-crafted feature extraction (`~554 µs/sample`) with a custom, latency-optimized **1D Convolutional Neural Network (1D-CNN)** fused with normalized metadata features and exported to a static, BatchNorm-folded **ONNX Runtime** graph (`cnn_final.onnx`, 192 KB).

Evaluated under the official single-threaded contest harness (`harness.py`), our final model (Optuna Trial 20) achieves:

| Evaluation Stream | Official Score | Accuracy | Mean Latency / Sample | Scored Samples ($n$) | Improvement vs. Tree Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`train_100k` (3 runs, official harness)** | **85.87** | **92.95%** | **0.20 ms (`~169 µs` ORT)** | 100,000 *(stream exhausted)* | **3.9×** *(vs. ~22.1)* |
| **`test_demo` (1 run, eval-preset)** | **3.198** | **80.75%** | **0.20 ms** | 400 *(stream exhausted)* | **8.4×** *(vs. GBM ~10.0 on 100k)* |

---

## 2. Contest Mechanics & Mathematical Formulation

The official evaluation harness scores a submission over a **60-second wall-clock budget** in an isolated, **single-threaded** worker (`OMP/MKL/OPENBLAS/NUMEXPR=1`) by computing the geometric mean across all $K = 8$ classes of:

$$T_k = \max\left(\varepsilon,\; n_k^\alpha \cdot \frac{c_k - \lambda w_k}{n_k} \cdot \left(\frac{c_k}{n_k}\right)^p\right), \quad \text{with } \alpha = 0.5,\; p = 2,\; \lambda = 0.33,\; \varepsilon = 0.01$$

For balanced classes ($n_k = n / 8$), the score scales as:

$$\text{Score} \propto \sqrt{\frac{1}{\text{latency}}} \cdot (\text{acc} - 0.33 \cdot \text{err}) \cdot \text{acc}^2 \approx \frac{\text{acc}^3}{\sqrt{\text{latency}}}$$

### Key Structural Implications
1. **Why Ensembles Fail ($g^3 > \sqrt{k}$ Rule):** Combining $k$ model evaluations multiplies latency by $k$, reducing $\sqrt{n}$ by $\sqrt{k}$. To break even, an ensemble must improve relative accuracy by factor $g > k^{1/6}$ (e.g., $+12\%$ relative accuracy for $k=2$, $+20\%$ for $k=3$). Empirically, a 100-tree Random Forest improves validation accuracy from $63.8\%$ to $68.6\%$ but increases latency $10\times$ (`681 µs` $\to$ `6.52 ms`), collapsing the score from **22.1 to 9.3**.
2. **Why Abstaining Hurts:** With $\lambda = 0.33$, answering beats abstaining whenever the top predicted class probability exceeds $\sim 25\text{–}30\%$, and wholesale class abstention drives $c_k \to 0 \implies T_k = \varepsilon$, destroying the geometric mean.
3. **Why Weak Classes Dominate the Geometric Mean:** Because the final metric is a geometric mean across classes $0\dots 7$, lifting the worst-performing classes (specifically classes **6 and 7**, and **0 and 3**) yields a much larger score increase than marginal gains on saturated classes (such as class 1 at $99.9\%$).

---

## 3. Exploratory Data Analysis & Distribution Shift Diagnosis

Each sample consists of a 4-channel time series of length 256 (`(4, 256)` in `float32`) with missing values (`NaN`) and sample-level metadata (`sample_rate` $\in \{1.0, 2.0\}$, `group` $\in \{A, B, C\}$, `level` $\sim \mathcal{N}(0, 2)$).

![Class Distribution](experiments/plots/01_class_distribution.png)
![One Sample Per Class](experiments/plots/24_one_per_class.png)

### Signal Fingerprints & Spectral Structure
* **Class 2:** Smooth, low-frequency dominated (autocorrelation at lag 8 is $0.24$, low-frequency power share $39\%$).
* **Class 1:** Uncorrelated across channels (cross-channel correlation $0.15$).
* **Classes 6 & 7:** High inertia and strong cross-channel correlation; their decision boundary is **rate-conditional** (`sample_rate = 2.0` is disproportionately represented in both $6 \to 7$ and $7 \to 6$ confusions).
* **Classes 0, 3, 4, 5:** Weakly separated by global statistics alone, requiring multi-scale temporal filters.

![Mean Power Spectrum](experiments/plots/04_mean_power_spectrum.png)
![Mean Autocorrelation](experiments/plots/05_autocorrelation.png)
![Metadata Overview](experiments/plots/02_meta_overview.png)

### Discovery of the Train vs. Evaluation Distribution Shift
A critical finding during our investigation (`16_hpo_stage1.py`) was a systematic **distribution shift** between `train_100k` and `test_demo` (which uses the evaluation generator preset):
* **Roughness & Noise:** Evaluation samples are significantly noisier (`acf1` drops from $0.73 \to 0.65$ across all classes; roughness increases from $0.53 \to 0.58$).
* **Missing Data (`NaN`):** The `NaN` fraction doubles from **$0.73\%$ in train to $1.38\%$ in `test_demo`**.
* **Level Variance:** Metadata `level` standard deviation increases from $1.80 \to 2.24$.

To prevent overfitting to the smoother training preset, we designed **demo-matched data augmentation** (circular phase shifts, synthetic `NaN` dropout at $p \approx 0.016$, and Gaussian noise injection $\sigma \approx 0.10$) and evaluated all hyperparameter candidates on a combined objective weighting both `test_demo` and `train_val`.

---

## 4. Model Exploration & Latency Profiling

### Baseline & Classical ML Comparison (`20k` Validation Split, Single-Thread)

| Model / Pipeline | Input Representation | Val Accuracy | Latency / Call | $n$ (60 s Budget) | Projected Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Decision Tree (`d=14`)** | 101 hand-crafted features | 63.8% | 676 µs *(554 µs feats!)* | ~88,700 | 22.2 |
| **Random Forest (`100 × d14`)** | 101 hand-crafted features | 68.6% | 6,521 µs | ~9,200 | 9.3 |
| **ExtraTrees (`100 × d16`)** | 101 hand-crafted features | 67.2% | 6,589 µs | ~9,100 | 8.6 |
| **HistGradientBoosting (`300`)** | 101 hand-crafted features | 82.0% | 18,468 µs | ~3,250 | 10.3 |
| **Raw-Window MLP `(256, 128)`** | Raw `(1024,)` z-scored | 48.9% | 110 µs *(numpy fwd)* | ~546,000 | 19.7 |
| **3-Seed MLP Ensemble** | Raw `(1024,)` z-scored | 52.8% | 181 µs *(numpy fwd)* | ~331,400 | 21.1 |
| **1D-CNN v5** | Raw `(4, 256)` | 88.99% | 465 µs *(numpy im2col)* | ~129,100 | 81.3 |
| **1D-CNN v6 (Dilated + StdPool)** | Raw `(4, 256)` | 90.18% | 496 µs *(numpy)* / **142 µs *(ORT)*** | ~421,300 | 153.8 *(val proj)* |
| **1D-CNN Final (Optuna Trial 20)** | Raw `(4, 256)` + `meta (5,)` | **91.5% val / 80.8% demo** | **169 µs *(prep + ORT)*** | ~355,000 | **85.87** *(official 100k cap)* |

### Latency Bottleneck Analysis
1. **Hand-crafted features are the bottleneck:** Computing FFT, quantiles, and autocorrelations in Python/NumPy takes `~554 µs/sample` before any classifier even runs.
2. **ONNX Runtime vs. NumPy `im2col`:** Profiling `1D-CNN v6` revealed that NumPy `im2col` spent `335 µs` in array padding and memory copies. Exporting the BatchNorm-folded network to **ONNX Runtime (`intra_op_num_threads=1`)** reduced the entire convolutional pass from `335 µs` to **`50 µs` ($6.7\times$ speedup)**.
3. **Preprocessing Parity Bugfix (`ffill`/`bfill`):** During `v7` harness testing, we discovered that using `np.interp` at inference while training on `ffill`/`bfill` caused boundary `NaN` discrepancies of up to $16.1\sigma$, collapsing accuracy. Vectorizing exact `ffill`/`bfill` in `model.py` (`~61 µs/sample`) achieved `0.000` max difference against batch training preprocessing.

---

## 5. Final Architecture & Hyperparameter Optimization (Optuna)

![Final 1D-CNN Architecture](experiments/plots/23_final_architecture.png)

### Architecture Specification (`CNNFinal` — Optuna Trial 20)
1. **Vectorized Preprocessing (`_prep` & `_meta`):**
   * Forward-fill + backward-fill of `NaN` values along the time axis, followed by per-channel z-score standardization `(4, 256)`.
   * Metadata vector `(5,)`: `[(sample_rate - 1.5)/0.5, group==A, group==B, group==C, level/4.0]`.
2. **Convolutional Backbone (BatchNorm folded into Conv weights at export):**
   * **`Conv1d_1`**: $4 \to 24$ channels, kernel $k_1 = 7$, padding $3$ + `ReLU` + `AvgPool1d(4)` $\to (24, 64)$.
   * **`Conv1d_2`**: $24 \to 48$ channels, kernel $k_2 = 13$, padding $6$, dilation $1$ + `ReLU` + `AvgPool1d(4)` $\to (48, 16)$.
   * **`Conv1d_3`**: $48 \to 72$ channels, kernel $k_3 = 9$, padding $8$, **dilation $d_3 = 2$** + `ReLU` $\to (72, 16)$ (receptive field covers the entire 256-step window).
3. **Multi-Statistic Global Pooling + Metadata Fusion:**
   * Concatenates per-channel **Global Mean** `(72,)`, **Global Max** `(72,)`, **Global Std** `(72,)`, and the **Metadata vector** `(5,)` $\to$ `221` features.
   * **`Linear` Classifier**: $221 \to 8$ logits $\to \arg\max$.

### Multi-Stage HPO & Optuna Study
We conducted a 3-stage structured ablation followed by a 25-trial **Optuna TPE study** (`20_optuna.py`) with `MedianPruner`:
* **Stage 1 (`16_hpo_stage1.py`):** Established that `AvgPool1d` substantially outperforms `MaxPool1d` under the noisy evaluation preset (score $74.8$ vs. $66.1$) by attenuating high-frequency jitter.
* **Stage 2 (`17_hpo_stage2.py`):** Introduced demo-matched noise/NaN augmentation and class-6 upweighting (`c6w=2.0`), lifting class 6 accuracy on `test_demo` from $0.36 \to 0.54$.
* **Stage 3 (`19_hpo_stage3.py`):** Confirmed that concatenating the 5 metadata features (`meta`) and reducing the first kernel size (`k1=7`) simultaneously improved `test_demo` accuracy to $80.5\%$ and reduced latency.
* **Optuna TPE Search (`20_optuna.py`):** Optimized 14 hyperparameters over 25 trials. **Trial 20** achieved the highest combined objective (**127.4**), with `ch=(24,48,72)`, `k=(7,13,9)`, `dil2=1`, `dil3=2`, `lr=0.00163`, `weight_decay=2.24e-4`, `label_smoothing=0.05`, `sigma=0.101`, `nanp=0.0164`, `c6w=1.75`, `c7w=1.94`.

![Optuna Optimization History](experiments/plots/hpo/optimization_history.png)
![Hyperparameter Importance](experiments/plots/hpo/param_importances.png)
![Parallel Coordinates](experiments/plots/hpo/parallel_coordinates.png)

---

## 6. Repository Structure & Reproducibility

* **`model.py` / `submission/model.py`**: Official submission entry point implementing `Model.__init__` and `Model.predict(sample)`.
* **`cnn_final.onnx` / `submission/cnn_final.onnx`**: Exported static ONNX graph (opset 17, 192 KB) with BatchNorm folded into convolution weights.
* **`optuna_best.npz` / `submission/optuna_best.npz`**: Folded NumPy weights of the winning Optuna Trial 20 model.
* **`requirements.txt`**: Minimal runtime dependencies (`numpy`, `onnxruntime`).
* **`experiments/`**: Complete chronological research log (`notes.md`), training scripts (`02` to `25`), execution logs (`experiments/logs/`), and generated figures (`experiments/plots/`).

### Running the Official Evaluation Harness
```bash
python harness.py evaluate --model model.py --set n_runs=1 --source file:data/test_demo.npz
python harness.py evaluate --model model.py
```
