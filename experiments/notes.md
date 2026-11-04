# Experiment notes

## Contest mechanics (from harness.py)

- Score/run: geomean over 8 classes of `T_k = max(eps, n_k^α · (c_k − λ·w_k)/n_k · (c_k/n_k)^p)`,
  α=0.5, p=2, λ=0.33, eps=0.01. Balanced classes → `T_k = √(n/8) · (acc − 0.33·err) · acc²`.
- n = 60 s budget / per-call latency (official source: hidden generator, infinite stream).
- Worker: fresh process, **single-thread** (OMP/MKL/OPENBLAS/NUMEXPR=1), per-sample
  soft timeout 0.5 s (+0.25 s grace = kill), warmup ≤100 samples/5 s unscored,
  setup ≤120 s untimed, no internet, cwd = model dir (also on sys.path).
- Abstain counts in n_k but neither correct nor wrong; answer beats abstain while
  p_max ≳ 0.25–0.30 (λ=0.33).
- Eval machine: Google Colab ("same conditions"), CPU assumed.

## Data

- train_100k: 100k × (4,256) float32, ~1.4 % NaN, labels 0..7 (synthetic generator, organizer-only),
  class counts slightly imbalanced (15993 … 9865).
- test_demo: 400 samples, exactly balanced, "drawn like the evaluation data".
- meta per sample: sample_rate ∈ {1.0, 2.0}, group ∈ {A,B,C}, level ~ N(0,2).
  Meta shifts signal stats (group A rougher, sr=2 smoother) but is NOT class-determined.
  Baseline's feature vector drops string meta (group) → one-hot A/B/C is a free feature.
- Class fingerprints (2k-sample subsample): class 2 = smooth/low-freq dominated (acf8 0.24,
  lowf 39 %), class 1 = channels uncorrelated (xcorr 0.15), 6/7 = more inertia + cross-corr,
  0/3/4/5 = noisy, weakly separated → need interaction features or learned features.

## Measured results (20k val split, train 80k)

| model | val acc | latency/call | n(60 s) | score proj |
|---|---|---|---|---|
| single tree d14 (101 feats) | 0.638 | 719 µs (features 554!) | ~83 k | ~21 |
| tree d20 | 0.635 | ~720 µs | ~83 k | ~21 |
| GBM 300 (= baseline, same feats) | 0.820 | 31 364 µs | ~1.9 k | ~8 |
| sklearn MLP (256,128) raw window, 40 iters | 0.483 | (killed; sklearn overhead unmeasured) | — | — |

- Feature extraction (baseline 101-dim vector) costs ~554 µs/sample → dominates every
  feature-based model. Tree traversal itself only ~166 µs.
- Trees plateau at ~64 % regardless of depth (d28 overfits: 0.614).
- Key structural insight: hand features are the shared bottleneck; a raw-window net
  replaces 554 µs of Python/numpy stats with one matmul inside the model.

## Ensemble math (why member-ensembles need care)

score ∝ √(1/lat) · acc³ (roughly). k members → lat ×k, need acc gain g with
**g³ > √k**: k=2 needs +12 % rel. acc, k=3 needs +20 %. Typical MLP seed ensembles
give +2–4 pp absolute → NOT worth it for k≥2 unless members are nearly free.
The GBM is the cautionary tale: 300 members = 31 ms. Ensembles that CAN pay:
shared-backbone multi-head (costs ~nothing), or averaging nets so small that total
latency stays ≪ tree's 719 µs.

## v3 (03_latency_fix, DONE 13:10) — final latency table

Fixed v2's baseline refit bug (tree/GBM had been fitted on raw 1024 dims, fed 101-dim
features → crash in latency bench; their v2 accs 21.3/71.0 were meaningless).
Refit on features, same split; reloaded v2 MLPs; measured everything single-thread:

| path | latency | n(60 s) | acc | score proj |
|---|---|---|---|---|
| prep_one (fill NaN + z-score) | 75 µs | — | — | — |
| MLP numpy forward (incl prep) | 110 µs | 546 k | 0.487 | 19.7 |
| ensemble(3) numpy | 181 µs | 331 k | 0.528 | 21.1 |
| MLP sklearn predict | 357 µs | 168 k | 0.487 | ~16 |
| tree d14 + baseline feats | 676 µs | 89 k | 0.638 | 22.2 |
| GBM 300 + baseline feats | 18.5 ms | 3.2 k | 0.820 | 10.3 |

- sklearn predict path is 3.2× slower than raw numpy forward (validation overhead) —
  submission uses the numpy path directly.
- Everything lands in a 20±2 score band: acc³ vs √n cancel out. To break out we need
  BOTH fast AND accurate: at 181 µs + tree-level acc (0.64) the ensemble scores ~43;
  at ≤500 µs + 0.8 it's ~57. The latency budget is huge — the MLP accuracy is the gap.
- v2 MLP postmortem: (256,128) adam converged at only 48.7–48.9 %; per-class 0.95/0.88
  on classes 1/2 but 0.26–0.52 on 0/3/4/5/7 — exactly the weakly-separated classes
  from the fingerprint. Raw-window MLP underfits the temporal structure.

## v4 (04_ensemble_prep, DONE 13:24) — tree ensembles + preprocessing, measured

Nils' questions: (a) Random Forest statt single tree? (b) preprocessing upgrades?

Answers, measured (same 20k split, latency single-thread harness-style):

| model | feats | val acc | worst cls | latency | score proj |
|---|---|---|---|---|---|
| tree d14 | base | 0.638 | 0.411 | 681 µs | 22.1 |
| tree d14 | upg | 0.635 | 0.413 | 739 µs | 20.9 |
| RF 100×d14 | base | 0.686 | 0.453 | 6.52 ms | 9.3 |
| RF 100×d14 | upg | 0.692 | 0.457 | 6.51 ms | 9.6 |
| ET 100×d16 | base/upg | 0.672/0.688 | 0.449+ | 6.59 ms | 8.6/9.3 |
| GBM 300 | base | 0.820 | 0.558 | 19.8 ms | 10.0 |
| GBM 300 | upg | 0.821 | 0.552 | 20.1 ms | 9.9 |

- **(a) Tree ensembles lose, structurally.** k-tree ensembles pay √k in the score
  (latency ×k → √n × 1/√k) and gain only acc³. RF(100): +4.8 pp acc = 1.28× on
  acc³, but √100 = 10× latency cost → 22 → 9.3. Ensemble rule: needs g³ > √k;
  for k=100 that's 10× acc — impossible. Only "nearly-free" members work
  (numpy MLP ensemble: 3 nets = 110→181 µs, exactly break-even).
- **(b) Rate-aware preprocessing is a dead end.** Upgraded features (ACF in
  seconds, spectrum on Hz grid, group one-hot, lagged cross-corr, per-channel
  NaN): Δacc ≤ +0.6 pp everywhere. Diagnostic: fdom(sr=2)/fdom(sr=1) = 0.99–1.26
  in Hz → generator processes are ~rate-invariant, features were already aligned.
- Class structure (MLP ensemble per-class acc): 1→0.95, 2→0.88 trivial;
  4/5→0.26 nearly interchangeable; GBM worst class 0.558. The signal is in
  temporal SHAPE, not in better global stats.
- Abstain: NEVER abstain a class wholesale — abstained class has c_k=0 →
  T_k → eps → geomean destroyed. Always answer; per-sample abstain only if
  p_correct < ~25 % (λ=0.33 break-even), and it still dilutes (c/n)².

## v5 (05_cnn + 06_cnn_latency, DONE 14:28) — 1D-CNN WINS

Architecture: conv(4→16,k9)+BN+ReLU+pool4 → conv(16→32,k9)+BN+ReLU+pool4 →
conv(32→64,k5)+BN+ReLU → global avg+max pool → fc(128→8). ~17k params.
Training: Adam lr=1e-3, batch 256, class weights, NaN-dropout + noise aug, 30 ep (7-8 s/ep).
Export: BN folded into conv; numpy im2col forward (max diff vs torch 4.3e-06).

**val_acc 0.890 | latency 465 µs single-thread | n(60 s) ≈ 129 k | geomean score ≈ 81**
(vs tree d14 ≈ 22, GBM ≈ 10 → CNN is the submission backbone, ×3.7 over tree)

Per-class: 1→0.998, 2→0.956, 4→0.949, 5→0.945, 3→0.899, 0→0.823, 6→0.750, 7→0.734.
- **4/5 are NOT degenerate** (0.95/0.95!) — the raw-window MLP's 0.26 was an architecture
  failure, convs fix it. My earlier worry was wrong, in the best way.
- The actual weak pair is **6↔7**: 369 (6→7) + 255 (7→6) confusions; plus 0↔3
  (226 + 157). These three pairs are where the score leaks.

Next (v6): (1) submission `model.py` (numpy path) + **official `harness.py evaluate`**
(the real proof — still pending!); (2) 6/7 targeted: circular-shift augmentation
(only ~64-step invariance from two 4× pools), wider conv within latency budget
(we can afford ~800 µs before score drops below current 81: sqrt cost, needs
+10% per-class acc on 6/7); (3) NO CNN ensemble: 3× latency → √3 score loss,
needs +19% acc — math says no (g³ > √k rule).

## v6 (08_cnn_v6, DONE 15:00) — targeted fixes: +acc overall, 6→7 got worse

Changes: conv3 k=9 dil=2 (RF ~272 steps), global mean+max+**std** pooling (192 features),
circular-shift aug, cosine LR, 40 epochs, 25.5k params. Export verified (5.3e-06).

**val_acc 0.9018 (v5: 0.8899) @ 496 µs → geomean score 82.4 (v5: 81.3, tree ~22, GBM ~10)**

| class | v5 acc | v6 acc | note |
|---|---|---|---|
| 0 | 0.823 | 0.842 | 0→3: 226→189 ✓ |
| 1 | 0.998 | 0.999 | |
| 2 | 0.956 | 0.967 | |
| 3 | 0.899 | 0.895 | |
| 4 | 0.949 | 0.972 | |
| 5 | 0.945 | 0.973 | |
| 6 | 0.750 | **0.732** | **6→7: 369→415 ✗** |
| 7 | 0.734 | 0.770 | 7→6: 255→190 ✓ |

**Key insight from error analysis:** sr=2 is overrepresented in BOTH 6/7 error
directions (6→7: 58% sr=2 vs 47% among correct 6s; 7→6: 62% vs 49%). NaN uncorrelated.
→ the 6/7 boundary is **rate-conditional**: the generator separates 6 vs 7 via a
sample_rate-dependent detail. v7 plan: concat sample_rate (+group one-hot, level)
to the pooled features (FC 192→196+8) so the net can learn rate-conditional
boundaries. Cheap, zero latency cost.

## v6.5 (10_latency_profile, DONE 15:25) — Profiler + ONNX-Bombshell

Phasenprofil der v6-numpy-Pipeline (400 Demo-Samples, single-thread):

| Phase | µs |
|---|---:|
| prep (NaN-interp + z-score) | 62 |
| conv1 (pad+view+matmul) | 88.5 |
| pool1 | 16.8 |
| conv2 | 79.2 |
| pool2 | 10.3 |
| conv3 (dil=2) | 84.4 |
| global pool (mean+max+std) | 46.7 |
| fc + argmax | 9.2 |
| **Gesamt** | **~397-496** |

- numpy-Conv ist OVERHEAD-dominiert (pad/copies/reshape + viele kleine Ops),
  nicht FLOP-dominiert: ~600k MACs in 250 µs = 2.4 GMAC/s; ORT schafft 12 GMAC/s.
- **ONNX Runtime (intra_op=1): 50 µs für dieselbe Conv+Pool+GlobalPool** = 6.7× schneller.
- **Konsequenz: Projektil ~130 µs/call (prep 62 + ORT 50 + fc/overhead)**
  → n(60s) ~ 460k → score-proj **~160** (v6 numpy: 82.4). Messung war mit
  Random-Weights (timing-repräsentativ), echter Export = v7.
- Params ≠ Latenz (gemessen): v5 17k @ 465 µs, v6 25k @ 496 µs (numpy),
  v6 25k @ 50 µs (ORT). Kernel > Größe.
- Unklar: Colab-CPU-Speed (±2×), Generator-Stream-Rate cappt n evtl. früher
  (n_producers=2) — √n sättigt dann, schneller schadet aber nie.

## v7 (13/14/15, DONE 15:38) — ONNX real + BUGFIX + offizieller Harness 🎯

**Der Bug, den nur der offizielle Harness findet:** erster harness-Lauf: demo acc 0.295 (!),
train 0.394 — während alle meine Bänke 0.90 sagten. Debug (15_debug_harness.py):
ONNX-Graph exakt identisch (0/400 Abweichungen via prep_batch), ABER model.py _prep
verwendete np.interp (linear), Training aber ffill/bfill → prep max|diff| 16.1 →
Logits-Diff 19 → Modell bricht auf eval-like-Daten zusammen. **Fill-Methode ist Teil
der Trainingsdistribution.** Fix: ffill/bfill in model.py (vektorisiert) →
re-verify: prep diff 0.000, 0/400 Abweichungen, val 0.906. 

**Offizieller harness.py evaluate (3 runs, file:train_100k):**

| metric | value |
|---|---|
| score (mean) | **78.16** |
| acc | 0.9112 (inkl. 80k eigene Train-Samples! val = 0.906) |
| latency | 0.18 ms/call |
| n | 100.000 (stream_exhausted — Datei leer vor Budget!) |

- test_demo: score 2.663 (n=400 → √n_k skaliert klein, nicht vergleichbar).
- **WICHTIG: Verteilungs-Shift entdeckt** — demo ("drawn like the evaluation data"):
  acc **0.790** vs train-val 0.906. Eval nutzt preset "test" ≠ train-Preset.
  Ehrliche Eval-Acc-Erwartung: ~0.79, nicht 0.90!
- Projektion fürs offizielle Eval: n = 60s/0.18ms ≈ 333k (falls Stream mithält)
  → ×1.83 über den 78.16 (n=100k) → aber mit Demo-Acc 0.79 statt 0.91:
  grob Score ~70-95 statt 140. Stream-Rate (2 Producer) = unbekannter Cap.

## v8 (16_hpo_stage1, RUNNING since 16:01) — shift diagnosis + HPO stage 1

**Stage A — shift diagnosis (DONE).** Demo (eval preset) vs train preset, per class:

- acf1 systematically LOWER in demo (0.73→0.65 avg): eval windows are noisier/less smooth.
- roughness HIGHER (0.53→0.58), low-freq share LOWER, xcorr LOWER: same direction.
- NaN fraction 2x (0.0073 → 0.0138), level std higher (1.80 → 2.24).
- The class ORDER is preserved (class 2 slowest, 6/7 inertial both presets) → same
  processes, different noise/NaN/level regime. Train val is NOT representative;
  the honest objective is demo acc (n=50/class → noisy, need >2-3pp deltas to trust).

**Implication for training drama (Nils' idea):** augment toward the demo regime —
NaN p=0.014 (matches), noise σ↑ toward 0.09-0.12, level/scale jitter, smoothness
variation. Stage 2 candidate: "demo-matched augmentation" probe.

**Stage B — HPO stage 1 (RUNNING):** 10 one-lever probes, 20 epochs each
(~3 min each, ~35 min total): baseline / k1=15 / k3=15 / wide 24-48-96 / narrow
12-24-48 / avg-pool / gelu / leaky-0.1 / gp=mean+max only / gp+quantiles.
Objective = official score formula on DEMO per-class acc × estimated ORT latency
(calibrated: ORT ≈ 0.24 × numpy fwd + 61 µs prep). Per-probe output: val_acc,
demo_acc, numpy µs, ORT µs, n(60 s), score. Winners → stage 2 combinations.

## v8.5 (17_hpo_stage2, DONE 16:58) — stage 2 winner combos

All probes avg_pool, 20-40 epochs, objective identical (demo per-class acc x calib. ORT lat):

| probe | val | demo | score | note |
|---|---|---|---|---|
| avg40 | 0.908 | 0.782 | 81.1 | longer training wins |
| avg_aug | 0.893 | 0.772 | 78.3 | aug alone: mixed |
| avg_aug40 | 0.906 | 0.782 | 76.9 | |
| avg_wide24 | 0.907 | 0.770 | 71.1 | wide loses again |
| **avg_aug40_c6w2** | 0.906 | 0.775 | **81.5** | **class-6 weight x2: 6: 0.36->0.54, 7: 0.68->0.54** |

- **Winner: avg_aug40_c6w2** — 40 epochs + sigma 0.10 + NaN p0.02 + class-6 x2.
  The c6 weight x2 trades class 7 down for class 6 up — geomean-optimal on demo
  (both now 0.54 instead of 0.36/0.66 split).
- Stage-2 gain: 74.8 -> 81.5 (+6.7). Class 6 remains THE bottleneck (0.54 demo).
- Weights saved: models/hpo2_avg_aug40_c6w2.npz (BN folded, numpy forward).
- Caveat: scores use calibrated ORT estimate (0.24x numpy + 61us), demo n=50/class noisy.
  Final proof = real ONNX export + official harness (started 17:20).

## v9 (18_final_export, DONE 17:22) — final export + official harness 🎯

Stage-2 winner exported (cnn_v8.onnx, avg_pool, 101 KB): torch-vs-numpy 2.9e-06,
ORT-vs-numpy 0/5000 deviations, ORT val_acc (5k) 0.9102, prep+ORT 151 µs/call.
model.py points to cnn_v8.onnx (ffill/bfill prep).

**Official harness.py evaluate results:**

| run | v6 (max_pool) | v8 (avg_pool c6w2) |
|---|---|---|
| train_100k stream (3 runs) | 78.16 | **78.71** |
| acc | 0.9112 | 0.9124 |
| test_demo (1 run, n=400) | 2.663 | 2.649 (tie, noisy) |
| demo acc | 0.790 | 0.775 |
| latency | 0.18 ms | 0.18 ms |

Honest reading:
- v8 wins the train stream (+0.55); demo is a statistical tie at n=400.
- My stage-2 projection (81.5) was optimistic: demo per-class n=50 is noisy, and the
  FILE source caps n at 100k (stream_exhausted) → latency advantage doesn't show.
- In the OFFICIAL generator run (infinite stream), n → 60s/0.18ms ≈ 333k → √n factor
  1.83x over the file-source score → realistic expectation ~80-110 depending on
  eval-preset accuracy (demo-like 0.775) and generator production rate.
- v8 is the submission (better balanced classes, same latency). Class 6 remains the
  bottleneck (demo 0.54 after c6w2).

## v10 (19_hpo_stage3, DONE 18:52) — stage 3 complete

All on stage-2 winner base (avg_pool, meta off unless noted), objective = 0.5*demo + 0.5*val:

| probe | val_acc | demo_acc | score_demo | score_val | pc6/pc7 (demo) |
|---|---|---|---|---|---|
| **k1_7** | 0.902 | **0.805** | **97.9** | **157.1** | 0.58/0.64 |
| ep60 | 0.907 | 0.797 | 92.8 | 157.0 | 0.56/0.58 |
| meta | 0.905 | 0.782 | 87.3 | 152.8 | 0.58/0.56 |
| augheavy | 0.903 | 0.785 | 87.0 | 154.5 | 0.54/0.60 |
| c67w2 | 0.902 | 0.782 | 81.5 | 155.4 | 0.40/0.70 |
| dil3_4 | 0.901 | 0.767 | 78.6 | 149.9 | 0.50/0.52 |

- **k1=7 is the surprise winner** (smaller first kernel): best demo acc AND cheaper
  latency (386 vs 403 µs numpy). Combined 127.5.
- ep60 confirms: longer training keeps helping.
- meta confirmed (+6 over stage-2 baseline 81.5).
- c67w2 backfired on demo: pushes 7 to 0.70 but crashes 6 to 0.40 — geomean says no.
- dil3_4 loses; dil3=2 stays.
- Caveat: demo n=50/class → per-class numbers bounce ±7pp; rankings validated
  against val (n=250/class) which agrees on k1_7/ep60/meta.

## v11 (20_optuna, RUNNING since 18:52) — TPE search on 10-dim space

Search space: ch (6 triples) × k1 (7/9/11) × k3 (9/11/13) × dil3 (2/3/4) × epochs
(25-60) × lr (log 5e-4..2e-3) × sigma (0.05..0.18) × nanp (0.008..0.03) × c6w
(1..3) × c7w (0.7..2.5). Fixed: avg_pool, meta features, mean+max+std gp, ReLU.
MedianPruner (warmup 8 ep), 40 trials, sqlite storage (resumable), every trial
saves weights. Objective = 0.5*demo + 0.5*val (official formula, calibrated latency).
ETA ~22:00 (pruned trials ~2 min, full ~6 min).

## Open questions

1. Does MLP@300it converge to ≥0.8? If not: more iters, wider (512,256), or 1D-CNN.
2. numpy-forward latency target ≤200 µs (incl. prep) → n ≥ 300k → score ≥ 100.
3. Abstain threshold sweep on demo set (p_max ~0.3) with real harness.
4. sample_rate/group as input features for the net (cheap, one-hot + scalar).
5. Per-class calibration: geomean punishes one weak class; check per-class acc of ensemble.
6. Final submission repo: numpy-only forward (no sklearn at eval → lean requirements.txt).