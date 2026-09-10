# CO5420 HAR Project — Viva Preparation Summary

Final leaderboard score: **0.96879** (Diverse Architecture Ensemble)

---

## 1. Data Understanding & EDA

- Loaded `train.csv`: 7352 rows × 563 columns (561 hand-crafted features + `subject` + `Activity`)
- Verified: no missing values, no duplicate rows
- Confirmed all 561 features are bounded in [-1, 1] **after** catching a bug where `subject` (values 1–30) was accidentally included as a "feature" — inflated PCA variance and feature range (max was 30.0 instead of 1.0) until fixed
- Class distribution: mild imbalance (986–1407 rows per class), WALKING_DOWNSTAIRS smallest
- 21 unique subjects in training data

## 2. Data Leakage Fix (subject-based splitting) — foundational decision

- **Problem**: naive random train/val split lets the same subject's rows appear in both train and validation. Since rows from one person are highly correlated (same gait, same device placement), the model can partly "recognize the person" rather than "recognize the activity" — inflating validation scores unrealistically.
- **Fix**: `GroupShuffleSplit` / `GroupKFold`, grouping by `subject`, so no person appears in both train and validation.
- **Verified explicitly**: asserted `train_subjects ∩ val_subjects = ∅` after every split.
- This mirrors the original UCI HAR dataset design (21 train subjects / 9 test subjects, subject-disjoint).
- **Later confirmed via literature critique**: found this exact leakage bug (unguarded random `train_test_split`) in a GeeksforGeeks HAR tutorial, and a probable version of it in a published paper (DCAM-Net, Xu et al. 2025) whose 5-fold CV description didn't confirm subject-disjoint folds — used as evidence this is a real, common pitfall, not an academic technicality.

## 3. PCA Investigation (tried, then deliberately dropped)

- 2 components explained 67.5% of variance (after fixing the subject-leakage-into-features bug); 65 components needed for 95% variance.
- **Compared PCA vs raw features on both Decision Tree and SVM**: raw won clearly for DT (0.87 vs 0.82) — trees do axis-aligned splits, which don't benefit from PCA's rotated feature combinations. Raw also won for SVM (0.96 vs 0.95), by a smaller margin.
- **Decision**: PCA dropped from the pipeline; documented as an investigated-and-rejected alternative.

## 4. Baseline Models

### Decision Tree
- `GridSearchCV` over `max_depth`, `min_samples_leaf`, `criterion`, using `GroupKFold` (5-fold, grouped by subject) for hyperparameter selection.
- Best: `criterion=entropy, max_depth=10, min_samples_leaf=5` → val Macro F1 **0.8697**

### SVM
- Initially found unscaled+rbf+C=1 outperformed a scaled version in a quick probe — **caught a methodology flaw**: had pre-decided scaling before the real hyperparameter search, which could rule out the true best combination. Fixed by folding "scaler or no scaler" into the `Pipeline`/`GridSearchCV` search itself, alongside kernel and C.
- Best: unscaled, `kernel=rbf, C=10, gamma=scale` → val Macro F1 **0.9581**

### Feedforward Neural Network (PyTorch)
- Staged hyperparameter investigation, each stage building on the previous winner, all using **multi-seed averaging** (not single runs) because NN results vary run-to-run due to random weight init and minibatch order — a key concept distinguishing NN from deterministic DT/SVM.
- **Depth**: swept 1–4 hidden layers → winner `[256, 128]`
- **Regularization**: dropout {0, 0.3, 0.5} × weight_decay {0, 1e-4} → winner dropout=0.5, weight_decay=0.0
- **Normalization**: BatchNorm on/off → winner: **True** (biggest single jump in the search)
- **Learning rate, batch size, optimizer**: further sweeps → confirmed Adam, lr=1e-3, batch_size=64 as strong defaults
- Val Macro F1: **0.9714 ± 0.0033** (5-seed mean)
- Added **class-weighted loss** (inverse frequency, targeting WALKING_DOWNSTAIRS) + **LR scheduling** (`ReduceLROnPlateau`) → 0.9746 ± 0.0020 (tighter std = more stable)

## 5. Model Comparison (DT vs SVM vs NN)

| Model | Val Macro F1 |
|---|---|
| Decision Tree | 0.8697 |
| SVM | 0.9581 |
| Feedforward NN | 0.9765 (best seed) |

Per-class F1 comparison showed NN winning or tying on **every class**; WALKING_DOWNSTAIRS recall improved most dramatically across models (DT 0.72 → SVM 0.81 → NN 0.93+), the previously hardest class.

## 6. Discovering the Real Ceiling: Subject Variance

- Ran **5-fold GroupKFold** (different subject compositions) on the confirmed NN config.
- Result: scores ranged **0.8856 to 0.9880** depending purely on which subjects were held out. Mean **0.9386 ± 0.0426** — much wider spread than the single 80/20 split (0.9746) suggested.
- **Key insight**: the original single split had landed on a favorable subject group. The honest generalization estimate is the CV mean, not the lucky single-split number.
- This directly predicted the gap later seen on the real leaderboard (first real submission: 0.95039, close to the CV mean, well below the optimistic single-split number).

## 7. Deep Error Analysis — the Subject 14 Case Study

- Reproduced the worst-performing fold (Macro F1 0.8856, subjects 14/15/19/25 held out).
- **Per-subject breakdown revealed subject 14 alone caused 90 of ~152 errors** (59%) — almost entirely WALKING and WALKING_DOWNSTAIRS misclassified as WALKING_UPSTAIRS. Other 3 held-out subjects scored 92–98%.
- **Interpretation**: one atypical individual's gait/movement pattern, globally normalized against the population average, resembled someone else's "upstairs" pattern. Not a general "unseen subjects are hard" problem — a specific person-level bias problem.

## 8. Per-Subject Normalization — the biggest breakthrough

- **Idea**: normalize each row using *that subject's own* mean/std (computed from their own rows only), instead of a single global `StandardScaler` fit across all training subjects. No labels needed — works identically on `test.csv` since it has a `subject` column.
- **Directly tested against the Subject 14 failure case**: errors dropped from 90 → 11 (88% reduction); fold Macro F1 jumped 0.8856 → 0.9806.
- **Validated across all 5 GroupKFold splits** (not just the diagnosed fold) to rule out overfitting the fix to one case: **5/5 folds improved**, mean 0.9386 → 0.9731, std tightened 0.0426 → 0.0176.
- **Real leaderboard confirmation**: 0.95039 → 0.96270 when combined with ensembling.

## 9. Ensembling (5-seed) — second confirmed real win

- Averaging softmax probabilities across 5 independently-seeded models trained on all data.
- Real leaderboard improvement: 0.95039 → 0.95701.
- **Tested and ruled out**: 10 seeds vs 5 (negligible gain, within noise — diminishing returns confirmed).

## 10. Extended Task 3 — RNN (LSTM/GRU) on Raw Signals

- Downloaded raw UCI HAR `Inertial Signals` (9 channels × 128 timesteps), since Kaggle's `train.csv` only had the 561 pre-computed features.
- **Verified alignment** between raw signals and `train.csv` before trusting it: subject IDs and activity labels matched row-for-row exactly (also cross-checked a computed statistic against a corresponding hand-crafted feature).
- Built configurable `RNNClassifier` (LSTM/GRU), staged investigation of architecture (GRU, hidden=64 won), dropout.
- Single-split result: **0.9930** — looked like it beat the NN.
- **Sanity-checked properly**: multi-seed (0.9860 ± 0.0099) and GroupKFold (**0.9343 ± 0.0352**) — the GroupKFold mean was essentially tied with the NN's own GroupKFold mean (0.9386), revealing the single-split 0.9930 was a favorable-draw artifact, not a real advantage.
- **Real submission test**: scored 0.88876 — first verified it wasn't a data-alignment bug (checked correlation between raw-signal stats and test.csv features = 1.0000, confirming alignment was correct) — confirmed it was a genuinely weaker model on that real subject draw.
- **Conclusion**: end-to-end learning from raw signals does **not** offer a meaningful advantage over hand-crafted features here — the domain-expert-engineered 561 features already capture nearly all extractable signal.

## 11. Extended Task 4 — Sensor Contribution (Accelerometer vs Gyroscope)

- Split 561 features by name into accelerometer-derived (345), gyroscope-derived (213), and "other" (3 angle features, excluded from both since gravity itself is accelerometer-derived, making the split imperfect — noted as an honest limitation).
- Accelerometer-only: 0.9478 (drop of only 0.0104 from full)
- Gyroscope-only: 0.8143 (drop of 0.1438)
- **Per-class pattern**: gyroscope-only collapsed specifically on static postures (LAYING −0.325, SITTING/STANDING ~−0.18) but held up on dynamic activities — physically explainable, since a gyroscope measures rotation rate and there's little rotation while stationary.
- **Conclusion**: sensors are complementary, not redundant — accelerometer dominates static-posture discrimination, gyroscope contributes specifically to dynamic/rotational motion.

## 12. Failed / Negative Experiments (all honestly tested and ruled out)

- **NN+SVM weighted blend**: failed — SVM dominated by NN on every class already (per-class comparison showed no complementary strength).
- **GroupKFold-diverse ensemble** (5 models, each missing different subjects): real leaderboard score 0.95236, below the seed-ensemble's 0.95701.
- **NN+RNN blend**: looked like a real 5/5-fold win locally (+0.0042) — but this used alpha tuned on the same fold being scored. Re-tested with an honestly fixed alpha (tuned on a separate split, applied everywhere): **collapsed to 2/5 folds, −0.0008 mean** — confirmed the original result was alpha overfitting, not a real effect. Important methodological lesson repeated from earlier (SVM scaler pre-decision mistake).
- **Feature concatenation** (global + per-subject normalized, 1122 features): hurt performance (4/5 folds lost) — redundant information, not new signal.
- **Outlier/transition-window removal**: marginal, inconsistent (+0.0023, below the "meaningful" threshold, mixed fold-by-fold results).
- **XGBoost**: 0.9575 GroupKFold mean, below NN — tabular gradient boosting didn't beat the NN on these already-redundant, highly-correlated features.
- **1D CNN** (plain): 0.9623 — closer than RNN but still below NN.
- **Multi-scale CNN + attention** (inspired by DCAM-Net paper): 0.9612 — no improvement over the plain CNN; confirms added architectural complexity isn't finding extra signal in this feature space.
- **RNN + per-subject normalization**: helped the RNN (0.9343 → 0.9499) but still didn't beat the NN.
- **Deliberately re-introducing subject leakage**: tested as a direct experiment — did not produce a better real-world model (the final model already trains on 100% of subjects regardless; leakage only affects validation honesty, not final training data).

## 13. AdaBN — Test-Time Domain Adaptation

- From literature (Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation," 2016): recalibrate each model's BatchNorm running statistics to each **test subject's own data** (no labels needed) before predicting — adapts the model's internal representations, not just the input.
- Validated via GroupKFold: **4/5 folds won**, mean +0.0024, tighter std.
- Combined with per-subject norm + 5-seed ensemble → real leaderboard: 0.96270 → **0.96566**.

## 14. Label Smoothing

- Motivated by the dataset's known "transition window" issue: sliding windows with 50% overlap sometimes straddle an activity transition, producing genuinely ambiguous hard labels.
- Swept values [0.0, 0.02, 0.05, 0.1, 0.15, 0.2] via GroupKFold: **0.05 confirmed as the (near-)optimal value**, not just a guess — 0.9731 → 0.9763 mean, winning/tying on 4/5 folds.
- Real leaderboard confirmation: 0.96566 → 0.96813 (same architecture, only change).

## 15. Final Model: Diverse Architecture Ensemble — current best (0.96879)

- Combines: per-subject normalization + label smoothing (0.05) + **4 different MLP architectures** ([256,128], [512,128], [256,256,64], [512,256,128]) × 3 seeds each = 12 models, weighted average of softmax probabilities (weights: 60%/15%/15%/10%, favoring the most-validated architecture).
- **Why architecture diversity over just more seeds**: different architectures make more genuinely different mistakes than the same architecture with different random initialization — confirmed by this being the best result of the whole project.

## 16. Submission Pipeline Methodology

- **Two-phase pattern** used throughout: Phase 1 uses the subject-grouped 80/20 split with early stopping to find the optimal epoch count; Phase 2 retrains fresh models on **100% of `train.csv`** for that fixed epoch count (no more held-out validation needed once hyperparameters are locked in — maximizes data for the model that actually ships).
- Verified `sample_submission.csv` format compliance (`id`, `Activity` columns, correct row count, valid labels only) with explicit assertions before every submission.
- Distinguished "interim/mock submissions" (using `X_val` as a stand-in before `test.csv` was released) from real submissions.

---

## Key Concepts to Be Ready to Explain

- **Macro F1** vs accuracy (per-class average, treats every class equally regardless of size)
- **Data leakage** via subject correlation, and why `GroupShuffleSplit`/`GroupKFold` fixes it
- **PCA** mechanics and why it helps some models (SVM) but not others (trees, due to axis-aligned splitting)
- **Overfitting** — visible in the DT max_depth sweep (train F1 → 1.0 while val F1 peaks then declines)
- **NN run-to-run variance** — why multi-seed averaging is necessary before trusting any single result
- **Cross-validation vs single split** — why GroupKFold mean is the honest number, not a favorable single split
- **Per-subject normalization** — the mechanism (removing person-specific bias) and the concrete evidence (subject 14 case study)
- **Ensembling** — why averaging probabilities across models/seeds reduces variance
- **AdaBN** — adapting model internals (not just input data) to each new subject at test time
- **Label smoothing** — softening hard targets to account for genuine label ambiguity
- **The "tune and apply to the same data" trap** — caught twice (SVM scaler pre-decision; NN+RNN blend alpha) and fixed both times with proper held-out validation