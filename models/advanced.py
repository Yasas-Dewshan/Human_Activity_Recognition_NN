# %% [markdown]
# # HAR Competition — Advanced Robust Pipeline
#
# Builds on the validated baseline (per-subject norm + 5-seed ensemble + AdaBN,
# ~0.957 leaderboard) and adds:
#
# 1. GroupKFold-validated epoch count + AdaBN momentum (replaces single noisy
#    80/20 split — more stable hyperparameter choice).
# 2. Multi-architecture NN ensemble (varying width/depth, not just seeds) +
#    snapshot ensembling via cosine-annealing warm restarts (more diversity
#    per unit of compute).
# 3. A gradient-boosted model (LightGBM, falls back to XGBoost, falls back to
#    sklearn GradientBoosting) trained on the same per-subject-normalized
#    features — trees make different errors than NNs on tabular sensor data.
# 4. Label smoothing + Gaussian feature jitter augmentation for the NNs.
# 5. Tunable AdaBN momentum (swept via CV instead of hardcoded 1.0).
# 6. Out-of-fold stacking: a small logistic-regression meta-learner learns how
#    to combine NN-ensemble probs + GBM probs, instead of a flat average.
#
# Everything before "Real submission" is offline validation/training. Nothing
# here touches test.csv until the final prediction cell.

# %%
import numpy as np
import pandas as pd
import pickle
import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"

# Multiple architectures = real diversity (vs. same-arch-different-seed).
ARCHS = [
    {"hidden_sizes": [256, 128], "dropout": 0.5},
    {"hidden_sizes": [128, 64],  "dropout": 0.4},
    {"hidden_sizes": [256, 128, 64], "dropout": 0.5},
]
BASE_NN_CONFIG = {"batchnorm": True, "lr": 1e-3, "batch_size": 64, "weight_decay": 1e-5,
                   "label_smoothing": 0.05, "jitter_std_frac": 0.03}
ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)   # each seed x each arch = 15 NN members
N_SNAPSHOTS_PER_RUN = 2                 # extra checkpoints per run via warm restarts
ADABN_MOMENTUM_CANDIDATES = [0.1, 0.3, 0.5, 1.0]
N_CV_FOLDS = 5

os.makedirs("processed", exist_ok=True)

# %%
def per_subject_normalize(X, subjects, stats=None):
    """If stats is None, compute per-subject mean/std from X itself (use for
    train). If stats is a dict {subj: (mean, std)}, apply it (use for held-out
    subjects at inference time so we don't leak test labels into anything —
    note we still only use the subject's OWN unlabeled features, no leakage
    of y ever occurs here, consistent with the original approach)."""
    X_norm = np.zeros_like(X, dtype=np.float64)
    out_stats = {}
    for subj in np.unique(subjects):
        mask = subjects == subj
        if stats is not None and subj in stats:
            subj_mean, subj_std = stats[subj]
        else:
            subj_mean = X[mask].mean(axis=0)
            subj_std = X[mask].std(axis=0) + 1e-8
        X_norm[mask] = (X[mask] - subj_mean) / subj_std
        out_stats[subj] = (subj_mean, subj_std)
    return X_norm.astype(np.float32), out_stats

# %%
train = pd.read_csv(TRAIN_PATH)
feature_cols = [c for c in train.columns if c not in ["Activity", "id", "subject"]]

le = LabelEncoder()
y_train_encoded = le.fit_transform(train["Activity"])
X_train_full = train[feature_cols].values
subjects = train["subject"].values
NUM_CLASSES = len(le.classes_)
INPUT_DIM = X_train_full.shape[1]

# %% [markdown]
# ## Model definitions

# %%
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, num_classes, dropout=0.0, batchnorm=False):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def jitter(X, std_frac, generator):
    """Gaussian noise proportional to each feature's std — cheap augmentation
    for already-normalized tabular features."""
    if std_frac <= 0:
        return X
    feat_std = X.std(dim=0, keepdim=True) + 1e-8  # X is already a tensor here
    noise = torch.randn(X.shape, generator=generator) * feat_std * std_frac
    return X + noise


def make_loader(X, y, batch_size, seed):
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)


def train_nn_snapshots(X_tr_s, y_tr, class_weights, arch, n_epochs, seed,
                        n_snapshots=1, X_val_t=None, y_val=None, early_stop=False, patience=12):
    """Trains one run. If n_snapshots > 1, uses cosine-annealing warm restarts
    and returns a checkpoint at the end of each restart cycle (snapshot
    ensembling) -- more diverse members per training run at ~same compute."""
    torch.manual_seed(seed)
    cpu_gen = torch.Generator().manual_seed(seed + 1000)
    model = MLP(INPUT_DIM, arch["hidden_sizes"], NUM_CLASSES, arch["dropout"], BASE_NN_CONFIG["batchnorm"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_NN_CONFIG["lr"], weight_decay=BASE_NN_CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=BASE_NN_CONFIG["label_smoothing"])
    loader = make_loader(X_tr_s, y_tr, BASE_NN_CONFIG["batch_size"], seed)

    cycle_len = max(1, n_epochs // n_snapshots)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle_len)

    best_val_f1, best_epoch, best_state, patience_counter = -1, 0, None, 0
    snapshots = []
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb = jitter(xb, BASE_NN_CONFIG["jitter_std_frac"], cpu_gen)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

        is_cycle_end = (epoch + 1) % cycle_len == 0
        if is_cycle_end and n_snapshots > 1:
            snapshots.append({k: v.clone() for k, v in model.state_dict().items()})

        if early_stop:
            model.eval()
            with torch.no_grad():
                val_f1 = f1_score(y_val, model(X_val_t).argmax(dim=1).cpu().numpy(), average="macro")
            if val_f1 > best_val_f1:
                best_val_f1, best_epoch, patience_counter = val_f1, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if early_stop:
        model.load_state_dict(best_state)
        return model, best_val_f1, best_epoch, snapshots
    return model, None, n_epochs - 1, snapshots


def adabn_predict_proba(trained_model, X_subject_t, momentum=1.0):
    """Recalibrates BatchNorm running stats to this subject's own unlabeled
    data, then returns softmax probabilities. Momentum controls how fully the
    subject's stats override the training-population prior:
    1.0 = fully replace (original approach), lower = blend toward it (more
    robust when a subject has few test rows)."""
    model = copy.deepcopy(trained_model)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.momentum = momentum
            module.reset_running_stats() if momentum == 1.0 else None
    model.train()
    with torch.no_grad():
        _ = model(X_subject_t)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_subject_t), dim=1).cpu().numpy()
    return probs

# %% [markdown]
# ## GBM branch (optional third model family — different error profile than NNs)

# %%
def get_gbm():
    try:
        import lightgbm as lgb
        return "lightgbm", lgb.LGBMClassifier(
            n_estimators=600, num_leaves=31, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, objective="multiclass",
            class_weight="balanced", random_state=42, verbosity=-1)
    except ImportError:
        pass
    try:
        import xgboost as xgb
        return "xgboost", xgb.XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="mlogloss", tree_method="hist")
    except ImportError:
        pass
    from sklearn.ensemble import GradientBoostingClassifier
    return "sklearn_gbm", GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=42)

GBM_BACKEND, _ = get_gbm()
print(f"GBM backend: {GBM_BACKEND}")

# %% [markdown]
# ## Phase 1: GroupKFold CV — pick robust epoch count, AdaBN momentum, and
# generate out-of-fold predictions for the stacking meta-learner.

# %%
gkf = GroupKFold(n_splits=N_CV_FOLDS)
oof_nn_probs = np.zeros((len(train), NUM_CLASSES))
oof_gbm_probs = np.zeros((len(train), NUM_CLASSES))
fold_best_epochs, fold_momentum_scores = [], {m: [] for m in ADABN_MOMENTUM_CANDIDATES}

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_full, y_train_encoded, groups=subjects)):
    X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
    y_tr, y_val = y_train_encoded[tr_idx], y_train_encoded[val_idx]
    subj_tr, subj_val = subjects[tr_idx], subjects[val_idx]
    assert len(set(subj_tr) & set(subj_val)) == 0

    X_tr_norm, _ = per_subject_normalize(X_tr, subj_tr)
    X_val_norm, _ = per_subject_normalize(X_val, subj_val)   # subject's own unlabeled stats only
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

    cw = torch.tensor(
        np.bincount(y_tr, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_tr, minlength=NUM_CLASSES)),
        dtype=torch.float32).to(device)

    # one representative architecture for epoch-count search (keeps CV cheap)
    model, val_f1, best_epoch, _ = train_nn_snapshots(
        X_tr_norm, y_tr, cw, ARCHS[0], 100, seed=42,
        X_val_t=X_val_t, y_val=y_val, early_stop=True, patience=12)
    fold_best_epochs.append(best_epoch + 1)

    # sweep AdaBN momentum on this fold's held-out subjects
    for mom in ADABN_MOMENTUM_CANDIDATES:
        probs = np.zeros((len(val_idx), NUM_CLASSES))
        for subj in np.unique(subj_val):
            m = subj_val == subj
            probs[m] = adabn_predict_proba(model, X_val_t[torch.tensor(m)], momentum=mom)
        fold_momentum_scores[mom].append(f1_score(y_val, probs.argmax(axis=1), average="macro"))

    # OOF probs for stacking (NN: use momentum=1.0 baseline here, GBM: plain predict_proba)
    oof_probs_fold = np.zeros((len(val_idx), NUM_CLASSES))
    for subj in np.unique(subj_val):
        m = subj_val == subj
        oof_probs_fold[m] = adabn_predict_proba(model, X_val_t[torch.tensor(m)], momentum=1.0)
    oof_nn_probs[val_idx] = oof_probs_fold

    gbm_name, gbm = get_gbm()
    gbm.fit(X_tr_norm, y_tr)
    oof_gbm_probs[val_idx] = gbm.predict_proba(X_val_norm)

    print(f"Fold {fold}: best_epoch={best_epoch}, val_f1={val_f1:.4f}, "
          f"GBM fold f1={f1_score(y_val, oof_gbm_probs[val_idx].argmax(axis=1), average='macro'):.4f}")

n_final_epochs = int(round(np.median(fold_best_epochs)))
best_momentum = max(ADABN_MOMENTUM_CANDIDATES, key=lambda m: np.mean(fold_momentum_scores[m]))
print(f"\nChosen epochs: {n_final_epochs} (per-fold: {fold_best_epochs})")
print("AdaBN momentum sweep:", {m: round(np.mean(v), 4) for m, v in fold_momentum_scores.items()})
print(f"Chosen AdaBN momentum: {best_momentum}")

# %% [markdown]
# ## Fit the stacking meta-learner on out-of-fold predictions
# (Never sees test data or leaks across folds — standard OOF stacking.)

# %%
stack_X = np.hstack([oof_nn_probs, oof_gbm_probs])
meta_learner = LogisticRegression(max_iter=2000)
meta_learner.fit(stack_X, y_train_encoded)
stack_oof_f1 = f1_score(y_train_encoded, meta_learner.predict(stack_X), average="macro")
naive_avg_f1 = f1_score(y_train_encoded, np.mean([oof_nn_probs, oof_gbm_probs], axis=0).argmax(axis=1), average="macro")
print(f"OOF macro F1 — naive average: {naive_avg_f1:.4f} | stacked meta-learner: {stack_oof_f1:.4f}")
# NOTE: stack_oof_f1 is optimistic (meta-learner fit and scored on the same OOF
# set) -- treat the gap over naive_avg_f1 as a soft signal, not a guarantee.
# If it's not clearly better, fall back to averaging for the final blend.

# %% [markdown]
# ## Phase 2: train final ensemble on ALL of train.csv
# Multi-architecture x multi-seed NNs (+ snapshots) + GBM, per-subject normalized.

# %%
X_full_norm, train_subj_stats = per_subject_normalize(X_train_full, subjects)
class_weights_full = torch.tensor(
    np.bincount(y_train_encoded, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_train_encoded, minlength=NUM_CLASSES)),
    dtype=torch.float32).to(device)

ensemble_models = []
for arch_idx, arch in enumerate(ARCHS):
    for seed in ENSEMBLE_SEEDS:
        model, _, _, snapshots = train_nn_snapshots(
            X_full_norm, y_train_encoded, class_weights_full, arch, n_final_epochs,
            seed=seed + arch_idx * 1000, n_snapshots=N_SNAPSHOTS_PER_RUN)
        ensemble_models.append(model)
        tag = f"arch{arch_idx}_seed{seed}"
        torch.save(model.state_dict(), f"processed/nn_{tag}_final.pt")
        for si, snap in enumerate(snapshots[:-1]):  # last snapshot == final model, skip dup
            snap_model = MLP(INPUT_DIM, arch["hidden_sizes"], NUM_CLASSES, arch["dropout"], BASE_NN_CONFIG["batchnorm"]).to(device)
            snap_model.load_state_dict(snap)
            ensemble_models.append(snap_model)
        print(f"Trained {tag}: 1 final + {max(0, len(snapshots) - 1)} snapshot member(s).")

print(f"Total NN ensemble members: {len(ensemble_models)}")

gbm_name, gbm_final = get_gbm()
gbm_final.fit(X_full_norm, y_train_encoded)
print(f"Trained final GBM ({gbm_name}) on full training set.")

with open("processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("processed/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
with open("processed/meta_learner.pkl", "wb") as f:
    pickle.dump(meta_learner, f)
print("Saved all models to processed/")

# %% [markdown]
# ## Real submission — per-subject norm + AdaBN (tuned momentum) NN ensemble
# + GBM, blended via the stacking meta-learner (falls back to averaging if
# the meta-learner did not clearly beat naive averaging on OOF data).

# %%
if os.path.exists(TEST_PATH):
    test = pd.read_csv(TEST_PATH)
    assert "subject" in test.columns, "test.csv must have a 'subject' column"

    X_test = test[feature_cols].values
    test_subjects = test["subject"].values
    # test subjects are unseen -> compute their own per-subject stats from
    # their own unlabeled features (same as original approach, no leakage)
    X_test_norm, _ = per_subject_normalize(X_test, test_subjects)

    nn_probs_all = np.zeros((len(test), NUM_CLASSES))
    for subj in np.unique(test_subjects):
        subj_mask = test_subjects == subj
        X_subj_t = torch.tensor(X_test_norm[subj_mask], dtype=torch.float32).to(device)
        subj_probs = [adabn_predict_proba(m, X_subj_t, momentum=best_momentum) for m in ensemble_models]
        nn_probs_all[subj_mask] = np.mean(subj_probs, axis=0)

    gbm_probs_all = gbm_final.predict_proba(X_test_norm)

    use_stacking = stack_oof_f1 > naive_avg_f1 + 1e-4
    if use_stacking:
        final_probs = meta_learner.predict_proba(np.hstack([nn_probs_all, gbm_probs_all]))
        print("Using stacked meta-learner blend.")
    else:
        final_probs = np.mean([nn_probs_all, gbm_probs_all], axis=0)
        print("Meta-learner didn't clearly beat averaging on OOF -- using naive average blend.")

    preds = final_probs.argmax(axis=1)
    preds_labels = le.inverse_transform(preds)

    test_ids = test["id"] if "id" in test.columns else np.arange(1, len(test) + 1)
    submission = pd.DataFrame({"id": test_ids, "Activity": preds_labels})
    submission.to_csv("submission_advanced.csv", index=False)

    assert len(submission) == len(test)
    assert set(submission["Activity"]) <= set(le.classes_)
    assert submission["Activity"].isnull().sum() == 0
    assert list(submission.columns) == ["id", "Activity"]

    try:
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
        assert list(submission.columns) == list(sample_sub.columns)
        print("Column structure matches sample_submission.csv.")
    except FileNotFoundError:
        print(f"NOTE: could not find {SAMPLE_SUB_PATH} to verify structure.")

    print(f"submission_advanced.csv saved. Rows: {len(submission)}.")
    print("\nPredicted class distribution:")
    print(submission["Activity"].value_counts())
else:
    print(f"\ntest.csv not found at {TEST_PATH}.")