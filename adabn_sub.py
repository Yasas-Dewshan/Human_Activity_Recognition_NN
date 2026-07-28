# %% [markdown]
# # HAR Competition — Final Submission (Per-Subject Norm + 5-Seed Ensemble + AdaBN)
#
# Combines all three validated techniques:
# 1. Per-subject input normalization (5/5 folds, mean 0.9386 -> 0.9731)
# 2. 5-seed ensemble (confirmed real leaderboard gain, 0.95039 -> 0.95701)
# 3. AdaBN test-time recalibration (4/5 folds, +0.0024 mean, adapts each
#    ensemble member's BatchNorm stats to each test subject individually
#    before predicting -- no labels needed, no retraining needed)
#
# Same two-phase pattern: Phase 1 finds the epoch count, Phase 2 trains the
# 5-seed ensemble on all of train.csv. AdaBN is applied at prediction time
# only, per test subject, using each ensemble member's own copy.

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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"

NN_CONFIG = {
    "hidden_sizes": [256, 128], "dropout": 0.5, "weight_decay": 0.0, "batchnorm": True,
    "lr": 1e-3, "batch_size": 64,
}
ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)

# %%
def per_subject_normalize(X, subjects):
    X_norm = np.zeros_like(X, dtype=np.float64)
    for subj in np.unique(subjects):
        mask = subjects == subj
        subj_mean = X[mask].mean(axis=0)
        subj_std = X[mask].std(axis=0) + 1e-8
        X_norm[mask] = (X[mask] - subj_mean) / subj_std
    return X_norm.astype(np.float32)

# %%
train = pd.read_csv(TRAIN_PATH)
feature_cols = [c for c in train.columns if c not in ["Activity", "id", "subject"]]

le = LabelEncoder()
y_train_encoded = le.fit_transform(train["Activity"])
X_train_full = train[feature_cols].values
subjects = train["subject"].values
NUM_CLASSES = len(le.classes_)
INPUT_DIM = X_train_full.shape[1]

os.makedirs("processed", exist_ok=True)

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


def make_loader(X, y, batch_size, seed):
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)


def train_nn(X_tr_s, y_tr, class_weights, n_epochs, seed, X_val_t=None, y_val=None, early_stop=False, patience=12):
    torch.manual_seed(seed)
    model = MLP(INPUT_DIM, NN_CONFIG["hidden_sizes"], NUM_CLASSES, NN_CONFIG["dropout"], NN_CONFIG["batchnorm"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_CONFIG["lr"], weight_decay=NN_CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4) if early_stop else None
    loader = make_loader(X_tr_s, y_tr, NN_CONFIG["batch_size"], seed)

    best_val_f1, best_epoch, best_state, patience_counter = -1, 0, None, 0
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        if early_stop:
            model.eval()
            with torch.no_grad():
                val_f1 = f1_score(y_val, model(X_val_t).argmax(dim=1).cpu().numpy(), average="macro")
            scheduler.step(val_f1)
            if val_f1 > best_val_f1:
                best_val_f1, best_epoch, patience_counter = val_f1, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if early_stop:
        model.load_state_dict(best_state)
        return model, best_val_f1, best_epoch
    return model, None, n_epochs - 1

# %%
def adabn_predict_proba(trained_model, X_subject_t):
    """Recalibrates BatchNorm running stats to this subject's own data (no
    labels used), then returns softmax probabilities. Fresh copy per call so
    no cross-subject contamination."""
    model = copy.deepcopy(trained_model)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.momentum = 1.0
    model.train()
    with torch.no_grad():
        _ = model(X_subject_t)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_subject_t), dim=1).cpu().numpy()
    return probs

# %% [markdown]
# ## Phase 1: find optimal epoch count via early stopping (per-subject normalized)

# %%
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(gss.split(X_train_full, y_train_encoded, groups=subjects))

X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
y_tr, y_val = y_train_encoded[tr_idx], y_train_encoded[val_idx]
subj_tr, subj_val = subjects[tr_idx], subjects[val_idx]
assert len(set(subj_tr) & set(subj_val)) == 0

X_tr_norm = per_subject_normalize(X_tr, subj_tr)
X_val_norm = per_subject_normalize(X_val, subj_val)
X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

class_weights_tr = torch.tensor(
    np.bincount(y_tr, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_tr, minlength=NUM_CLASSES)),
    dtype=torch.float32,
).to(device)

_, best_val_f1, best_epoch = train_nn(X_tr_norm, y_tr, class_weights_tr, 100, seed=42,
                                        X_val_t=X_val_t, y_val=y_val, early_stop=True, patience=12)
n_final_epochs = best_epoch + 1
print(f"Phase 1 done. Best epoch: {best_epoch} | val Macro F1: {best_val_f1:.4f}")

# %% [markdown]
# ## Phase 2: train 5-seed ensemble on ALL of train.csv (per-subject normalized)

# %%
X_full_norm = per_subject_normalize(X_train_full, subjects)

class_weights_full = torch.tensor(
    np.bincount(y_train_encoded, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_train_encoded, minlength=NUM_CLASSES)),
    dtype=torch.float32,
).to(device)

ensemble_models = []
for seed in ENSEMBLE_SEEDS:
    model, _, _ = train_nn(X_full_norm, y_train_encoded, class_weights_full, n_final_epochs, seed=seed)
    ensemble_models.append(model)
    torch.save(model.state_dict(), f"processed/nn_persubj_adabn_seed{seed}.pt")
    print(f"Trained seed {seed} ({n_final_epochs} epochs).")

with open("processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("processed/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print(f"Trained and saved {len(ensemble_models)} models to processed/")

# %% [markdown]
# ## Real submission -- per-subject normalization + AdaBN recalibration + 5-seed ensemble

# %%
if os.path.exists(TEST_PATH):
    test = pd.read_csv(TEST_PATH)
    assert "subject" in test.columns, "test.csv must have a 'subject' column"

    X_test = test[feature_cols].values
    test_subjects = test["subject"].values
    X_test_norm = per_subject_normalize(X_test, test_subjects)

    # For each test subject, get each ensemble member's AdaBN-recalibrated
    # probabilities, then average across seeds (same as before, now with
    # AdaBN applied per subject before averaging).
    all_preds_probs = np.zeros((len(test), NUM_CLASSES))
    for subj in np.unique(test_subjects):
        subj_mask = test_subjects == subj
        X_subj_t = torch.tensor(X_test_norm[subj_mask], dtype=torch.float32).to(device)

        subj_probs = []
        for model in ensemble_models:
            probs = adabn_predict_proba(model, X_subj_t)
            subj_probs.append(probs)
        avg_subj_probs = np.mean(subj_probs, axis=0)
        all_preds_probs[subj_mask] = avg_subj_probs

    preds = all_preds_probs.argmax(axis=1)
    preds_labels = le.inverse_transform(preds)

    test_ids = test["id"] if "id" in test.columns else np.arange(1, len(test) + 1)
    submission = pd.DataFrame({"id": test_ids, "Activity": preds_labels})
    submission.to_csv("submission_adabn.csv", index=False)

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

    print(f"submission_persubj_adabn_ensemble.csv saved. Rows: {len(submission)}.")
    print("\nPredicted class distribution:")
    print(submission["Activity"].value_counts())
else:
    print(f"\ntest.csv not found at {TEST_PATH}.")