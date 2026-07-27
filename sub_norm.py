# %% [markdown]
# # Per-Subject Normalization Test (targets the subject-14 failure pattern)
#
# Instead of a single global StandardScaler fit across all training subjects,
# each row is normalized using ITS OWN SUBJECT's mean/std (computed from that
# subject's own rows only -- no labels needed, so this works identically for
# train.csv and test.csv now that test.csv has a 'subject' column).
#
# Tested directly against Fold 0 (subjects 14, 15, 19, 25 held out), where we
# already found subject 14 alone caused 90 of ~152 errors, almost entirely
# WALKING/WALKING_DOWNSTAIRS misclassified as WALKING_UPSTAIRS -- a plausible
# sign of an atypical personal gait being globally normalized into looking
# like someone else's "upstairs" pattern. This tests whether per-subject
# normalization specifically fixes that.

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = "train.csv"
NN_CONFIG = {"hidden_sizes": [256, 128], "dropout": 0.5, "weight_decay": 0.0, "batchnorm": True, "lr": 1e-3, "batch_size": 64}

# %%
train = pd.read_csv(TRAIN_PATH)
feature_cols = [c for c in train.columns if c not in ["Activity", "id", "subject"]]

le = LabelEncoder()
y_all = le.fit_transform(train["Activity"])
X_all = train[feature_cols].values
subjects_all = train["subject"].values
NUM_CLASSES = len(le.classes_)
INPUT_DIM = X_all.shape[1]

# %%
def per_subject_normalize(X, subjects):
    """Normalize each row using its OWN subject's mean/std, computed from
    that subject's own rows in X. Works for both train (labels available)
    and test (no labels needed -- only uses feature values + subject ID)."""
    X_norm = np.zeros_like(X, dtype=np.float64)
    for subj in np.unique(subjects):
        mask = subjects == subj
        subj_mean = X[mask].mean(axis=0)
        subj_std = X[mask].std(axis=0) + 1e-8
        X_norm[mask] = (X[mask] - subj_mean) / subj_std
    return X_norm

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


def train_and_predict(X_tr_s, y_tr, X_val_s, y_val, seed=42, max_epochs=100, patience=12):
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
    class_weights = torch.tensor(
        np.bincount(y_tr, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_tr, minlength=NUM_CLASSES)),
        dtype=torch.float32,
    ).to(device)

    torch.manual_seed(seed)
    model = MLP(INPUT_DIM, NN_CONFIG["hidden_sizes"], NUM_CLASSES, NN_CONFIG["dropout"], NN_CONFIG["batchnorm"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_CONFIG["lr"], weight_decay=NN_CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
                         batch_size=NN_CONFIG["batch_size"], shuffle=True, generator=gen)

    best_val_f1, best_state, patience_counter = -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_f1 = f1_score(y_val, model(X_val_t).argmax(dim=1).cpu().numpy(), average="macro")
        scheduler.step(val_f1)
        if val_f1 > best_val_f1:
            best_val_f1, best_state, patience_counter = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val_t).argmax(dim=1).cpu().numpy()
    return y_pred, best_val_f1

# %% [markdown]
# ## Reproduce Fold 0, comparing GLOBAL vs PER-SUBJECT normalization

# %%
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_all, y_all, groups=subjects_all))
tr_idx, val_idx = folds[0]

X_tr_fold, X_val_fold = X_all[tr_idx], X_all[val_idx]
y_tr_fold, y_val_fold = y_all[tr_idx], y_all[val_idx]
subj_tr_fold, subj_val_fold = subjects_all[tr_idx], subjects_all[val_idx]
val_subjects = sorted(set(subj_val_fold))
print(f"Fold 0 held-out subjects: {val_subjects}")

# %%
print("\n--- GLOBAL normalization (baseline, for comparison) ---")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_tr_global = scaler.fit_transform(X_tr_fold)
X_val_global = scaler.transform(X_val_fold)
y_pred_global, f1_global = train_and_predict(X_tr_global, y_tr_fold, X_val_global, y_val_fold)
print(f"Global normalization val Macro F1: {f1_global:.4f}")

# %%
print("\n--- PER-SUBJECT normalization ---")
X_tr_persubj = per_subject_normalize(X_tr_fold, subj_tr_fold)
X_val_persubj = per_subject_normalize(X_val_fold, subj_val_fold)  # each val subject normalized using ONLY their own rows, no labels used
y_pred_persubj, f1_persubj = train_and_predict(X_tr_persubj, y_tr_fold, X_val_persubj, y_val_fold)
print(f"Per-subject normalization val Macro F1: {f1_persubj:.4f}")

# %% [markdown]
# ## Did it fix subject 14 specifically?

# %%
for label, y_pred in [("GLOBAL", y_pred_global), ("PER-SUBJECT", y_pred_persubj)]:
    print(f"\n--- {label} normalization: per-subject breakdown ---")
    rows = []
    for subj in val_subjects:
        mask = subj_val_fold == subj
        subj_f1 = f1_score(y_val_fold[mask], y_pred[mask], average="macro")
        n_errors = (y_val_fold[mask] != y_pred[mask]).sum()
        rows.append({"subject": subj, "n_errors": n_errors, "macro_f1": subj_f1})
    print(pd.DataFrame(rows).sort_values("macro_f1").to_string(index=False))

# %%
print("\n--- Subject 14 error breakdown, PER-SUBJECT normalization ---")
mask_14 = subj_val_fold == 14
worst_df = pd.DataFrame({
    "true": le.inverse_transform(y_val_fold[mask_14]),
    "predicted": le.inverse_transform(y_pred_persubj[mask_14]),
})
mismatch = worst_df[worst_df["true"] != worst_df["predicted"]]
print(f"Subject 14 errors with per-subject normalization: {len(mismatch)} (was 90 with global normalization)")
if len(mismatch) > 0:
    print(mismatch.groupby(["true", "predicted"]).size().to_string())

# %%
print(f"\n=== SUMMARY ===")
print(f"Global normalization:      Fold 0 val Macro F1 = {f1_global:.4f}")
print(f"Per-subject normalization: Fold 0 val Macro F1 = {f1_persubj:.4f}")
print(f"Improvement: {f1_persubj - f1_global:+.4f}")