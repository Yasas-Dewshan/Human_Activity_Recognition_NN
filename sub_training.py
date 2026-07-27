# %% [markdown]
# # Full 5-Fold Validation: Global vs Per-Subject Normalization
#
# Fold 0 alone showed a striking +0.0951 improvement from per-subject
# normalization, driven by fixing subject 14's specific gait-related bias.
# Before rebuilding the submission pipeline around this, we confirm it holds
# up across ALL 5 folds -- not just the one fold we happened to diagnose and
# fix for. This guards against overfitting our fix to a single observed case,
# the same trap that made SVM blending / GroupKFold ensembling / RNN look
# promising locally before underperforming on the real leaderboard.

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


def train_and_eval(X_tr_s, y_tr, X_val_s, y_val, seed=42, max_epochs=100, patience=12):
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

    best_val_f1, patience_counter = -1, 0
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
            best_val_f1, patience_counter = val_f1, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return best_val_f1

# %% [markdown]
# ## Run all 5 folds under both normalization schemes

# %%
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_all, y_all, groups=subjects_all))

global_scores, persubj_scores = [], []
for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    X_tr_fold, X_val_fold = X_all[tr_idx], X_all[val_idx]
    y_tr_fold, y_val_fold = y_all[tr_idx], y_all[val_idx]
    subj_tr_fold, subj_val_fold = subjects_all[tr_idx], subjects_all[val_idx]
    val_subjects = sorted(set(subj_val_fold))

    # Global normalization
    scaler = StandardScaler()
    X_tr_g = scaler.fit_transform(X_tr_fold)
    X_val_g = scaler.transform(X_val_fold)
    f1_g = train_and_eval(X_tr_g, y_tr_fold, X_val_g, y_val_fold)

    # Per-subject normalization
    X_tr_p = per_subject_normalize(X_tr_fold, subj_tr_fold)
    X_val_p = per_subject_normalize(X_val_fold, subj_val_fold)
    f1_p = train_and_eval(X_tr_p, y_tr_fold, X_val_p, y_val_fold)

    global_scores.append(f1_g)
    persubj_scores.append(f1_p)
    print(f"Fold {fold_idx} (subjects {val_subjects}) | global = {f1_g:.4f} | per-subject = {f1_p:.4f} | diff = {f1_p - f1_g:+.4f}")

global_scores = np.array(global_scores)
persubj_scores = np.array(persubj_scores)

# %% [markdown]
# ## Summary

# %%
print(f"\nGlobal normalization:      mean {global_scores.mean():.4f} +/- {global_scores.std():.4f} "
      f"(min {global_scores.min():.4f}, max {global_scores.max():.4f})")
print(f"Per-subject normalization: mean {persubj_scores.mean():.4f} +/- {persubj_scores.std():.4f} "
      f"(min {persubj_scores.min():.4f}, max {persubj_scores.max():.4f})")
print(f"\nMean improvement: {persubj_scores.mean() - global_scores.mean():+.4f}")
print(f"Folds where per-subject WON: {(persubj_scores > global_scores).sum()} / 5")
print(f"Folds where per-subject LOST: {(persubj_scores < global_scores).sum()} / 5")

if (persubj_scores > global_scores).sum() >= 4 and persubj_scores.mean() > global_scores.mean():
    print("\nCONSISTENT IMPROVEMENT across folds -- strong evidence this is a real "
          "fix, not an artifact of one favorable fold. Worth building into the "
          "final submission pipeline.")
elif persubj_scores.mean() > global_scores.mean():
    print("\nImprovement on average but not consistent across all folds -- real "
          "signal, but check which folds got worse before fully committing.")
else:
    print("\nNo overall improvement -- Fold 0's result may have been a "
          "fold-specific fix (subject 14's issue solved) that doesn't "
          "generalize as a universal preprocessing change.")

pd.DataFrame({
    "fold": range(5), "global": global_scores, "per_subject": persubj_scores,
}).to_csv("processed_persubj_comparison.csv", index=False)
print("\nSaved processed_persubj_comparison.csv")