# %% [markdown]
# # Outlier/Transition-Window Removal Test
#
# The original HAR windows are 2.56s slices with 50% overlap -- some windows
# inevitably straddle an activity transition (e.g. captured mid-way between
# sitting and standing), yet still get a single hard label. These are
# genuinely blurry, not wrong, training examples. This tests whether
# removing the most atypical windows WITHIN each subject's own activity
# group (using per-subject-normalized features, so "atypical" means relative
# to that person's own typical version of that activity, not the population
# average) produces a cleaner training set and better generalization.
#
# Only applied to TRAINING data -- validation/test stays untouched, since we
# can't identify activity groups for unlabeled data anyway, and the model
# still needs to predict on every real test row regardless of how "typical"
# it looks.

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
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
    return X_norm.astype(np.float32)


def remove_outliers(X_norm, y, subjects, removal_pct):
    """Within each (subject, activity) group, compute distance to that
    group's own centroid, and drop the removal_pct% most distant rows.
    Only meaningful/applied to labeled TRAINING data."""
    if removal_pct == 0:
        return np.arange(len(X_norm))  # keep everyone

    keep_mask = np.ones(len(X_norm), dtype=bool)
    for subj in np.unique(subjects):
        for activity in np.unique(y):
            group_mask = (subjects == subj) & (y == activity)
            if group_mask.sum() < 10:  # too few rows to safely trim
                continue
            group_idx = np.where(group_mask)[0]
            group_X = X_norm[group_idx]
            centroid = group_X.mean(axis=0)
            distances = np.linalg.norm(group_X - centroid, axis=1)
            n_remove = max(1, int(len(group_idx) * removal_pct))
            worst_idx = group_idx[np.argsort(distances)[-n_remove:]]
            keep_mask[worst_idx] = False
    return np.where(keep_mask)[0]

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
# ## Test several removal percentages across all 5 GroupKFold splits

# %%
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_all, y_all, groups=subjects_all))

removal_pcts = [0.0, 0.02, 0.05, 0.10]
all_results = {pct: [] for pct in removal_pcts}

for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    subj_tr_fold, subj_val_fold = subjects_all[tr_idx], subjects_all[val_idx]
    y_tr_fold, y_val_fold = y_all[tr_idx], y_all[val_idx]

    X_tr_norm = per_subject_normalize(X_all[tr_idx], subj_tr_fold)
    X_val_norm = per_subject_normalize(X_all[val_idx], subj_val_fold)

    for pct in removal_pcts:
        keep_idx = remove_outliers(X_tr_norm, y_tr_fold, subj_tr_fold, pct)
        f1 = train_and_eval(X_tr_norm[keep_idx], y_tr_fold[keep_idx], X_val_norm, y_val_fold)
        all_results[pct].append(f1)
        print(f"Fold {fold_idx}, removal={pct*100:.0f}% (kept {len(keep_idx)}/{len(X_tr_norm)} rows) "
              f"| val Macro F1 = {f1:.4f}")

# %% [markdown]
# ## Summary

# %%
print("\n=== Summary across removal percentages ===")
for pct in removal_pcts:
    scores = np.array(all_results[pct])
    print(f"Removal {pct*100:4.0f}% | mean = {scores.mean():.4f} +/- {scores.std():.4f}")

baseline_mean = np.array(all_results[0.0]).mean()
best_pct = max(removal_pcts[1:], key=lambda p: np.array(all_results[p]).mean())
best_mean = np.array(all_results[best_pct]).mean()
print(f"\nBaseline (no removal): {baseline_mean:.4f}")
print(f"Best removal ({best_pct*100:.0f}%): {best_mean:.4f}")
print(f"Improvement: {best_mean - baseline_mean:+.4f}")

if best_mean > baseline_mean + 0.005:  # require a real, non-trivial improvement
    print("\nMeaningful improvement -- worth testing with a fixed removal percentage "
          "(not per-fold-tuned) before building into the submission.")
else:
    print("\nNo meaningful improvement -- training data doesn't have enough removable "
          "noise to matter, or the per-subject normalization already implicitly handles "
          "this by centering each subject's own typical behavior.")