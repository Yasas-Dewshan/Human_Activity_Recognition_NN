# %% [markdown]
# # Label Smoothing Value Sweep (GroupKFold)
#
# Your teammate's runs confirmed label_smoothing=0.05 helps (0.96566 ->
# 0.96813 real leaderboard score). This finds the actual best value via
# GroupKFold rather than assuming 0.05 is optimal -- same rigor as every
# other hyperparameter tested in this project.

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


def train_and_eval(X_tr_s, y_tr, X_val_s, y_val, label_smoothing=0.0, seed=42, max_epochs=100, patience=12):
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
    class_weights = torch.tensor(
        np.bincount(y_tr, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_tr, minlength=NUM_CLASSES)),
        dtype=torch.float32,
    ).to(device)

    torch.manual_seed(seed)
    model = MLP(INPUT_DIM, NN_CONFIG["hidden_sizes"], NUM_CLASSES, NN_CONFIG["dropout"], NN_CONFIG["batchnorm"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_CONFIG["lr"], weight_decay=NN_CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
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
# ## Sweep label smoothing values across all 5 GroupKFold splits

# %%
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_all, y_all, groups=subjects_all))

smoothing_values = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
all_results = {ls: [] for ls in smoothing_values}

for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    subj_tr_fold, subj_val_fold = subjects_all[tr_idx], subjects_all[val_idx]
    y_tr_fold, y_val_fold = y_all[tr_idx], y_all[val_idx]

    X_tr_norm = per_subject_normalize(X_all[tr_idx], subj_tr_fold)
    X_val_norm = per_subject_normalize(X_all[val_idx], subj_val_fold)

    for ls in smoothing_values:
        f1 = train_and_eval(X_tr_norm, y_tr_fold, X_val_norm, y_val_fold, label_smoothing=ls)
        all_results[ls].append(f1)
        print(f"Fold {fold_idx}, label_smoothing={ls} | val Macro F1 = {f1:.4f}")

# %% [markdown]
# ## Summary

# %%
print("\n=== Summary across label smoothing values ===")
summary_rows = []
for ls in smoothing_values:
    scores = np.array(all_results[ls])
    summary_rows.append({"label_smoothing": ls, "mean": scores.mean(), "std": scores.std()})
    print(f"label_smoothing={ls:.2f} | mean = {scores.mean():.4f} +/- {scores.std():.4f}")

summary_df = pd.DataFrame(summary_rows).sort_values("mean", ascending=False)
best_ls = summary_df.iloc[0]["label_smoothing"]
print(f"\nBest label_smoothing value: {best_ls}")
print(f"Baseline (0.0): {summary_df[summary_df.label_smoothing==0.0]['mean'].values[0]:.4f}")
print(f"Best ({best_ls}): {summary_df.iloc[0]['mean']:.4f}")