# %% [markdown]
# # HAR Diagnostics — where is the model actually failing?
#
# GroupKFold with no shuffle is deterministic, so re-running it here
# reproduces the exact same folds as the main pipeline -- including the weak
# fold 1 (val_f1=0.9406). This script isolates that fold and answers:
#   1. Which subject(s) are hard to generalize to?
#   2. Which activity pairs get confused?
#   3. Is it a normalization issue or a genuinely ambiguous subject?
#
# Run this AFTER har_advanced_pipeline.py (reuses the same train.csv).

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = "train.csv"
N_CV_FOLDS = 5
TARGET_FOLD = 1  # the weak one from your run

# %%
train = pd.read_csv(TRAIN_PATH)
feature_cols = [c for c in train.columns if c not in ["Activity", "id", "subject"]]

le = LabelEncoder()
y_train_encoded = le.fit_transform(train["Activity"])
X_train_full = train[feature_cols].values
subjects = train["subject"].values
NUM_CLASSES = len(le.classes_)
INPUT_DIM = X_train_full.shape[1]

# %%
def per_subject_normalize(X, subj):
    X_norm = np.zeros_like(X, dtype=np.float64)
    for s in np.unique(subj):
        mask = subj == s
        X_norm[mask] = (X[mask] - X[mask].mean(axis=0)) / (X[mask].std(axis=0) + 1e-8)
    return X_norm.astype(np.float32)

# %%
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, num_classes, dropout=0.0, batchnorm=False):
        super().__init__()
        layers, prev = [], input_dim
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

# %% [markdown]
# ## Isolate fold 1, train a quick model, get predictions on the held-out subjects

# %%
gkf = GroupKFold(n_splits=N_CV_FOLDS)
splits = list(gkf.split(X_train_full, y_train_encoded, groups=subjects))
tr_idx, val_idx = splits[TARGET_FOLD]

X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
y_tr, y_val = y_train_encoded[tr_idx], y_train_encoded[val_idx]
subj_tr, subj_val = subjects[tr_idx], subjects[val_idx]

print(f"Fold {TARGET_FOLD} held-out subjects: {sorted(np.unique(subj_val))}")
print(f"Held-out subject sample counts: "
      f"{dict(zip(*np.unique(subj_val, return_counts=True)))}")

X_tr_norm = per_subject_normalize(X_tr, subj_tr)
X_val_norm = per_subject_normalize(X_val, subj_val)
X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

cw = torch.tensor(
    np.bincount(y_tr, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_tr, minlength=NUM_CLASSES)),
    dtype=torch.float32).to(device)

torch.manual_seed(42)
model = MLP(INPUT_DIM, [256, 128], NUM_CLASSES, 0.5, True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=cw)
loader = DataLoader(TensorDataset(torch.tensor(X_tr_norm), torch.tensor(y_tr, dtype=torch.long)),
                     batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42))

for epoch in range(20):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    val_preds = model(X_val_t).argmax(dim=1).cpu().numpy()

print(f"\nFold {TARGET_FOLD} macro F1: {f1_score(y_val, val_preds, average='macro'):.4f}")

# %% [markdown]
# ## 1. Per-subject breakdown — which subject(s) drag the score down?

# %%
print("\n--- Per-subject F1 (held-out subjects only) ---")
for s in sorted(np.unique(subj_val)):
    m = subj_val == s
    s_f1 = f1_score(y_val[m], val_preds[m], average="macro")
    print(f"  subject {s}: n={m.sum():4d}  macro F1={s_f1:.4f}")

# %% [markdown]
# ## 2. Per-class breakdown — which activities does the model struggle with?

# %%
print("\n--- Per-class report ---")
print(classification_report(y_val, val_preds, target_names=le.classes_, zero_division=0))

# %% [markdown]
# ## 3. Confusion matrix — which activity PAIRS get mixed up?

# %%
cm = confusion_matrix(y_val, val_preds)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print("\n--- Confusion matrix (rows=true, cols=predicted) ---")
print(cm_df)

# top confused off-diagonal pairs
print("\n--- Top confused pairs (true -> predicted, count) ---")
pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm[i, j] > 0:
            pairs.append((le.classes_[i], le.classes_[j], cm[i, j]))
pairs.sort(key=lambda x: -x[2])
for true_c, pred_c, cnt in pairs[:10]:
    print(f"  {true_c:20s} -> {pred_c:20s}  ({cnt} times)")

# %% [markdown]
# ## 4. Is the worst subject's raw feature distribution unusual?
# Compares per-feature mean of the worst subject vs the rest of train (BEFORE
# per-subject normalization) -- large gaps here suggest a genuine sensor/
# placement difference rather than something normalization should have fixed.

# %%
worst_subject = min(np.unique(subj_val),
                     key=lambda s: f1_score(y_val[subj_val == s], val_preds[subj_val == s], average="macro"))
print(f"\nWorst subject: {worst_subject}")

worst_mask_train = subjects == worst_subject  # in case they appear elsewhere too (they won't, groups are disjoint)
rest_mask = subjects != worst_subject
worst_raw = train.loc[np.isin(subjects, [worst_subject]), feature_cols].values if worst_mask_train.any() else X_val[subj_val == worst_subject]

# use the val split's raw (pre-normalization) rows for the worst subject
worst_raw = X_val[subj_val == worst_subject]
rest_raw = X_tr  # training subjects' raw features as the reference population

feat_diff = np.abs(worst_raw.mean(axis=0) - rest_raw.mean(axis=0))
top_diff_idx = np.argsort(-feat_diff)[:10]
print("\nTop 10 features where this subject's raw mean differs most from the training population:")
for idx in top_diff_idx:
    print(f"  {feature_cols[idx]:40s}  worst_subj_mean={worst_raw[:, idx].mean():+.3f}  "
          f"rest_mean={rest_raw[:, idx].mean():+.3f}  diff={feat_diff[idx]:.3f}")