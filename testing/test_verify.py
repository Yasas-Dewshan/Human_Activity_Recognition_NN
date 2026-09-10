# %% [markdown]
# # Deep Error Analysis: Fold 0 (hardest subject group, val Macro F1 0.8856)
#
# Since real test.csv has no labels, this is the best available proxy for
# "what does the model get wrong on genuinely hard unseen subjects" -- Fold 0
# held out subjects [14, 15, 19, 25] and scored notably below the other folds
# (0.8856 vs mean 0.9386). Understanding WHY this specific group was hard
# tells us more about real generalization failure than any single-split
# validation number could.

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
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


def train_and_predict_fold(X_tr, y_tr, X_val, y_val, seed=42, max_epochs=100, patience=12):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
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
# ## Reproduce Fold 0 specifically (subjects 14, 15, 19, 25 held out)

# %%
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_all, y_all, groups=subjects_all))
tr_idx, val_idx = folds[0]  # Fold 0, matches the earlier robustness check ordering

val_subjects = sorted(set(subjects_all[val_idx]))
print(f"Fold 0 held-out subjects: {val_subjects}")

X_tr_fold, X_val_fold = X_all[tr_idx], X_all[val_idx]
y_tr_fold, y_val_fold = y_all[tr_idx], y_all[val_idx]
subj_val_fold = subjects_all[val_idx]

y_pred_fold, fold_f1 = train_and_predict_fold(X_tr_fold, y_tr_fold, X_val_fold, y_val_fold)
print(f"Fold 0 val Macro F1: {fold_f1:.4f}")

# %% [markdown]
# ## Overall confusion matrix + classification report for this hard fold

# %%
print(classification_report(y_val_fold, y_pred_fold, target_names=le.classes_))

fig, ax = plt.subplots(figsize=(7, 7))
ConfusionMatrixDisplay.from_predictions(y_val_fold, y_pred_fold, display_labels=le.classes_, xticks_rotation=45, ax=ax, cmap="Reds")
plt.title("Fold 0 (hardest subjects) Confusion Matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Per-SUBJECT breakdown -- which of the 4 held-out subjects is driving the errors?

# %%
per_subject_rows = []
for subj in val_subjects:
    mask = subj_val_fold == subj
    subj_f1 = f1_score(y_val_fold[mask], y_pred_fold[mask], average="macro")
    n_errors = (y_val_fold[mask] != y_pred_fold[mask]).sum()
    n_total = mask.sum()
    per_subject_rows.append({"subject": subj, "n_rows": n_total, "n_errors": n_errors,
                               "error_rate": n_errors / n_total, "macro_f1": subj_f1})

per_subject_df = pd.DataFrame(per_subject_rows).sort_values("macro_f1")
print(per_subject_df.to_string(index=False))

# %% [markdown]
# ## For the worst subject specifically: what gets confused with what?

# %%
worst_subject = int(per_subject_df.iloc[0]["subject"])
mask = subj_val_fold == worst_subject
y_true_worst = y_val_fold[mask]
y_pred_worst = y_pred_fold[mask]

print(f"\nSubject {worst_subject} (worst performer) -- true vs predicted activity breakdown:")
worst_df = pd.DataFrame({
    "true": le.inverse_transform(y_true_worst),
    "predicted": le.inverse_transform(y_pred_worst),
})
mismatch_df = worst_df[worst_df["true"] != worst_df["predicted"]]
print(f"\nTotal rows for this subject: {len(worst_df)}, errors: {len(mismatch_df)}")
if len(mismatch_df) > 0:
    print("\nError breakdown (true -> predicted):")
    print(mismatch_df.groupby(["true", "predicted"]).size().to_string())