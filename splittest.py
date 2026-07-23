# %% [markdown]
# # HAR Competition — Part 8: Train/Test Split Robustness Check
#
# Tests whether the NN's 0.9746 score is specific to the one 80/20 subject
# split used so far, or holds up across different splits. With only 21
# subjects total, WHICH subjects land in validation can matter -- this is the
# same class of concern as NN weight-init randomness, but at the data-split
# level instead. Uses GroupKFold (grouped by subject) to rotate through
# several different splits and reports mean +/- std across all of them.
#
# NOTE: this retrains the NN once per fold, so it takes roughly as long as
# the earlier 5-seed run. Uses the already-confirmed best hyperparameters
# (no re-tuning here -- this only tests split sensitivity).

# %%
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
TRAIN_PATH = "train.csv"  # update on Kaggle

train = pd.read_csv(TRAIN_PATH)
feature_cols = [c for c in train.columns if c not in ["Activity", "id", "subject"]]

le = LabelEncoder()
y_all = le.fit_transform(train["Activity"])
X_all = train[feature_cols].values
subjects_all = train["subject"].values

NUM_CLASSES = len(le.classes_)
INPUT_DIM = X_all.shape[1]
print(f"Full dataset: {X_all.shape}, unique subjects: {len(np.unique(subjects_all))}")

# %%
# Confirmed best hyperparameters (from the earlier staged sweep) -- not
# re-tuned here, this test is purely about split sensitivity.
CONFIG = {
    "hidden_sizes": [256, 128],
    "dropout": 0.5,
    "weight_decay": 0.0,
    "batchnorm": True,
    "lr": 1e-3,
    "batch_size": 64,
    "optimizer_name": "adam",
}

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


def train_and_eval_fold(X_tr, y_tr, X_val, y_val, cfg, max_epochs=100, patience=12, seed=42):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)  # fit on this fold's train only
    X_val_s = scaler.transform(X_val)

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)

    class_counts = np.bincount(y_tr, minlength=NUM_CLASSES)
    class_weights = torch.tensor(class_counts.sum() / (NUM_CLASSES * class_counts), dtype=torch.float32).to(device)

    torch.manual_seed(seed)
    model = MLP(INPUT_DIM, cfg["hidden_sizes"], NUM_CLASSES, cfg["dropout"], cfg["batchnorm"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=cfg["batch_size"], shuffle=True, generator=gen)

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
            val_pred = model(X_val_t).argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, val_pred, average="macro")
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_f1

# %%
N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

fold_results = []
for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups=subjects_all)):
    X_tr_fold, X_val_fold = X_all[tr_idx], X_all[val_idx]
    y_tr_fold, y_val_fold = y_all[tr_idx], y_all[val_idx]
    val_subjects = sorted(set(subjects_all[val_idx]))

    fold_f1 = train_and_eval_fold(X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, CONFIG)
    fold_results.append({"fold": fold_idx, "val_subjects": val_subjects, "val_macro_f1": fold_f1})
    print(f"Fold {fold_idx} (subjects {val_subjects}) | val Macro F1 = {fold_f1:.4f}")

fold_df = pd.DataFrame(fold_results)
print(f"\nAcross {N_FOLDS} different subject splits:")
print(f"Mean val Macro F1: {fold_df['val_macro_f1'].mean():.4f} +/- {fold_df['val_macro_f1'].std():.4f}")
print(f"Min: {fold_df['val_macro_f1'].min():.4f}, Max: {fold_df['val_macro_f1'].max():.4f}")

# %%
# Compare against the single-split result we've been using so far
single_split_score = 0.9746  # from the earlier 5-seed run on the original 80/20 split
print(f"\nSingle 80/20 split (original): {single_split_score:.4f}")
print(f"5-fold GroupKFold mean:        {fold_df['val_macro_f1'].mean():.4f}")
print(f"Difference: {abs(single_split_score - fold_df['val_macro_f1'].mean()):.4f}")

if fold_df["val_macro_f1"].std() > 0.02:
    print("\nNOTE: std across folds is fairly large -- performance is meaningfully "
          "sensitive to which subjects land in validation. Worth reporting this "
          "range rather than a single point estimate.")
else:
    print("\nNOTE: std across folds is small -- the original single-split score "
          "appears to be a reasonably stable, representative estimate.")

fold_df.to_csv("processed/split_robustness_results.csv", index=False)
print("\nSaved processed/split_robustness_results.csv")