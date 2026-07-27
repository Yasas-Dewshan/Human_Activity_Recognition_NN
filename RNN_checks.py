# %% [markdown]
# # Extended Task 3, Sanity Checks: Multi-seed + GroupKFold for the RNN
#
# Two checks, mirroring what we already did for the feedforward NN:
# 1. Multi-seed averaging on the same X_tr/X_val split -- is 0.9930 a stable
#    result, or a lucky single draw of weight init / minibatch order?
# 2. GroupKFold across different subject compositions -- does the RNN's
#    advantage hold up across different held-out subjects, or was this split
#    particularly favorable (as happened with the feedforward NN, which
#    dropped from 0.9746 single-split to 0.9386 mean under 5-fold CV)?

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

UCI_DIR = "UCI HAR Dataset"
KAGGLE_TRAIN_PATH = "train.csv"

BEST_RNN_CFG = {"rnn_type": "gru", "hidden_size": 64, "num_layers": 1, "dropout": 0.0}

# %%
signal_names = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

def load_signals(split):
    signals = [np.loadtxt(f"{UCI_DIR}/{split}/Inertial Signals/{name}_{split}.txt") for name in signal_names]
    return np.stack(signals, axis=-1)

X_raw = load_signals("train")
kaggle_train = pd.read_csv(KAGGLE_TRAIN_PATH)
le = LabelEncoder()
y_all = le.fit_transform(kaggle_train["Activity"])
subjects_all = kaggle_train["subject"].values
NUM_CLASSES = len(le.classes_)

# %%
class RNNClassifier(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=1, rnn_type="lstm",
                 bidirectional=False, dropout=0.0, num_classes=6):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=(dropout if num_layers > 1 else 0.0))
        direction_mult = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * direction_mult, num_classes)

    def forward(self, x):
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)
        if self.bidirectional:
            h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
            last = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=-1)
        else:
            h_n = h_n.view(self.num_layers, 1, -1, self.hidden_size)
            last = h_n[-1, 0]
        return self.fc(self.dropout(last))


def train_rnn_on(X_tr_raw, y_tr, X_val_raw, y_val, cfg, seed=42, max_epochs=100, patience=15, batch_size=64, lr=1e-3):
    channel_mean = X_tr_raw.mean(axis=(0, 1), keepdims=True)
    channel_std = X_tr_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    X_tr_norm = (X_tr_raw - channel_mean) / channel_std
    X_val_norm = (X_val_raw - channel_mean) / channel_std

    X_tr_t = torch.tensor(X_tr_norm, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

    class_counts = np.bincount(y_tr, minlength=NUM_CLASSES)
    class_weights = torch.tensor(class_counts.sum() / (NUM_CLASSES * class_counts), dtype=torch.float32).to(device)

    torch.manual_seed(seed)
    model = RNNClassifier(input_size=9, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
                           rnn_type=cfg["rnn_type"], dropout=cfg.get("dropout", 0.0), num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True, generator=gen)

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

    return best_val_f1

# %% [markdown]
# ## Check 1: Multi-seed averaging on the original X_tr/X_val split

# %%
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(gss.split(X_raw, y_all, groups=subjects_all))
X_tr_raw, X_val_raw = X_raw[tr_idx], X_raw[val_idx]
y_tr, y_val = y_all[tr_idx], y_all[val_idx]

seeds = (42, 43, 44, 45, 46)
seed_scores = []
for s in seeds:
    val_f1 = train_rnn_on(X_tr_raw, y_tr, X_val_raw, y_val, BEST_RNN_CFG, seed=s)
    seed_scores.append(val_f1)
    print(f"seed={s} | val Macro F1 = {val_f1:.4f}")

seed_scores = np.array(seed_scores)
print(f"\nMulti-seed: mean {seed_scores.mean():.4f} +/- {seed_scores.std():.4f} "
      f"(min {seed_scores.min():.4f}, max {seed_scores.max():.4f})")

# %% [markdown]
# ## Check 2: GroupKFold across different subject compositions
# Same 5-fold structure used for the feedforward NN's robustness check, for
# direct comparability.

# %%
N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)
fold_scores = []
for fold_idx, (fold_tr_idx, fold_val_idx) in enumerate(gkf.split(X_raw, y_all, groups=subjects_all)):
    X_tr_fold, X_val_fold = X_raw[fold_tr_idx], X_raw[fold_val_idx]
    y_tr_fold, y_val_fold = y_all[fold_tr_idx], y_all[fold_val_idx]
    val_subjects = sorted(set(subjects_all[fold_val_idx]))

    fold_f1 = train_rnn_on(X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, BEST_RNN_CFG, seed=42)
    fold_scores.append(fold_f1)
    print(f"Fold {fold_idx} (subjects {val_subjects}) | val Macro F1 = {fold_f1:.4f}")

fold_scores = np.array(fold_scores)
print(f"\nGroupKFold: mean {fold_scores.mean():.4f} +/- {fold_scores.std():.4f} "
      f"(min {fold_scores.min():.4f}, max {fold_scores.max():.4f})")

# %% [markdown]
# ## Summary: comparing single-split, multi-seed, and GroupKFold estimates

# %%
summary = pd.DataFrame([
    {"check": "Single split (original)", "mean": 0.9930, "std": np.nan, "min": np.nan, "max": np.nan},
    {"check": "Multi-seed (5 seeds, same split)", "mean": seed_scores.mean(), "std": seed_scores.std(),
     "min": seed_scores.min(), "max": seed_scores.max()},
    {"check": "GroupKFold (5 subject splits)", "mean": fold_scores.mean(), "std": fold_scores.std(),
     "min": fold_scores.min(), "max": fold_scores.max()},
])
print(summary.to_string(index=False))

# Compare against the feedforward NN's own GroupKFold result for context
nn_groupkfold_mean = 0.9386
print(f"\nFor context, feedforward NN's GroupKFold mean was: {nn_groupkfold_mean:.4f}")
print(f"RNN's GroupKFold mean: {fold_scores.mean():.4f}")
if fold_scores.mean() > nn_groupkfold_mean:
    print("RNN still ahead of the feedforward NN under fair subject cross-validation.")
else:
    print("RNN's advantage did not hold up under subject cross-validation -- "
          "the single-split result was likely favorable rather than representative.")

summary.to_csv("processed/rnn_sanity_check_summary.csv", index=False)
print("\nSaved processed/rnn_sanity_check_summary.csv")