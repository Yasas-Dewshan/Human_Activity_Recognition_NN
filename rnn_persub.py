# %% [markdown]
# # RNN + Per-Subject Normalization: Full 5-Fold Validation
#
# Tests whether per-subject normalization (already validated for the NN:
# 5/5 folds improved, mean 0.9386 -> 0.9731) also helps the RNN on raw
# signals. The RNN's GroupKFold mean (0.9343) was dragged down by the same
# subject-variance problem -- if this fixes it the same way, the RNN could
# combine temporal richness (which the NN's hand-crafted features can't
# capture) with subject-bias correction, potentially beating the NN outright.
#
# Per-subject normalization here works per CHANNEL (9 channels) rather than
# per FEATURE (561 features) -- same principle, applied to raw signal data:
# each subject's own mean/std (across all their own timesteps and windows)
# is used to normalize their own channels.

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
    return np.stack(signals, axis=-1)  # (n_samples, 128, 9)

X_raw = load_signals("train")
kaggle_train = pd.read_csv(KAGGLE_TRAIN_PATH)
le = LabelEncoder()
y_all = le.fit_transform(kaggle_train["Activity"])
subjects_all = kaggle_train["subject"].values
NUM_CLASSES = len(le.classes_)

# %%
def per_subject_normalize_raw(X, subjects):
    """Per-channel, per-subject normalization for raw signal data.
    X shape: (n_samples, 128, 9). Each subject's own mean/std per channel,
    computed across all their own samples AND timesteps."""
    X_norm = np.zeros_like(X, dtype=np.float32)
    for subj in np.unique(subjects):
        mask = subjects == subj
        subj_mean = X[mask].mean(axis=(0, 1), keepdims=True)  # (1, 1, 9)
        subj_std = X[mask].std(axis=(0, 1), keepdims=True) + 1e-8
        X_norm[mask] = (X[mask] - subj_mean) / subj_std
    return X_norm

def global_normalize_raw(X_tr, X_val):
    mean = X_tr.mean(axis=(0, 1), keepdims=True)
    std = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_tr - mean) / std, (X_val - mean) / std

# %%
class RNNClassifier(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=1, rnn_type="lstm", dropout=0.0, num_classes=6):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=(dropout if num_layers > 1 else 0.0))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)
        h_n = h_n.view(self.num_layers, 1, -1, self.hidden_size)
        last = h_n[-1, 0]
        return self.fc(self.dropout(last))


def train_rnn_eval(X_tr_norm, y_tr, X_val_norm, y_val, seed=42, max_epochs=100, patience=12):
    X_tr_t = torch.tensor(X_tr_norm, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

    class_weights = torch.tensor(
        np.bincount(y_tr, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_tr, minlength=NUM_CLASSES)),
        dtype=torch.float32,
    ).to(device)

    torch.manual_seed(seed)
    model = RNNClassifier(9, BEST_RNN_CFG["hidden_size"], BEST_RNN_CFG["num_layers"],
                           BEST_RNN_CFG["rnn_type"], BEST_RNN_CFG["dropout"], NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True, generator=gen)

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
folds = list(gkf.split(X_raw, y_all, groups=subjects_all))

global_scores, persubj_scores = [], []
for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    X_tr_fold, X_val_fold = X_raw[tr_idx], X_raw[val_idx]
    y_tr_fold, y_val_fold = y_all[tr_idx], y_all[val_idx]
    subj_tr_fold, subj_val_fold = subjects_all[tr_idx], subjects_all[val_idx]
    val_subjects = sorted(set(subj_val_fold))

    X_tr_g, X_val_g = global_normalize_raw(X_tr_fold, X_val_fold)
    f1_g = train_rnn_eval(X_tr_g, y_tr_fold, X_val_g, y_val_fold)

    X_tr_p = per_subject_normalize_raw(X_tr_fold, subj_tr_fold)
    X_val_p = per_subject_normalize_raw(X_val_fold, subj_val_fold)
    f1_p = train_rnn_eval(X_tr_p, y_tr_fold, X_val_p, y_val_fold)

    global_scores.append(f1_g)
    persubj_scores.append(f1_p)
    print(f"Fold {fold_idx} (subjects {val_subjects}) | global = {f1_g:.4f} | per-subject = {f1_p:.4f} | diff = {f1_p - f1_g:+.4f}")

global_scores = np.array(global_scores)
persubj_scores = np.array(persubj_scores)

# %% [markdown]
# ## Summary + comparison against NN's per-subject result

# %%
print(f"\nRNN global normalization:      mean {global_scores.mean():.4f} +/- {global_scores.std():.4f}")
print(f"RNN per-subject normalization: mean {persubj_scores.mean():.4f} +/- {persubj_scores.std():.4f}")
print(f"Mean improvement: {persubj_scores.mean() - global_scores.mean():+.4f}")
print(f"Folds where per-subject WON: {(persubj_scores > global_scores).sum()} / 5")

nn_persubj_mean = 0.9731  # already confirmed
print(f"\nFor comparison -- NN per-subject GroupKFold mean: {nn_persubj_mean:.4f}")
print(f"RNN per-subject GroupKFold mean: {persubj_scores.mean():.4f}")
if persubj_scores.mean() > nn_persubj_mean:
    print("RNN + per-subject normalization BEATS NN + per-subject normalization -- "
          "worth building into the final submission.")
else:
    print("RNN + per-subject normalization does not beat the NN version -- "
          "stick with the NN as the primary model.")