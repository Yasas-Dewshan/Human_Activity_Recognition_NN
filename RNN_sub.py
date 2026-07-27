# %% [markdown]
# # HAR Competition — Final Submission (RNN, GRU on raw signals)
#
# Same two-phase pattern used for the feedforward NN:
# 1. Train on X_tr (subject-grouped split, seed=42 -- matching the run that
#    got 0.9930), monitor X_val with early stopping to find the optimal
#    epoch count.
# 2. Retrain a fresh model on ALL of train.csv's raw signals for that same
#    epoch count, then predict on the real test.csv's raw signals.
#
# HONEST EXPECTATION: the sanity checks showed this RNN's GroupKFold mean was
# 0.9343 (vs the single-split 0.9930) -- essentially tied with the feedforward
# NN's own 0.9386 GroupKFold mean. The real leaderboard score is more likely
# to land in that ~0.93-0.95 range than near 0.9930, following the same
# pattern seen with the NN (local 0.9746 -> real 0.95039).

# %%
import numpy as np
import pandas as pd
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

UCI_TRAIN_DIR = "UCI HAR Dataset/train"
UCI_TEST_DIR = "UCI HAR Dataset/test"
KAGGLE_TRAIN_PATH = "train.csv"
KAGGLE_TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"

BEST_RNN_CFG = {"rnn_type": "gru", "hidden_size": 64, "num_layers": 1, "dropout": 0.0}
SEED = 42

signal_names = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

def load_signals(uci_dir, split):
    signals = [np.loadtxt(f"{uci_dir}/Inertial Signals/{name}_{split}.txt") for name in signal_names]
    return np.stack(signals, axis=-1)

# %%
X_raw_train = load_signals(UCI_TRAIN_DIR, "train")
kaggle_train = pd.read_csv(KAGGLE_TRAIN_PATH)
le = LabelEncoder()
y_all = le.fit_transform(kaggle_train["Activity"])
subjects_all = kaggle_train["subject"].values
NUM_CLASSES = len(le.classes_)
os.makedirs("processed", exist_ok=True)

print(f"Raw train signals: {X_raw_train.shape}")

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


def make_loader(X, y, batch_size, seed):
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)

# %% [markdown]
# ## Phase 1: find optimal epoch count via early stopping on X_val

# %%
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(gss.split(X_raw_train, y_all, groups=subjects_all))
X_tr_raw, X_val_raw = X_raw_train[tr_idx], X_raw_train[val_idx]
y_tr, y_val = y_all[tr_idx], y_all[val_idx]
assert len(set(subjects_all[tr_idx]) & set(subjects_all[val_idx])) == 0

channel_mean_p1 = X_tr_raw.mean(axis=(0, 1), keepdims=True)
channel_std_p1 = X_tr_raw.std(axis=(0, 1), keepdims=True) + 1e-8
X_tr_norm = (X_tr_raw - channel_mean_p1) / channel_std_p1
X_val_norm = (X_val_raw - channel_mean_p1) / channel_std_p1
X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

class_weights_tr = torch.tensor(
    np.bincount(y_tr, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_tr, minlength=NUM_CLASSES)),
    dtype=torch.float32,
).to(device)

torch.manual_seed(SEED)
model_p1 = RNNClassifier(9, BEST_RNN_CFG["hidden_size"], BEST_RNN_CFG["num_layers"],
                          BEST_RNN_CFG["rnn_type"], BEST_RNN_CFG["dropout"], NUM_CLASSES).to(device)
optimizer_p1 = torch.optim.Adam(model_p1.parameters(), lr=1e-3)
criterion_p1 = nn.CrossEntropyLoss(weight=class_weights_tr)
scheduler_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1, mode="max", factor=0.5, patience=4)
loader_p1 = make_loader(X_tr_norm, y_tr, 64, SEED)

best_val_f1, best_epoch, patience_counter = -1, 0, 0
for epoch in range(100):
    model_p1.train()
    for xb, yb in loader_p1:
        xb, yb = xb.to(device), yb.to(device)
        optimizer_p1.zero_grad()
        loss = criterion_p1(model_p1(xb), yb)
        loss.backward()
        optimizer_p1.step()
    model_p1.eval()
    with torch.no_grad():
        val_f1 = f1_score(y_val, model_p1(X_val_t).argmax(dim=1).cpu().numpy(), average="macro")
    scheduler_p1.step(val_f1)
    if val_f1 > best_val_f1:
        best_val_f1, best_epoch, patience_counter = val_f1, epoch, 0
    else:
        patience_counter += 1
        if patience_counter >= 15:
            break

n_final_epochs = best_epoch + 1
print(f"Phase 1 done. Best epoch: {best_epoch} | val Macro F1: {best_val_f1:.4f}")

# %% [markdown]
# ## Phase 2: retrain on ALL of train.csv's raw signals for that epoch count

# %%
channel_mean_full = X_raw_train.mean(axis=(0, 1), keepdims=True)
channel_std_full = X_raw_train.std(axis=(0, 1), keepdims=True) + 1e-8
X_full_norm = (X_raw_train - channel_mean_full) / channel_std_full

class_weights_full = torch.tensor(
    np.bincount(y_all, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_all, minlength=NUM_CLASSES)),
    dtype=torch.float32,
).to(device)

torch.manual_seed(SEED)
model_final = RNNClassifier(9, BEST_RNN_CFG["hidden_size"], BEST_RNN_CFG["num_layers"],
                             BEST_RNN_CFG["rnn_type"], BEST_RNN_CFG["dropout"], NUM_CLASSES).to(device)
optimizer_final = torch.optim.Adam(model_final.parameters(), lr=1e-3)
criterion_final = nn.CrossEntropyLoss(weight=class_weights_full)
loader_final = make_loader(X_full_norm, y_all, 64, SEED)

for epoch in range(n_final_epochs):
    model_final.train()
    for xb, yb in loader_final:
        xb, yb = xb.to(device), yb.to(device)
        optimizer_final.zero_grad()
        loss = criterion_final(model_final(xb), yb)
        loss.backward()
        optimizer_final.step()

print(f"Phase 2 done. Trained final RNN on all {len(kaggle_train)} rows for {n_final_epochs} epochs.")

torch.save(model_final.state_dict(), "processed/rnn_final_model.pt")
with open("processed/rnn_channel_stats.pkl", "wb") as f:
    pickle.dump({"mean": channel_mean_full, "std": channel_std_full}, f)
with open("processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Saved final RNN model to processed/")

# %% [markdown]
# ## Real submission -- predict on test.csv's raw signals

# %%
if os.path.exists(KAGGLE_TEST_PATH) and os.path.exists(f"{UCI_TEST_DIR}/Inertial Signals"):
    X_raw_test = load_signals(UCI_TEST_DIR, "test")
    test = pd.read_csv(KAGGLE_TEST_PATH)

    # Basic alignment sanity check -- row counts should match
    assert len(X_raw_test) == len(test), \
        f"Row count mismatch: raw test signals ({len(X_raw_test)}) vs test.csv ({len(test)})"
    print(f"Row count check passed: {len(test)} rows in both sources.")

    X_test_norm = (X_raw_test - channel_mean_full) / channel_std_full
    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32).to(device)

    model_final.eval()
    with torch.no_grad():
        preds = model_final(X_test_t).argmax(dim=1).cpu().numpy()
    preds_labels = le.inverse_transform(preds)

    test_ids = test["id"] if "id" in test.columns else np.arange(1, len(test) + 1)
    submission = pd.DataFrame({"id": test_ids, "Activity": preds_labels})
    submission.to_csv("submission_rnn.csv", index=False)

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

    print(f"submission_rnn.csv saved. Rows: {len(submission)}.")
    print("\nPredicted class distribution:")
    print(submission["Activity"].value_counts())
else:
    print(f"\ntest.csv or raw test signals not found. Check KAGGLE_TEST_PATH and UCI_TEST_DIR.")