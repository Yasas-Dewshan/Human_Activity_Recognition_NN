# %% [markdown]
# # HAR Competition — Final Submission (1D CNN, per-subject norm + 5-seed ensemble)
#
# Same winning pattern as the NN submission (per-subject normalization +
# 5-seed ensemble), applied to the 1D CNN architecture instead. Two-phase:
# 1. Find optimal epoch count via early stopping on X_tr/X_val (per-subject
#    normalized raw signals, winning architecture: 3 conv blocks, dropout=0.3).
# 2. Train 5 seeds on ALL of train.csv's raw signals, per-subject normalized.
#    Predict on test.csv's raw signals, normalized using test.csv's own
#    'subject' column (no labels needed).

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

CNN_CFG = dict(num_conv_blocks=3, dropout=0.3, conv_channels=(64, 128, 128))
ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)

# %%
signal_names = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

def load_signals(uci_dir, split):
    signals = [np.loadtxt(f"{uci_dir}/Inertial Signals/{name}_{split}.txt") for name in signal_names]
    return np.stack(signals, axis=-1)

def per_subject_normalize_raw(X, subjects):
    X_norm = np.zeros_like(X, dtype=np.float32)
    for subj in np.unique(subjects):
        mask = subjects == subj
        subj_mean = X[mask].mean(axis=(0, 1), keepdims=True)
        subj_std = X[mask].std(axis=(0, 1), keepdims=True) + 1e-8
        X_norm[mask] = (X[mask] - subj_mean) / subj_std
    return X_norm

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
class CNN1D(nn.Module):
    def __init__(self, num_channels=9, num_classes=6, conv_channels=(64, 128, 128),
                 kernel_size=5, dropout=0.3, num_conv_blocks=3):
        super().__init__()
        layers = []
        in_ch = num_channels
        for i in range(num_conv_blocks):
            out_ch = conv_channels[min(i, len(conv_channels) - 1)]
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


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

X_tr_norm = per_subject_normalize_raw(X_tr_raw, subjects_all[tr_idx])
X_val_norm = per_subject_normalize_raw(X_val_raw, subjects_all[val_idx])
X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

class_weights_tr = torch.tensor(
    np.bincount(y_tr, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_tr, minlength=NUM_CLASSES)),
    dtype=torch.float32,
).to(device)

torch.manual_seed(42)
model_p1 = CNN1D(num_channels=9, num_classes=NUM_CLASSES, **CNN_CFG).to(device)
optimizer_p1 = torch.optim.Adam(model_p1.parameters(), lr=1e-3)
criterion_p1 = nn.CrossEntropyLoss(weight=class_weights_tr)
scheduler_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1, mode="max", factor=0.5, patience=4)
loader_p1 = make_loader(X_tr_norm, y_tr, 64, 42)

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
# ## Phase 2: train 5-seed ensemble on ALL of train.csv's raw signals

# %%
X_full_norm = per_subject_normalize_raw(X_raw_train, subjects_all)

class_weights_full = torch.tensor(
    np.bincount(y_all, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_all, minlength=NUM_CLASSES)),
    dtype=torch.float32,
).to(device)

ensemble_models = []
for seed in ENSEMBLE_SEEDS:
    torch.manual_seed(seed)
    model = CNN1D(num_channels=9, num_classes=NUM_CLASSES, **CNN_CFG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights_full)
    loader = make_loader(X_full_norm, y_all, 64, seed)

    for epoch in range(n_final_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    ensemble_models.append(model)
    torch.save(model.state_dict(), f"processed/cnn1d_seed{seed}.pt")
    print(f"Trained seed {seed} ({n_final_epochs} epochs).")

with open("processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"Trained and saved {len(ensemble_models)} CNN models to processed/")

# %% [markdown]
# ## Real submission -- test.csv's raw signals, per-subject normalized using test.csv's own subject column

# %%
if os.path.exists(KAGGLE_TEST_PATH) and os.path.exists(f"{UCI_TEST_DIR}/Inertial Signals"):
    X_raw_test = load_signals(UCI_TEST_DIR, "test")
    test = pd.read_csv(KAGGLE_TEST_PATH)
    assert "subject" in test.columns, "test.csv must have a 'subject' column for per-subject normalization"
    assert len(X_raw_test) == len(test), f"Row count mismatch: raw signals ({len(X_raw_test)}) vs test.csv ({len(test)})"

    test_subjects = test["subject"].values
    X_test_norm = per_subject_normalize_raw(X_raw_test, test_subjects)
    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32).to(device)

    all_probs = []
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(X_test_t), dim=1).cpu().numpy()
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    preds = avg_probs.argmax(axis=1)
    preds_labels = le.inverse_transform(preds)

    test_ids = test["id"] if "id" in test.columns else np.arange(1, len(test) + 1)
    submission = pd.DataFrame({"id": test_ids, "Activity": preds_labels})
    submission.to_csv("submission_cnn1d.csv", index=False)

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

    print(f"submission_cnn1d.csv saved (per-subject norm + {len(ensemble_models)}-seed CNN ensemble). Rows: {len(submission)}.")
    print("\nPredicted class distribution:")
    print(submission["Activity"].value_counts())
else:
    print(f"\ntest.csv or raw test signals not found.")