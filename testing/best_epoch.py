# %% [markdown]
# # HAR Competition — Final Submission (Feedforward NN)
#
# Fully self-contained: loads train.csv fresh, trains the confirmed-best NN
# config, and produces submission.csv. Two phases:
#
# 1. Train on X_tr (subject-grouped 80% split), monitor on X_val to find the
#    optimal number of training epochs via early stopping. Also used to
#    generate an INTERIM submission (X_val standing in for test.csv, since
#    it's not released yet) so there's something valid to submit now.
# 2. Retrain a fresh model on ALL of train.csv, for that same optimal epoch
#    count (no more early stopping needed -- we already know when to stop),
#    since more data should only help the model that actually ships. This is
#    the model used for the real test.csv once it's available.
#
# If test.csv IS present at TEST_PATH, this automatically uses it instead of
# the interim/mock fallback -- no manual code editing needed either way.

# %%
import numpy as np
import pandas as pd
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRAIN_PATH = "train.csv"  # update on Kaggle
TEST_PATH = "test.csv"    # update once available
SAMPLE_SUB_PATH = "sample_submission.csv"  # update to its actual location

NN_CONFIG = {
    "hidden_sizes": [256, 128],
    "dropout": 0.5,
    "weight_decay": 0.0,
    "batchnorm": True,
    "lr": 1e-3,
    "batch_size": 64,
    "optimizer_name": "adam",
}
SEED = 42

# %%
train = pd.read_csv(TRAIN_PATH)
feature_cols = [c for c in train.columns if c not in ["Activity", "id", "subject"]]

le = LabelEncoder()
y_train_encoded = le.fit_transform(train["Activity"])
X_train_full = train[feature_cols]
subjects = train["subject"].values
NUM_CLASSES = len(le.classes_)
INPUT_DIM = len(feature_cols)

os.makedirs("processed", exist_ok=True)

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


def make_loader(X, y, batch_size, seed):
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)

# %% [markdown]
# ## Phase 1: train on X_tr, find optimal epoch count via early stopping on X_val

# %%
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(gss.split(X_train_full, y_train_encoded, groups=subjects))

X_tr, X_val = X_train_full.iloc[tr_idx].values, X_train_full.iloc[val_idx].values
y_tr, y_val = y_train_encoded[tr_idx], y_train_encoded[val_idx]
assert len(set(subjects[tr_idx]) & set(subjects[val_idx])) == 0

scaler_phase1 = StandardScaler()
X_tr_s = scaler_phase1.fit_transform(X_tr)
X_val_s = scaler_phase1.transform(X_val)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)

class_weights_tr = torch.tensor(
    np.bincount(y_tr, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_tr, minlength=NUM_CLASSES)),
    dtype=torch.float32,
).to(device)

torch.manual_seed(SEED)
model_phase1 = MLP(INPUT_DIM, NN_CONFIG["hidden_sizes"], NUM_CLASSES, NN_CONFIG["dropout"], NN_CONFIG["batchnorm"]).to(device)
optimizer = torch.optim.Adam(model_phase1.parameters(), lr=NN_CONFIG["lr"], weight_decay=NN_CONFIG["weight_decay"])
criterion = nn.CrossEntropyLoss(weight=class_weights_tr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
loader = make_loader(X_tr_s, y_tr, NN_CONFIG["batch_size"], SEED)

best_val_f1, best_epoch, best_state, patience_counter = -1, 0, None, 0
MAX_EPOCHS, PATIENCE = 100, 12
for epoch in range(MAX_EPOCHS):
    model_phase1.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model_phase1(xb), yb)
        loss.backward()
        optimizer.step()

    model_phase1.eval()
    with torch.no_grad():
        val_f1 = f1_score(y_val, model_phase1(X_val_t).argmax(dim=1).cpu().numpy(), average="macro")
    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1, best_epoch = val_f1, epoch
        best_state = {k: v.clone() for k, v in model_phase1.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break

model_phase1.load_state_dict(best_state)
print(f"Phase 1 done. Best epoch: {best_epoch} | val Macro F1: {best_val_f1:.4f}")

# %% [markdown]
# ## Interim submission (X_val standing in for test.csv until it's released)

# %%
model_phase1.eval()
with torch.no_grad():
    interim_preds = le.inverse_transform(model_phase1(X_val_t).argmax(dim=1).cpu().numpy())

interim_ids = np.arange(1, len(X_val) + 1)
interim_submission = pd.DataFrame({"id": interim_ids, "Activity": interim_preds})
interim_submission.to_csv("interim_submission.csv", index=False)
print(f"interim_submission.csv saved ({len(interim_submission)} rows, mock Macro F1 {best_val_f1:.4f}) "
      f"-- NOT the real leaderboard file, just a demonstration until test.csv exists.")

# %% [markdown]
# ## Phase 2: retrain on ALL of train.csv for best_epoch epochs (final model)

# %%
scaler_final = StandardScaler()
X_full_s = scaler_final.fit_transform(X_train_full.values)

class_weights_full = torch.tensor(
    np.bincount(y_train_encoded, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_train_encoded, minlength=NUM_CLASSES)),
    dtype=torch.float32,
).to(device)

torch.manual_seed(SEED)
model_final = MLP(INPUT_DIM, NN_CONFIG["hidden_sizes"], NUM_CLASSES, NN_CONFIG["dropout"], NN_CONFIG["batchnorm"]).to(device)
optimizer_final = torch.optim.Adam(model_final.parameters(), lr=NN_CONFIG["lr"], weight_decay=NN_CONFIG["weight_decay"])
criterion_final = nn.CrossEntropyLoss(weight=class_weights_full)
loader_final = make_loader(X_full_s, y_train_encoded, NN_CONFIG["batch_size"], SEED)

# No validation set left to early-stop against -- train for exactly the
# epoch count Phase 1 already found to be optimal. No LR scheduler here
# either, since it also needs a validation signal to monitor; a fixed LR
# for this fixed, pre-validated epoch budget is a reasonable simplification.
n_final_epochs = best_epoch + 1
for epoch in range(n_final_epochs):
    model_final.train()
    for xb, yb in loader_final:
        xb, yb = xb.to(device), yb.to(device)
        optimizer_final.zero_grad()
        loss = criterion_final(model_final(xb), yb)
        loss.backward()
        optimizer_final.step()

print(f"Phase 2 done. Trained final model on all {len(train)} rows for {n_final_epochs} epochs.")

torch.save(model_final.state_dict(), "processed/nn_final_model.pt")
with open("processed/nn_scaler.pkl", "wb") as f:
    pickle.dump(scaler_final, f)
with open("processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("processed/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print("Saved final model + scaler to processed/")

# %% [markdown]
# ## Real submission -- uses test.csv if present, else skips with a clear note

# %%
if os.path.exists(TEST_PATH):
    test = pd.read_csv(TEST_PATH)
    X_test = test[feature_cols].values
    X_test_s = scaler_final.transform(X_test)

    model_final.eval()
    with torch.no_grad():
        preds = model_final(torch.tensor(X_test_s, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()
    preds_labels = le.inverse_transform(preds)

    test_ids = test["id"] if "id" in test.columns else np.arange(1, len(test) + 1)
    submission = pd.DataFrame({"id": test_ids, "Activity": preds_labels})
    submission.to_csv("submission.csv", index=False)

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

    print(f"submission.csv saved. Rows: {len(submission)}.")
else:
    print(f"\ntest.csv not found at {TEST_PATH} -- real submission.csv not generated yet.")
    print("Update TEST_PATH above once test.csv is released, then just re-run this cell "
          "(no need to retrain -- the final model is already saved to processed/).")

submission = pd.read_csv("submission.csv")
print(submission["Activity"].value_counts())