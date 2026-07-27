# %% [markdown]
# # HAR Competition — Final Submission (Ensemble of 5 NN seeds)
#
# Same two-phase structure as before, with one change: Phase 2 trains 5
# independently-seeded models on all of train.csv (instead of 1), and the
# final prediction averages their softmax probabilities before taking the
# argmax. Different seeds land at meaningfully different scores due to
# random weight init / minibatch order (we saw 0.9686-0.9767 across seeds
# earlier) -- averaging several models' probability outputs tends to cancel
# out each individual model's specific mistakes, typically giving a small,
# genuine boost over any single seed.

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

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"

NN_CONFIG = {
    "hidden_sizes": [256, 128],
    "dropout": 0.5,
    "weight_decay": 0.0,
    "batchnorm": True,
    "lr": 1e-3,
    "batch_size": 64,
    "optimizer_name": "adam",
}
ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)

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
# ## Phase 1: find optimal epoch count via early stopping (once, not per seed)

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

torch.manual_seed(42)
model_phase1 = MLP(INPUT_DIM, NN_CONFIG["hidden_sizes"], NUM_CLASSES, NN_CONFIG["dropout"], NN_CONFIG["batchnorm"]).to(device)
optimizer = torch.optim.Adam(model_phase1.parameters(), lr=NN_CONFIG["lr"], weight_decay=NN_CONFIG["weight_decay"])
criterion = nn.CrossEntropyLoss(weight=class_weights_tr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
loader = make_loader(X_tr_s, y_tr, NN_CONFIG["batch_size"], 42)

best_val_f1, best_epoch, patience_counter = -1, 0, 0
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
        best_val_f1, best_epoch, patience_counter = val_f1, epoch, 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break

print(f"Phase 1 done. Best epoch: {best_epoch} | val Macro F1: {best_val_f1:.4f}")
n_final_epochs = best_epoch + 1

# %% [markdown]
# ## Phase 2: train ENSEMBLE_SEEDS models on ALL of train.csv, same epoch count

# %%
scaler_final = StandardScaler()
X_full_s = scaler_final.fit_transform(X_train_full.values)

class_weights_full = torch.tensor(
    np.bincount(y_train_encoded, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_train_encoded, minlength=NUM_CLASSES)),
    dtype=torch.float32,
).to(device)

ensemble_models = []
for seed in ENSEMBLE_SEEDS:
    torch.manual_seed(seed)
    model = MLP(INPUT_DIM, NN_CONFIG["hidden_sizes"], NUM_CLASSES, NN_CONFIG["dropout"], NN_CONFIG["batchnorm"]).to(device)
    optimizer_s = torch.optim.Adam(model.parameters(), lr=NN_CONFIG["lr"], weight_decay=NN_CONFIG["weight_decay"])
    criterion_s = nn.CrossEntropyLoss(weight=class_weights_full)
    loader_s = make_loader(X_full_s, y_train_encoded, NN_CONFIG["batch_size"], seed)

    for epoch in range(n_final_epochs):
        model.train()
        for xb, yb in loader_s:
            xb, yb = xb.to(device), yb.to(device)
            optimizer_s.zero_grad()
            loss = criterion_s(model(xb), yb)
            loss.backward()
            optimizer_s.step()

    ensemble_models.append(model)
    torch.save(model.state_dict(), f"processed/nn_final_model_seed{seed}.pt")
    print(f"Trained seed {seed} ({n_final_epochs} epochs).")

with open("processed/nn_scaler.pkl", "wb") as f:
    pickle.dump(scaler_final, f)
with open("processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("processed/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print(f"Trained and saved {len(ensemble_models)} ensemble models to processed/")

# %% [markdown]
# ## Note on evaluation
# The ensemble members are trained on ALL of train.csv (Phase 2), so there's
# no leftover held-out data to fairly re-check the ensemble's benefit without
# testing on data it has already seen. We rely on the earlier out-of-sample
# evidence instead: individual seeds ranged 0.9686-0.9767 in the 5-seed sweep
# run on the X_tr/X_val split -- that spread is exactly what ensembling is
# meant to average out. The real test comes from the leaderboard score itself.

# %% [markdown]
# ## Real submission -- ensemble prediction via averaged softmax probabilities

# %%
if os.path.exists(TEST_PATH):
    test = pd.read_csv(TEST_PATH)
    X_test = test[feature_cols].values
    X_test_s = scaler_final.transform(X_test)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)

    all_probs = []
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(X_test_t), dim=1).cpu().numpy()
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)  # average across the 5 models
    preds = avg_probs.argmax(axis=1)
    preds_labels = le.inverse_transform(preds)

    test_ids = test["id"] if "id" in test.columns else np.arange(1, len(test) + 1)
    submission = pd.DataFrame({"id": test_ids, "Activity": preds_labels})
    submission.to_csv("submission1.csv", index=False)

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

    print(f"submission1.csv saved (ensemble of {len(ensemble_models)} models). Rows: {len(submission)}.")
    print("\nPredicted class distribution:")
    print(submission["Activity"].value_counts())
else:
    print(f"\ntest.csv not found at {TEST_PATH}.")