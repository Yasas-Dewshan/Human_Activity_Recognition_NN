# %% [markdown]
# # HAR Competition — Final Submission (NN + SVM weighted blend)
#
# Blends the NN ensemble (5 seeds) with SVM's predicted probabilities.
# The blend weight isn't guessed -- it's tuned using X_tr/X_val (Phase 1's
# held-out split, before either model has seen X_val), then applied to the
# final full-data models. Same "tune via validation, retrain on all data for
# final" pattern used throughout this pipeline.
#
# Given DT/SVM/NN comparison showed NN winning or tying on every class, the
# blend weight is expected to land NN-heavy (or even at 100% NN, meaning SVM
# doesn't help) -- we let the data decide rather than assuming a 50/50 split.

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
from sklearn.svm import SVC
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"

NN_CONFIG = {
    "hidden_sizes": [256, 128], "dropout": 0.5, "weight_decay": 0.0, "batchnorm": True,
    "lr": 1e-3, "batch_size": 64,
}
SVM_PARAMS = {"kernel": "rbf", "C": 10, "gamma": "scale"}  # confirmed best, unscaled
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


def train_nn(X_tr_s, y_tr, class_weights, n_epochs, seed):
    torch.manual_seed(seed)
    model = MLP(INPUT_DIM, NN_CONFIG["hidden_sizes"], NUM_CLASSES, NN_CONFIG["dropout"], NN_CONFIG["batchnorm"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_CONFIG["lr"], weight_decay=NN_CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    loader = make_loader(X_tr_s, y_tr, NN_CONFIG["batch_size"], seed)
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model

# %% [markdown]
# ## Phase 1: find NN epoch count + tune NN/SVM blend weight on held-out X_val

# %%
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(gss.split(X_train_full, y_train_encoded, groups=subjects))

X_tr, X_val = X_train_full.iloc[tr_idx].values, X_train_full.iloc[val_idx].values
y_tr, y_val = y_train_encoded[tr_idx], y_train_encoded[val_idx]
assert len(set(subjects[tr_idx]) & set(subjects[val_idx])) == 0

scaler_p1 = StandardScaler()
X_tr_s = scaler_p1.fit_transform(X_tr)
X_val_s = scaler_p1.transform(X_val)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)

class_weights_tr = torch.tensor(
    np.bincount(y_tr, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_tr, minlength=NUM_CLASSES)),
    dtype=torch.float32,
).to(device)

# NN with early stopping to find optimal epoch count
torch.manual_seed(42)
model_p1 = MLP(INPUT_DIM, NN_CONFIG["hidden_sizes"], NUM_CLASSES, NN_CONFIG["dropout"], NN_CONFIG["batchnorm"]).to(device)
optimizer_p1 = torch.optim.Adam(model_p1.parameters(), lr=NN_CONFIG["lr"], weight_decay=NN_CONFIG["weight_decay"])
criterion_p1 = nn.CrossEntropyLoss(weight=class_weights_tr)
scheduler_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1, mode="max", factor=0.5, patience=4)
loader_p1 = make_loader(X_tr_s, y_tr, NN_CONFIG["batch_size"], 42)

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
        if patience_counter >= 12:
            break

n_final_epochs = best_epoch + 1
print(f"NN epoch count found: {n_final_epochs} (val Macro F1 {best_val_f1:.4f})")

with torch.no_grad():
    nn_val_probs = torch.softmax(model_p1(X_val_t), dim=1).cpu().numpy()

# %%
# SVM trained on X_tr, probability=True needed for blending (adds real
# training time -- internal cross-validation for probability calibration)
print("Training SVM with probability=True (slower than usual due to calibration)...")
svm_p1 = SVC(random_state=42, probability=True, **SVM_PARAMS)
svm_p1.fit(X_tr, y_tr)  # SVM confirmed best on UNSCALED features
svm_val_probs = svm_p1.predict_proba(X_val)
svm_val_f1 = f1_score(y_val, svm_val_probs.argmax(axis=1), average="macro")
print(f"SVM val Macro F1: {svm_val_f1:.4f}")

# %%
# Tune blend weight: alpha * NN_probs + (1-alpha) * SVM_probs
alphas = np.arange(0.5, 1.01, 0.05)
blend_results = []
for alpha in alphas:
    blended = alpha * nn_val_probs + (1 - alpha) * svm_val_probs
    blend_f1 = f1_score(y_val, blended.argmax(axis=1), average="macro")
    blend_results.append({"alpha_nn_weight": alpha, "val_macro_f1": blend_f1})
    print(f"alpha (NN weight) = {alpha:.2f} | val Macro F1 = {blend_f1:.4f}")

blend_df = pd.DataFrame(blend_results)
best_alpha = float(blend_df.loc[blend_df["val_macro_f1"].idxmax(), "alpha_nn_weight"])
print(f"\nBest blend weight: alpha={best_alpha:.2f} (NN weight), {1-best_alpha:.2f} (SVM weight)")
print(f"NN alone: {best_val_f1:.4f} | SVM alone: {svm_val_f1:.4f} | Best blend: {blend_df['val_macro_f1'].max():.4f}")

# %% [markdown]
# ## Phase 2: retrain NN ensemble (5 seeds) + SVM on ALL of train.csv

# %%
scaler_final = StandardScaler()
X_full_s = scaler_final.fit_transform(X_train_full.values)

class_weights_full = torch.tensor(
    np.bincount(y_train_encoded, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_train_encoded, minlength=NUM_CLASSES)),
    dtype=torch.float32,
).to(device)

ensemble_models = []
for seed in ENSEMBLE_SEEDS:
    model = train_nn(X_full_s, y_train_encoded, class_weights_full, n_final_epochs, seed)
    ensemble_models.append(model)
    torch.save(model.state_dict(), f"processed/nn_final_model_seed{seed}.pt")
    print(f"Trained NN seed {seed}.")

print("Training final SVM on all data (probability=True)...")
svm_final = SVC(random_state=42, probability=True, **SVM_PARAMS)
svm_final.fit(X_train_full.values, y_train_encoded)  # unscaled, per confirmed best config

with open("processed/nn_scaler.pkl", "wb") as f:
    pickle.dump(scaler_final, f)
with open("processed/svm_final.pkl", "wb") as f:
    pickle.dump(svm_final, f)
with open("processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("processed/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
with open("processed/blend_alpha.pkl", "wb") as f:
    pickle.dump(best_alpha, f)
print("Saved all final models to processed/")

# %% [markdown]
# ## Real submission -- blended NN ensemble + SVM predictions

# %%
if os.path.exists(TEST_PATH):
    test = pd.read_csv(TEST_PATH)
    X_test = test[feature_cols].values
    X_test_s = scaler_final.transform(X_test)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)

    nn_probs_list = []
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            nn_probs_list.append(torch.softmax(model(X_test_t), dim=1).cpu().numpy())
    nn_test_probs = np.mean(nn_probs_list, axis=0)

    svm_test_probs = svm_final.predict_proba(X_test)  # unscaled X_test, per confirmed best config

    blended_test_probs = best_alpha * nn_test_probs + (1 - best_alpha) * svm_test_probs
    preds = blended_test_probs.argmax(axis=1)
    preds_labels = le.inverse_transform(preds)

    test_ids = test["id"] if "id" in test.columns else np.arange(1, len(test) + 1)
    submission = pd.DataFrame({"id": test_ids, "Activity": preds_labels})
    submission.to_csv("submission_blend.csv", index=False)

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

    print(f"submission_blend.csv saved (alpha={best_alpha:.2f} NN / {1-best_alpha:.2f} SVM). Rows: {len(submission)}.")
    print("\nPredicted class distribution:")
    print(submission["Activity"].value_counts())
else:
    print(f"\ntest.csv not found at {TEST_PATH}.")