# %% [markdown]
# # HAR Competition — Final Submission (GroupKFold Ensemble)
#
# Different from the earlier 5-seed ensemble: instead of 5 models trained on
# the SAME full dataset with different random seeds, this trains 5 models
# where each one sees a DIFFERENT ~80% subject composition (via GroupKFold),
# using its own held-out subjects for early stopping. This targets the real
# driver of the gap between local validation and leaderboard score -- subject
# variance (confirmed via the earlier GroupKFold robustness check: scores
# ranged 0.8856-0.9880 depending on which subjects were held out) -- rather
# than just weight-init noise, which the previous seed-ensemble addressed.
#
# Each fold's model keeps its own StandardScaler (fit on that fold's training
# subjects only) -- predictions on test.csv use each model with its own
# matching scaler, then average all 5 models' softmax probabilities.

# %%
import numpy as np
import pandas as pd
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
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
N_FOLDS = 5
MAX_EPOCHS, PATIENCE = 100, 12

# %%
train = pd.read_csv(TRAIN_PATH)
feature_cols = [c for c in train.columns if c not in ["Activity", "id", "subject"]]

le = LabelEncoder()
y_all = le.fit_transform(train["Activity"])
X_all = train[feature_cols].values
subjects_all = train["subject"].values
NUM_CLASSES = len(le.classes_)
INPUT_DIM = X_all.shape[1]

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


def train_fold_model(X_tr, y_tr, X_val, y_val, seed):
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
    loader = make_loader(X_tr_s, y_tr, NN_CONFIG["batch_size"], seed)

    best_val_f1, best_state, patience_counter = -1, None, 0
    for epoch in range(MAX_EPOCHS):
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
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, scaler, best_val_f1

# %% [markdown]
# ## Train 5 fold models, each with a different subject composition

# %%
gkf = GroupKFold(n_splits=N_FOLDS)
fold_models = []  # list of (model, scaler) tuples
fold_scores = []

for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups=subjects_all)):
    X_tr_fold, X_val_fold = X_all[tr_idx], X_all[val_idx]
    y_tr_fold, y_val_fold = y_all[tr_idx], y_all[val_idx]
    val_subjects = sorted(set(subjects_all[val_idx]))

    model, scaler, val_f1 = train_fold_model(X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, seed=42 + fold_idx)
    fold_models.append((model, scaler))
    fold_scores.append(val_f1)

    torch.save(model.state_dict(), f"processed/nn_fold{fold_idx}_model.pt")
    with open(f"processed/nn_fold{fold_idx}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"Fold {fold_idx} (held-out subjects {val_subjects}) | val Macro F1 = {val_f1:.4f}")

print(f"\nFold scores: mean {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

with open("processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("processed/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print(f"Saved {N_FOLDS} fold models to processed/")

# %% [markdown]
# ## Real submission -- average all 5 fold models' predictions (each with its own scaler)

# %%
if os.path.exists(TEST_PATH):
    test = pd.read_csv(TEST_PATH)
    X_test = test[feature_cols].values

    all_probs = []
    for model, scaler in fold_models:
        X_test_s = scaler.transform(X_test)  # each fold model uses ITS OWN fitted scaler
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(X_test_t), dim=1).cpu().numpy()
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    preds = avg_probs.argmax(axis=1)
    preds_labels = le.inverse_transform(preds)

    test_ids = test["id"] if "id" in test.columns else np.arange(1, len(test) + 1)
    submission = pd.DataFrame({"id": test_ids, "Activity": preds_labels})
    submission.to_csv("submission_groupkfold.csv", index=False)

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

    print(f"submission_groupkfold.csv saved ({N_FOLDS}-fold ensemble). Rows: {len(submission)}.")
    print("\nPredicted class distribution:")
    print(submission["Activity"].value_counts())
else:
    print(f"\ntest.csv not found at {TEST_PATH}.")