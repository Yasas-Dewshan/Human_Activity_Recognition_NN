# %% [markdown]
# # AdaBN: Adaptive Batch Normalization (test-time domain adaptation)
#
# From Li et al., "Revisiting Batch Normalization for Practical Domain
# Adaptation" (2016). Different from per-subject INPUT normalization (which
# we've already validated): this adapts the MODEL's internal BatchNorm
# statistics to each subject at prediction time, using only that subject's
# own unlabeled data -- no labels needed, so it works identically on
# test.csv (which has 'subject' but no 'Activity').
#
# Mechanism: for each subject being evaluated, temporarily set BatchNorm
# momentum to 1.0 and run one forward pass over that subject's own rows in
# train() mode -- this replaces the trained running_mean/running_var with
# statistics computed purely from that subject's own activations. Then
# switch to eval() mode and predict. The BN weights/biases (learned during
# training) stay the same; only the running statistics adapt per-subject.

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
def per_subject_normalize(X, subjects):
    X_norm = np.zeros_like(X, dtype=np.float64)
    for subj in np.unique(subjects):
        mask = subjects == subj
        subj_mean = X[mask].mean(axis=0)
        subj_std = X[mask].std(axis=0) + 1e-8
        X_norm[mask] = (X[mask] - subj_mean) / subj_std
    return X_norm.astype(np.float32)

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


def train_model(X_tr_s, y_tr, X_val_s, y_val, seed=42, max_epochs=100, patience=12):
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
    return model

# %%
def adabn_predict(trained_model, X_subject_t):
    """Recalibrates BatchNorm running stats to this subject's own data (no
    labels used), then predicts. Uses a fresh copy of the model each time so
    one subject's adaptation never leaks into another's."""
    model = copy.deepcopy(trained_model)
    original_momenta = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            original_momenta[name] = module.momentum
            module.momentum = 1.0  # running stats become exactly this batch's stats

    model.train()
    with torch.no_grad():
        _ = model(X_subject_t)  # forward pass updates BN running stats to this subject
    model.eval()
    with torch.no_grad():
        preds = model(X_subject_t).argmax(dim=1).cpu().numpy()
    return preds

# %% [markdown]
# ## Full 5-fold GroupKFold: standard eval vs AdaBN per-subject recalibration

# %%
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_all, y_all, groups=subjects_all))

standard_scores, adabn_scores = [], []
for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    subj_tr_fold, subj_val_fold = subjects_all[tr_idx], subjects_all[val_idx]
    y_tr_fold, y_val_fold = y_all[tr_idx], y_all[val_idx]
    val_subjects = sorted(set(subj_val_fold))

    X_tr_norm = per_subject_normalize(X_all[tr_idx], subj_tr_fold)
    X_val_norm = per_subject_normalize(X_all[val_idx], subj_val_fold)

    model = train_model(X_tr_norm, y_tr_fold, X_val_norm, y_val_fold)

    # Standard evaluation (whole val set at once, trained BN stats)
    X_val_t_all = torch.tensor(X_val_norm, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        standard_preds = model(X_val_t_all).argmax(dim=1).cpu().numpy()
    standard_f1 = f1_score(y_val_fold, standard_preds, average="macro")

    # AdaBN: recalibrate per subject, predict, aggregate
    adabn_preds = np.zeros(len(y_val_fold), dtype=int)
    for subj in val_subjects:
        subj_mask = subj_val_fold == subj
        X_subj_t = torch.tensor(X_val_norm[subj_mask], dtype=torch.float32).to(device)
        adabn_preds[subj_mask] = adabn_predict(model, X_subj_t)
    adabn_f1 = f1_score(y_val_fold, adabn_preds, average="macro")

    standard_scores.append(standard_f1)
    adabn_scores.append(adabn_f1)
    print(f"Fold {fold_idx} (subjects {val_subjects}) | standard = {standard_f1:.4f} | "
          f"AdaBN = {adabn_f1:.4f} | diff = {adabn_f1 - standard_f1:+.4f}")

standard_scores, adabn_scores = np.array(standard_scores), np.array(adabn_scores)

# %% [markdown]
# ## Summary

# %%
print(f"\nStandard (per-subject input norm only): mean {standard_scores.mean():.4f} +/- {standard_scores.std():.4f}")
print(f"+ AdaBN (adapted internal BN stats):     mean {adabn_scores.mean():.4f} +/- {adabn_scores.std():.4f}")
print(f"Mean improvement: {adabn_scores.mean() - standard_scores.mean():+.4f}")
print(f"Folds where AdaBN WON: {(adabn_scores > standard_scores).sum()} / 5")

if (adabn_scores > standard_scores).sum() >= 4 and adabn_scores.mean() > standard_scores.mean():
    print("\nCONSISTENT IMPROVEMENT -- worth building into the final submission.")
else:
    print("\nNo consistent improvement -- per-subject input normalization likely already "
          "captures most of what AdaBN would additionally correct for.")