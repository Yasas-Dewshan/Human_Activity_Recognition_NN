# %% [markdown]
# # 5-seed vs 10-seed Ensemble: diminishing returns check
#
# Tests whether doubling the ensemble from 5 to 10 seeds meaningfully
# improves performance, or whether we've already captured most of the
# available benefit (expected, since individual per-subject-normalized NN
# seeds were already quite consistent -- low variance suggests limited
# further gain from more averaging).
#
# Uses the standard 80/20 subject split (not full GroupKFold, to keep this
# fast) since the question here is about seed count, not fold sensitivity --
# that was already thoroughly tested separately.

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
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


def train_seed(X_tr, y_tr, class_weights, seed, n_epochs=100, patience=12, X_val_t=None, y_val=None):
    torch.manual_seed(seed)
    model = MLP(INPUT_DIM, NN_CONFIG["hidden_sizes"], NUM_CLASSES, NN_CONFIG["dropout"], NN_CONFIG["batchnorm"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_CONFIG["lr"], weight_decay=NN_CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
                         batch_size=NN_CONFIG["batch_size"], shuffle=True, generator=gen)

    best_val_f1, best_state, patience_counter = -1, None, 0
    for epoch in range(n_epochs):
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

# %% [markdown]
# ## Train 10 seeds, compare 5-seed vs 10-seed ensemble averaging

# %%
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(gss.split(X_all, y_all, groups=subjects_all))

X_tr, X_val = X_all[tr_idx], X_all[val_idx]
y_tr, y_val = y_all[tr_idx], y_all[val_idx]
subj_tr, subj_val = subjects_all[tr_idx], subjects_all[val_idx]

X_tr_norm = per_subject_normalize(X_tr, subj_tr)
X_val_norm = per_subject_normalize(X_val, subj_val)
X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

class_weights = torch.tensor(
    np.bincount(y_tr, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_tr, minlength=NUM_CLASSES)),
    dtype=torch.float32,
).to(device)

seeds = list(range(42, 52))  # 10 seeds: 42-51
all_probs = []
individual_f1s = []
for seed in seeds:
    model = train_seed(X_tr_norm, y_tr, class_weights, seed, X_val_t=X_val_t, y_val=y_val)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_val_t), dim=1).cpu().numpy()
    all_probs.append(probs)
    individual_f1 = f1_score(y_val, probs.argmax(axis=1), average="macro")
    individual_f1s.append(individual_f1)
    print(f"seed={seed} | individual val Macro F1 = {individual_f1:.4f}")

print(f"\nIndividual seeds: mean {np.mean(individual_f1s):.4f} +/- {np.std(individual_f1s):.4f}")

# %% [markdown]
# ## Compare ensemble sizes: 1, 3, 5, 7, 10 seeds

# %%
ensemble_results = []
for n_seeds in [1, 3, 5, 7, 10]:
    avg_probs = np.mean(all_probs[:n_seeds], axis=0)
    ens_f1 = f1_score(y_val, avg_probs.argmax(axis=1), average="macro")
    ensemble_results.append({"n_seeds": n_seeds, "val_macro_f1": ens_f1})
    print(f"Ensemble of {n_seeds:2d} seeds | val Macro F1 = {ens_f1:.4f}")

ensemble_df = pd.DataFrame(ensemble_results)
print(f"\nImprovement from 5 -> 10 seeds: "
      f"{ensemble_df[ensemble_df.n_seeds==10].val_macro_f1.values[0] - ensemble_df[ensemble_df.n_seeds==5].val_macro_f1.values[0]:+.4f}")