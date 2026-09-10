# %% [markdown]
# # Rigorous NN+RNN Blend Test: Fixed Alpha (tuned once, applied everywhere)
#
# The previous test tuned alpha separately on each fold's own validation
# data, which is optimistic (searching over multiple values and reporting
# the best one on the same data being scored). This version: tune alpha ONCE
# on the original 80/20 split (X_tr/X_val), then apply that SAME fixed value
# across all 5 GroupKFold splits -- an honest test of whether one universal
# blend weight generalizes, not whether we can find a good weight in
# hindsight for each specific fold.

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
NN_CONFIG = {"hidden_sizes": [256, 128], "dropout": 0.5, "weight_decay": 0.0, "batchnorm": True}
RNN_CONFIG = {"rnn_type": "gru", "hidden_size": 64, "num_layers": 1, "dropout": 0.0}

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
feature_cols = [c for c in kaggle_train.columns if c not in ["Activity", "id", "subject"]]
X_feat = kaggle_train[feature_cols].values

le = LabelEncoder()
y_all = le.fit_transform(kaggle_train["Activity"])
subjects_all = kaggle_train["subject"].values
NUM_CLASSES = len(le.classes_)

# %%
def per_subject_normalize(X, subjects):
    X_norm = np.zeros_like(X, dtype=np.float64)
    for subj in np.unique(subjects):
        mask = subjects == subj
        subj_mean = X[mask].mean(axis=0)
        subj_std = X[mask].std(axis=0) + 1e-8
        X_norm[mask] = (X[mask] - subj_mean) / subj_std
    return X_norm.astype(np.float32)

def per_subject_normalize_raw(X, subjects):
    X_norm = np.zeros_like(X, dtype=np.float32)
    for subj in np.unique(subjects):
        mask = subjects == subj
        subj_mean = X[mask].mean(axis=(0, 1), keepdims=True)
        subj_std = X[mask].std(axis=(0, 1), keepdims=True) + 1e-8
        X_norm[mask] = (X[mask] - subj_mean) / subj_std
    return X_norm

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
        return self.fc(self.dropout(h_n[-1, 0]))


def train_generic(model, X_tr, y_tr, X_val_t, y_val, seed=42, max_epochs=100, patience=12, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_weights = torch.tensor(
        np.bincount(y_tr, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_tr, minlength=NUM_CLASSES)),
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
                         batch_size=64, shuffle=True, generator=gen)

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
    return model, best_val_f1

# %% [markdown]
# ## Step 1: tune a SINGLE alpha on the original 80/20 split

# %%
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx0, val_idx0 = next(gss.split(X_feat, y_all, groups=subjects_all))
subj_tr0, subj_val0 = subjects_all[tr_idx0], subjects_all[val_idx0]

X_tr_nn0 = per_subject_normalize(X_feat[tr_idx0], subj_tr0)
X_val_nn0 = per_subject_normalize(X_feat[val_idx0], subj_val0)
X_val_nn0_t = torch.tensor(X_val_nn0, dtype=torch.float32).to(device)
torch.manual_seed(42)
nn_model0 = MLP(561, NN_CONFIG["hidden_sizes"], NUM_CLASSES, NN_CONFIG["dropout"], NN_CONFIG["batchnorm"]).to(device)
nn_model0, _ = train_generic(nn_model0, X_tr_nn0, y_all[tr_idx0], X_val_nn0_t, y_all[val_idx0])
nn_model0.eval()
with torch.no_grad():
    nn_val_probs0 = torch.softmax(nn_model0(X_val_nn0_t), dim=1).cpu().numpy()

X_tr_rnn0 = per_subject_normalize_raw(X_raw[tr_idx0], subj_tr0)
X_val_rnn0 = per_subject_normalize_raw(X_raw[val_idx0], subj_val0)
X_val_rnn0_t = torch.tensor(X_val_rnn0, dtype=torch.float32).to(device)
torch.manual_seed(42)
rnn_model0 = RNNClassifier(9, RNN_CONFIG["hidden_size"], RNN_CONFIG["num_layers"],
                            RNN_CONFIG["rnn_type"], RNN_CONFIG["dropout"], NUM_CLASSES).to(device)
rnn_model0, _ = train_generic(rnn_model0, X_tr_rnn0, y_all[tr_idx0], X_val_rnn0_t, y_all[val_idx0])
rnn_model0.eval()
with torch.no_grad():
    rnn_val_probs0 = torch.softmax(rnn_model0(X_val_rnn0_t), dim=1).cpu().numpy()

best_alpha_fixed, best_f1_0 = 1.0, f1_score(y_all[val_idx0], nn_val_probs0.argmax(axis=1), average="macro")
for alpha in np.arange(0.5, 1.01, 0.05):
    blended = alpha * nn_val_probs0 + (1 - alpha) * rnn_val_probs0
    f1_a = f1_score(y_all[val_idx0], blended.argmax(axis=1), average="macro")
    if f1_a > best_f1_0:
        best_f1_0, best_alpha_fixed = f1_a, alpha

print(f"Fixed alpha (NN weight) chosen from 80/20 split: {best_alpha_fixed:.2f}")

# %% [markdown]
# ## Step 2: apply this FIXED alpha across all 5 GroupKFold splits (honest test)

# %%
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_feat, y_all, groups=subjects_all))

nn_scores, fixed_blend_scores = [], []
for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    y_tr_fold, y_val_fold = y_all[tr_idx], y_all[val_idx]
    subj_tr_fold, subj_val_fold = subjects_all[tr_idx], subjects_all[val_idx]
    val_subjects = sorted(set(subj_val_fold))

    X_tr_nn = per_subject_normalize(X_feat[tr_idx], subj_tr_fold)
    X_val_nn = per_subject_normalize(X_feat[val_idx], subj_val_fold)
    X_val_nn_t = torch.tensor(X_val_nn, dtype=torch.float32).to(device)
    torch.manual_seed(42)
    nn_model = MLP(561, NN_CONFIG["hidden_sizes"], NUM_CLASSES, NN_CONFIG["dropout"], NN_CONFIG["batchnorm"]).to(device)
    nn_model, nn_f1 = train_generic(nn_model, X_tr_nn, y_tr_fold, X_val_nn_t, y_val_fold)
    nn_model.eval()
    with torch.no_grad():
        nn_val_probs = torch.softmax(nn_model(X_val_nn_t), dim=1).cpu().numpy()

    X_tr_rnn = per_subject_normalize_raw(X_raw[tr_idx], subj_tr_fold)
    X_val_rnn = per_subject_normalize_raw(X_raw[val_idx], subj_val_fold)
    X_val_rnn_t = torch.tensor(X_val_rnn, dtype=torch.float32).to(device)
    torch.manual_seed(42)
    rnn_model = RNNClassifier(9, RNN_CONFIG["hidden_size"], RNN_CONFIG["num_layers"],
                               RNN_CONFIG["rnn_type"], RNN_CONFIG["dropout"], NUM_CLASSES).to(device)
    rnn_model, rnn_f1 = train_generic(rnn_model, X_tr_rnn, y_tr_fold, X_val_rnn_t, y_val_fold)
    rnn_model.eval()
    with torch.no_grad():
        rnn_val_probs = torch.softmax(rnn_model(X_val_rnn_t), dim=1).cpu().numpy()

    # Use the FIXED alpha found in Step 1 -- no re-tuning per fold
    blended = best_alpha_fixed * nn_val_probs + (1 - best_alpha_fixed) * rnn_val_probs
    fixed_blend_f1 = f1_score(y_val_fold, blended.argmax(axis=1), average="macro")

    nn_scores.append(nn_f1)
    fixed_blend_scores.append(fixed_blend_f1)
    print(f"Fold {fold_idx} (subjects {val_subjects}) | NN alone = {nn_f1:.4f} | "
          f"fixed-alpha blend ({best_alpha_fixed:.2f}) = {fixed_blend_f1:.4f} | diff = {fixed_blend_f1 - nn_f1:+.4f}")

nn_scores, fixed_blend_scores = np.array(nn_scores), np.array(fixed_blend_scores)

# %% [markdown]
# ## Summary

# %%
print(f"\nNN alone:                mean {nn_scores.mean():.4f} +/- {nn_scores.std():.4f}")
print(f"Fixed-alpha blend:       mean {fixed_blend_scores.mean():.4f} +/- {fixed_blend_scores.std():.4f}")
print(f"Mean improvement: {fixed_blend_scores.mean() - nn_scores.mean():+.4f}")
print(f"Folds where fixed blend WON: {(fixed_blend_scores > nn_scores).sum()} / 5")

if (fixed_blend_scores > nn_scores).sum() >= 4 and fixed_blend_scores.mean() > nn_scores.mean():
    print(f"\nCONFIRMED with honest fixed alpha ({best_alpha_fixed:.2f}) -- worth building "
          f"into the final submission.")
else:
    print("\nDid not hold up with a fixed, honestly-tuned alpha -- the earlier per-fold "
          "tuning was likely overfitting to each fold's own validation data. NN alone "
          "remains the best choice.")