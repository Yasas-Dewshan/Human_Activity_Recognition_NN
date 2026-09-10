# %% [markdown]
# # Multi-Scale CNN + Attention (inspired by DCAM-Net paper)
#
# New techniques not yet tried, borrowed from the DCAM-Net paper (Xu, Gao &
# Wang, 2025) but validated under OUR OWN honest subject-grouped protocol --
# the paper's reported 99.03% is on the 12-class HAPT dataset (includes
# postural transitions, not directly comparable to our 6-class problem), and
# its 5-fold CV description doesn't explicitly confirm subject-disjoint
# folds, so we treat their number as inspiration for architecture ideas, not
# a target to match exactly.
#
# New ingredients tested together here:
# 1. Multi-scale parallel convolutions (kernel sizes 5, 4, 3 simultaneously,
#    concatenated) instead of one fixed kernel size.
# 2. Self-attention layer over the temporal sequence, to weight which parts
#    of the signal matter most (e.g. suppress noisy axes during LAYING).
# 3. Residual connection for gradient flow.
# 4. Label smoothing (0.1) -- softens hard targets, helps with the
#    documented "transition window" label ambiguity in this dataset.
# 5. AdamW (weight decay handled correctly, unlike plain Adam) + cosine
#    annealing LR schedule with a linear warmup phase.
# 6. Gradient clipping (safety net against unstable updates).
#
# Per-subject normalization included from the start, given how consistently
# it's helped every model tried so far.

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
    X_norm = np.zeros_like(X, dtype=np.float32)
    for subj in np.unique(subjects):
        mask = subjects == subj
        subj_mean = X[mask].mean(axis=(0, 1), keepdims=True)
        subj_std = X[mask].std(axis=(0, 1), keepdims=True) + 1e-8
        X_norm[mask] = (X[mask] - subj_mean) / subj_std
    return X_norm

# %% [markdown]
# ## Model definition

# %%
class MultiScaleConvBlock(nn.Module):
    """Three parallel Conv1d branches (kernel sizes 5, 4, 3), concatenated --
    captures patterns at multiple time-scales simultaneously rather than
    sequentially through separate layers."""
    def __init__(self, in_ch, out_ch_per_branch=32):
        super().__init__()
        self.branch5 = nn.Conv1d(in_ch, out_ch_per_branch, kernel_size=5, padding=2)
        self.branch4 = nn.Conv1d(in_ch, out_ch_per_branch, kernel_size=4, padding=2)
        self.branch3 = nn.Conv1d(in_ch, out_ch_per_branch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_ch_per_branch * 3)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        b5 = self.branch5(x)
        b4 = self.branch4(x)[:, :, :b5.shape[-1]]  # kernel=4 with padding=2 gives +1 length, trim to match
        b3 = self.branch3(x)
        out = torch.cat([b5, b4, b3], dim=1)
        return self.act(self.bn(out))


class CNNAttention(nn.Module):
    def __init__(self, num_channels=9, num_classes=6, dropout=0.3):
        super().__init__()
        self.multiscale = MultiScaleConvBlock(num_channels, out_ch_per_branch=32)  # -> 96 channels
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(96, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.LeakyReLU(0.2)
        self.residual_proj = nn.Conv1d(96, 128, kernel_size=1)  # match channels for residual add
        self.pool2 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(dropout * 0.5)  # smaller dropout in conv layers

        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, dropout=dropout * 0.3)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, 128, 9) -> (batch, 9, 128)
        x = self.multiscale(x)  # (batch, 96, 128)
        x = self.pool1(x)  # (batch, 96, 64)

        residual = self.residual_proj(x)  # (batch, 128, 64)
        x = self.act2(self.bn2(self.conv2(x)))  # (batch, 128, 64)
        x = x + residual  # residual connection
        x = self.pool2(x)  # (batch, 128, 32)
        x = self.dropout1(x)

        x_seq = x.transpose(1, 2)  # (batch, 32, 128) for attention (seq_len, embed_dim)
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        pooled = attn_out.mean(dim=1)  # global average pool over the attended sequence

        return self.classifier(pooled)

# %%
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# %%
def train_model(X_tr_norm, y_tr, X_val_norm, y_val, seed=42, max_epochs=80, patience=15,
                 lr=1e-3, weight_decay=0.01, warmup_epochs=5, label_smoothing=0.1, grad_clip=1.0):
    X_tr_t = torch.tensor(X_tr_norm, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

    class_weights = torch.tensor(
        np.bincount(y_tr, minlength=NUM_CLASSES).sum() / (NUM_CLASSES * np.bincount(y_tr, minlength=NUM_CLASSES)),
        dtype=torch.float32,
    ).to(device)

    torch.manual_seed(seed)
    model = CNNAttention(num_channels=9, num_classes=NUM_CLASSES, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, max_epochs)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_f1 = f1_score(y_val, model(X_val_t).argmax(dim=1).cpu().numpy(), average="macro")
        if val_f1 > best_val_f1:
            best_val_f1, patience_counter = val_f1, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return best_val_f1

# %% [markdown]
# ## Quick check on Fold 0 before the full run (this architecture is heavier than the plain CNN)

# %%
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_raw, y_all, groups=subjects_all))
tr_idx0, val_idx0 = folds[0]

X_tr0 = per_subject_normalize_raw(X_raw[tr_idx0], subjects_all[tr_idx0])
X_val0 = per_subject_normalize_raw(X_raw[val_idx0], subjects_all[val_idx0])

quick_f1 = train_model(X_tr0, y_all[tr_idx0], X_val0, y_all[val_idx0])
print(f"Fold 0 quick check | val Macro F1 = {quick_f1:.4f}")

# %% [markdown]
# ## Full 5-fold GroupKFold validation

# %%
fold_scores = []
for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    X_tr = per_subject_normalize_raw(X_raw[tr_idx], subjects_all[tr_idx])
    X_val = per_subject_normalize_raw(X_raw[val_idx], subjects_all[val_idx])
    val_subjects = sorted(set(subjects_all[val_idx]))

    f1 = train_model(X_tr, y_all[tr_idx], X_val, y_all[val_idx])
    fold_scores.append(f1)
    print(f"Fold {fold_idx} (subjects {val_subjects}) | val Macro F1 = {f1:.4f}")

fold_scores = np.array(fold_scores)
print(f"\nMulti-scale CNN+Attention (per-subject norm) GroupKFold mean: {fold_scores.mean():.4f} +/- {fold_scores.std():.4f}")

# %%
nn_persubj_mean = 0.9731
plain_cnn_mean = 0.9623
print(f"\nNN (per-subject norm) GroupKFold mean:              {nn_persubj_mean:.4f}")
print(f"Plain 1D CNN (per-subject norm) GroupKFold mean:     {plain_cnn_mean:.4f}")
print(f"Multi-scale CNN+Attention GroupKFold mean:           {fold_scores.mean():.4f}")
if fold_scores.mean() > nn_persubj_mean:
    print("Multi-scale CNN+Attention BEATS the NN -- worth building into the final submission.")
elif fold_scores.mean() > plain_cnn_mean:
    print("Improvement over the plain CNN, but still below the NN.")
else:
    print("No improvement over the plain CNN.")