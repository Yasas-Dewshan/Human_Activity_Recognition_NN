# %% [markdown]
# # Extended Task 3: RNN (LSTM/GRU) on Raw Time-Series Signals
#
# Reformulates the problem as sequence modeling: instead of the 561
# hand-crafted features, the model sees the raw (128 timesteps x 9 channels)
# sensor windows directly and learns its own temporal representations.
#
# Uses the EXACT SAME subject-grouped split (random_state=42) as the DT/SVM/
# feedforward NN models, verified row-aligned with train.csv -- this makes
# the final comparison a fair, apples-to-apples test of "does end-to-end
# learning from raw signals offer an advantage" over hand-crafted features.

# %%
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

UCI_DIR = "UCI HAR Dataset"
KAGGLE_TRAIN_PATH = "train.csv"

# %% [markdown]
# ## Load raw signals + reuse the exact same subject-grouped split

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
print(f"Raw signals shape: {X_raw.shape}")

kaggle_train = pd.read_csv(KAGGLE_TRAIN_PATH)
le = LabelEncoder()
y_all = le.fit_transform(kaggle_train["Activity"])
subjects_all = kaggle_train["subject"].values
NUM_CLASSES = len(le.classes_)

# Same split as DT/SVM/NN -- verified row-aligned, so tr_idx/val_idx here
# select the exact same underlying samples as before.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(gss.split(X_raw, y_all, groups=subjects_all))

X_tr_raw, X_val_raw = X_raw[tr_idx], X_raw[val_idx]
y_tr, y_val = y_all[tr_idx], y_all[val_idx]
assert len(set(subjects_all[tr_idx]) & set(subjects_all[val_idx])) == 0

print(f"Train: {X_tr_raw.shape}, Val: {X_val_raw.shape}")

# %%
# Per-channel normalization -- fit mean/std on X_tr only (across all samples
# and timesteps for each of the 9 channels), same leakage-safe pattern used
# throughout this project.
channel_mean = X_tr_raw.mean(axis=(0, 1), keepdims=True)  # (1, 1, 9)
channel_std = X_tr_raw.std(axis=(0, 1), keepdims=True) + 1e-8

X_tr_norm = (X_tr_raw - channel_mean) / channel_std
X_val_norm = (X_val_raw - channel_mean) / channel_std

X_tr_t = torch.tensor(X_tr_norm, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)
X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

train_ds = TensorDataset(X_tr_t, y_tr_t)

class_counts = np.bincount(y_tr, minlength=NUM_CLASSES)
class_weights = class_counts.sum() / (NUM_CLASSES * class_counts)

# %% [markdown]
# ## RNN model definition

# %%
class RNNClassifier(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=1, rnn_type="lstm",
                 bidirectional=False, dropout=0.0, num_classes=6):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size, hidden_size, num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional, dropout=(dropout if num_layers > 1 else 0.0),
        )
        direction_mult = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * direction_mult, num_classes)

    def forward(self, x):
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)

        # h_n shape: (num_layers * num_directions, batch, hidden_size) --
        # extract the LAST layer's final hidden state(s)
        if self.bidirectional:
            h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
            last = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=-1)
        else:
            h_n = h_n.view(self.num_layers, 1, -1, self.hidden_size)
            last = h_n[-1, 0]

        return self.fc(self.dropout(last))

# %%
def train_rnn(cfg, class_weights=None, max_epochs=60, patience=10, seed=42, batch_size=64, lr=1e-3):
    torch.manual_seed(seed)
    model = RNNClassifier(
        input_size=9, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
        rnn_type=cfg["rnn_type"], bidirectional=cfg.get("bidirectional", False),
        dropout=cfg.get("dropout", 0.0), num_classes=NUM_CLASSES,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)

    best_val_f1, best_state, patience_counter = -1, None, 0
    history = []
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
            val_pred = model(X_val_t).argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, val_pred, average="macro")
        scheduler.step(val_f1)
        history.append({"epoch": epoch, "val_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_f1, pd.DataFrame(history)

# %% [markdown]
# ## Investigation 1: LSTM vs GRU, hidden size (depth/capacity)
# Kept modest given RNN training is slower than the feedforward NN -- a
# focused sweep rather than an exhaustive one.

# %%
rnn_configs = {
    "GRU, hidden=32":          {"rnn_type": "gru",  "hidden_size": 32, "num_layers": 1},
    "GRU, hidden=64":          {"rnn_type": "gru",  "hidden_size": 64, "num_layers": 1},
    "LSTM, hidden=64":         {"rnn_type": "lstm", "hidden_size": 64, "num_layers": 1},
    "LSTM, hidden=128":        {"rnn_type": "lstm", "hidden_size": 128, "num_layers": 1},
    "LSTM, hidden=64, 2 layers": {"rnn_type": "lstm", "hidden_size": 64, "num_layers": 2},
}

rnn_results = []
for name, cfg in rnn_configs.items():
    _, val_f1, _ = train_rnn(cfg, class_weights=class_weights)
    rnn_results.append({"config": name, "val_macro_f1": val_f1})
    print(f"{name:30s} | val Macro F1 = {val_f1:.4f}")

rnn_df = pd.DataFrame(rnn_results).sort_values("val_macro_f1", ascending=False)
best_rnn_name = rnn_df.iloc[0]["config"]
best_rnn_cfg = rnn_configs[best_rnn_name]
print(f"\nBest RNN config: {best_rnn_name}")

# %% [markdown]
# ## Investigation 2: dropout regularization on the winning architecture

# %%
dropout_results = []
for dropout in [0.0, 0.3, 0.5]:
    cfg = {**best_rnn_cfg, "dropout": dropout}
    _, val_f1, _ = train_rnn(cfg, class_weights=class_weights)
    dropout_results.append({"dropout": dropout, "val_macro_f1": val_f1})
    print(f"dropout={dropout} | val Macro F1 = {val_f1:.4f}")

dropout_df = pd.DataFrame(dropout_results).sort_values("val_macro_f1", ascending=False)
best_dropout = float(dropout_df.iloc[0]["dropout"])
print(f"\nBest dropout: {best_dropout}")

# %% [markdown]
# ## Final RNN model + detailed evaluation

# %%
final_cfg = {**best_rnn_cfg, "dropout": best_dropout}
final_rnn, final_val_f1, final_history = train_rnn(final_cfg, class_weights=class_weights, max_epochs=100, patience=15)

print(f"\nFinal RNN config: {final_cfg}")
print(f"Final validation Macro F1: {final_val_f1:.4f}")

final_rnn.eval()
with torch.no_grad():
    y_pred_rnn = final_rnn(X_val_t).argmax(dim=1).cpu().numpy()

print(classification_report(y_val, y_pred_rnn, target_names=le.classes_))

# %%
fig, ax = plt.subplots(figsize=(7, 7))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred_rnn, display_labels=le.classes_, xticks_rotation=45, ax=ax, cmap="Oranges")
plt.title(f"RNN ({final_cfg['rnn_type'].upper()}) Confusion Matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Comparison: RNN (raw signals) vs hand-crafted feature models
# Same X_val, same subjects, same underlying samples -- a fair test of
# whether end-to-end learning from raw signals offers an advantage.

# %%
# Fill in your actual DT/SVM/NN val Macro F1 scores from earlier runs, on the
# SAME split, for a direct comparison:
prior_scores = {
    "Decision Tree (hand-crafted)": 0.8697,
    "SVM (hand-crafted)": 0.9581,
    "Feedforward NN (hand-crafted)": 0.9746,  # or your latest confirmed number
}

comparison = pd.DataFrame(
    [{"model": k, "val_macro_f1": v} for k, v in prior_scores.items()]
    + [{"model": f"RNN ({final_cfg['rnn_type'].upper()}, raw signals)", "val_macro_f1": final_val_f1}]
).sort_values("val_macro_f1", ascending=False)
print(comparison.to_string(index=False))

comparison.to_csv("processed/rnn_vs_handcrafted_comparison.csv", index=False)
print("\nSaved processed/rnn_vs_handcrafted_comparison.csv")