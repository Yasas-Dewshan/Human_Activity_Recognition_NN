# %% [markdown]
# # HAR Competition — Part 6 (final): Train NN with confirmed hyperparameters
#
# Uses the hyperparameters already confirmed by the staged sweep:
#   - Architecture: [256, 128] (2 hidden layers)
#   - Dropout: 0.5
#   - Weight decay: 0.0
#   - BatchNorm: True
# Learning rate / batch size / optimizer use sensible defaults (Adam, 1e-3,
# 64) since those specific sweeps haven't been run yet -- update the CONFIG
# block below if you've already found better values for those.
#
# Trains across 5 seeds (same reasoning as before -- NN results vary run to
# run, so we report mean ± std rather than a single lucky number) and
# evaluates the best seed's model in detail: classification report +
# confusion matrix, so you can see exactly how it performs per class.

# %%
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
CONFIG = {
    "hidden_sizes": [256, 128],
    "dropout": 0.5,
    "weight_decay": 0.0,
    "batchnorm": True,
    "lr": 1e-3,
    "batch_size": 64,
    "optimizer_name": "adam",
}
print("Training with config:", CONFIG)

# %%
X_tr = np.load("processed/X_tr.npy")
X_val = np.load("processed/X_val.npy")
y_tr = np.load("processed/y_tr.npy")
y_val = np.load("processed/y_val.npy")

with open("processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

NUM_CLASSES = len(le.classes_)
INPUT_DIM = X_tr.shape[1]

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

X_tr_t = torch.tensor(X_tr_scaled, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)

train_ds = TensorDataset(X_tr_t, y_tr_t)

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


def make_optimizer(name, params, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def train_model(cfg, max_epochs=100, patience=12, seed=42):
    torch.manual_seed(seed)
    model = MLP(INPUT_DIM, cfg["hidden_sizes"], NUM_CLASSES, cfg["dropout"], cfg["batchnorm"]).to(device)
    optimizer = make_optimizer(cfg["optimizer_name"], model.parameters(), cfg["lr"], cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=gen)

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
            train_pred = model(X_tr_t.to(device)).argmax(dim=1).cpu().numpy()
            train_f1 = f1_score(y_tr, train_pred, average="macro")
        history.append({"epoch": epoch, "train_f1": train_f1, "val_f1": val_f1})

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

# %%
# Train across 5 seeds, report mean ± std, keep the best-performing seed's model
seeds = (42, 43, 44, 45, 46)
models, f1s, histories = [], [], []
for s in seeds:
    model, val_f1, hist = train_model(CONFIG, seed=s)
    models.append(model)
    f1s.append(val_f1)
    histories.append(hist)
    print(f"seed={s} | val Macro F1 = {val_f1:.4f}")

f1s = np.array(f1s)
best_idx = f1s.argmax()
final_model = models[best_idx]
final_history = histories[best_idx]

print(f"\nMean val Macro F1 across {len(seeds)} seeds: {f1s.mean():.4f} ± {f1s.std():.4f}")
print(f"Best seed: {seeds[best_idx]} ({f1s[best_idx]:.4f}) -- used for detailed evaluation below")

# %%
# Detailed evaluation of the best seed's model
final_model.eval()
with torch.no_grad():
    y_pred = final_model(X_val_t).argmax(dim=1).cpu().numpy()

print(classification_report(y_val, y_pred, target_names=le.classes_))

# %%
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred, display_labels=le.classes_, xticks_rotation=45, ax=ax, cmap="Purples")
plt.title(f"Feedforward NN Confusion Matrix (seed {seeds[best_idx]})")
plt.tight_layout()
plt.show()

# %%
# Train vs val curve for the best seed -- useful evidence of over/underfitting
plt.figure(figsize=(9, 5))
plt.plot(final_history["epoch"], final_history["train_f1"], label="Train Macro F1")
plt.plot(final_history["epoch"], final_history["val_f1"], label="Validation Macro F1")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("Feedforward NN: Train vs Validation Macro F1 over training")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
nn_summary = {
    "model": "Feedforward NN",
    "feature_set": "raw",
    **CONFIG,
    "val_macro_f1_mean": f1s.mean(),
    "val_macro_f1_std": f1s.std(),
    "val_macro_f1_best_seed": f1s[best_idx],
}
print(nn_summary)

torch.save(final_model.state_dict(), "processed/nn_final_model.pt")
with open("processed/nn_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("processed/nn_results.pkl", "wb") as f:
    pickle.dump({"summary": nn_summary, "final_history": final_history}, f)

print("Saved NN model + scaler + results to processed/")