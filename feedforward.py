# %% [markdown]
# # HAR Competition — Part 6: Feedforward Neural Network
#
# Uses the same raw 561 features as the DT/SVM baselines (per guideline
# requirement: "train a feedforward neural network using the same 561
# features"). Investigates depth, regularization, and normalization in three
# staged sweeps, each building on the best result from the previous stage.
#
# Requires: processed/X_tr.npy, X_val.npy, y_tr.npy, y_val.npy, subj_tr.npy
# (produced earlier in the main notebook).

# %%
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# %%
X_tr = np.load("processed/X_tr.npy")
X_val = np.load("processed/X_val.npy")
y_tr = np.load("processed/y_tr.npy")
y_val = np.load("processed/y_val.npy")

with open("processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

NUM_CLASSES = len(le.classes_)
INPUT_DIM = X_tr.shape[1]
print(f"Train: {X_tr.shape}, Val: {X_val.shape}, classes: {NUM_CLASSES}")

# %%
# Input normalization -- fit on X_tr only, same leakage-safe pattern as
# before. Neural nets trained with gradient descent generally benefit from
# zero-mean/unit-variance inputs, even though these features are already
# bounded in [-1, 1] -- bounded range isn't the same as consistent variance
# across all 561 features.
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

X_tr_t = torch.tensor(X_tr_scaled, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_np = y_val  # keep as numpy for sklearn's f1_score

train_ds = TensorDataset(X_tr_t, y_tr_t)

# %%
class MLP(nn.Module):
    """Configurable feedforward NN -- depth, dropout, and batchnorm are all
    tunable so the same class serves every experiment below."""
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

# %%
def train_model(hidden_sizes, dropout=0.0, batchnorm=False, weight_decay=0.0,
                 lr=1e-3, max_epochs=60, patience=8, seed=42, verbose=False):
    """Trains one MLP config with early stopping on validation Macro F1.
    Returns the best model state, best val F1, and per-epoch history."""
    torch.manual_seed(seed)
    model = MLP(INPUT_DIM, hidden_sizes, NUM_CLASSES, dropout, batchnorm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Fresh, seeded DataLoader per call -- otherwise shuffling order differs
    # between calls even at the "same" config, adding noise on top of the
    # weight-init randomness we're already controlling for.
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=gen)

    best_val_f1 = -1
    best_state = None
    patience_counter = 0
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
            val_f1 = f1_score(y_val_np, val_pred, average="macro")
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

        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d} | train F1 {train_f1:.4f} | val F1 {val_f1:.4f}")

    model.load_state_dict(best_state)
    return model, best_val_f1, pd.DataFrame(history)


def eval_config_multiseed(hidden_sizes, dropout=0.0, batchnorm=False, weight_decay=0.0,
                            seeds=(42, 43, 44), **kwargs):
    """Trains the same config across multiple seeds, returns mean/std val F1 --
    a single run can get lucky or unlucky due to random init/shuffling, so we
    average to get a stable estimate before comparing configs against each other."""
    f1s = [train_model(hidden_sizes, dropout, batchnorm, weight_decay, seed=s, **kwargs)[1] for s in seeds]
    return np.mean(f1s), np.std(f1s)

# %% [markdown]
# ## 6.1 — Depth investigation
# Too shallow -> can't capture enough structure. Too deep (relative to this
# amount of data) -> more parameters to overfit with, longer training, often
# no real accuracy gain. We sweep a range and let validation Macro F1 decide.

# %%
depth_configs = {
    "1 layer  [128]":         [128],
    "1 layer  [256]":         [256],
    "2 layers [256,128]":     [256, 128],
    "3 layers [256,128,64]":  [256, 128, 64],
    "4 layers [512,256,128,64]": [512, 256, 128, 64],
}

depth_results = []
for name, hidden_sizes in depth_configs.items():
    mean_f1, std_f1 = eval_config_multiseed(hidden_sizes)
    depth_results.append({"config": name, "val_macro_f1_mean": mean_f1, "val_macro_f1_std": std_f1})
    print(f"{name:30s} | val Macro F1 = {mean_f1:.4f} ± {std_f1:.4f}")

depth_df = pd.DataFrame(depth_results).sort_values("val_macro_f1_mean", ascending=False)
best_depth_name = depth_df.iloc[0]["config"]
best_hidden_sizes = depth_configs[best_depth_name]
print(f"\nBest depth config: {best_depth_name}")

# %% [markdown]
# ## 6.2 — Regularization investigation (dropout + weight decay)
# Using the winning architecture from 6.1. Dropout randomly zeroes a fraction
# of neurons each training step, forcing the network not to rely too heavily
# on any single one -- reduces overfitting. Weight decay (L2 penalty) shrinks
# weight magnitudes, favoring simpler solutions.

# %%
reg_configs = [
    {"dropout": 0.0, "weight_decay": 0.0},
    {"dropout": 0.3, "weight_decay": 0.0},
    {"dropout": 0.5, "weight_decay": 0.0},
    {"dropout": 0.0, "weight_decay": 1e-4},
    {"dropout": 0.3, "weight_decay": 1e-4},
]

reg_results = []
for cfg in reg_configs:
    mean_f1, std_f1 = eval_config_multiseed(best_hidden_sizes, dropout=cfg["dropout"], weight_decay=cfg["weight_decay"])
    reg_results.append({**cfg, "val_macro_f1_mean": mean_f1, "val_macro_f1_std": std_f1})
    print(f"dropout={cfg['dropout']}, weight_decay={cfg['weight_decay']} | val Macro F1 = {mean_f1:.4f} ± {std_f1:.4f}")

reg_df = pd.DataFrame(reg_results).sort_values("val_macro_f1_mean", ascending=False)
best_reg = reg_df.iloc[0][["dropout", "weight_decay"]].to_dict()
print(f"\nBest regularization: {best_reg}")

# %% [markdown]
# ## 6.3 — Normalization investigation (BatchNorm)
# Using the winning architecture + regularization from 6.1/6.2. BatchNorm
# normalizes each layer's activations during training, which typically speeds
# up and stabilizes convergence -- tested here rather than assumed.

# %%
norm_results = []
for use_bn in [False, True]:
    mean_f1, std_f1 = eval_config_multiseed(best_hidden_sizes, dropout=best_reg["dropout"],
                                              weight_decay=best_reg["weight_decay"], batchnorm=use_bn)
    norm_results.append({"batchnorm": use_bn, "val_macro_f1_mean": mean_f1, "val_macro_f1_std": std_f1})
    print(f"batchnorm={use_bn} | val Macro F1 = {mean_f1:.4f} ± {std_f1:.4f}")

norm_df = pd.DataFrame(norm_results).sort_values("val_macro_f1_mean", ascending=False)
best_batchnorm = bool(norm_df.iloc[0]["batchnorm"])
print(f"\nBest normalization setting: batchnorm={best_batchnorm}")

# %% [markdown]
# ## 6.4 — Final model: best depth + regularization + normalization together

# %%
# Run the final winning config across several seeds -- report mean ± std for
# an honest final number, and keep the single best-performing seed's model
# (a reasonable, common choice: it's a real trained model, not an average of
# weights, and its score is a genuine draw from the same distribution we just
# reported).
final_seeds = (42, 43, 44, 45, 46)
final_models, final_f1s = [], []
for s in final_seeds:
    model, val_f1, hist = train_model(
        best_hidden_sizes, dropout=best_reg["dropout"], weight_decay=best_reg["weight_decay"],
        batchnorm=best_batchnorm, max_epochs=100, patience=12, seed=s,
    )
    final_models.append(model)
    final_f1s.append(val_f1)
    if val_f1 == max(final_f1s):
        final_history = hist

final_f1s = np.array(final_f1s)
best_idx = final_f1s.argmax()
final_model = final_models[best_idx]
final_val_f1_mean = final_f1s.mean()
final_val_f1_std = final_f1s.std()

print(f"\nFinal NN config: hidden_sizes={best_hidden_sizes}, dropout={best_reg['dropout']}, "
      f"weight_decay={best_reg['weight_decay']}, batchnorm={best_batchnorm}")
print(f"Validation Macro F1 across {len(final_seeds)} seeds: {final_val_f1_mean:.4f} ± {final_val_f1_std:.4f}")
print(f"Best single seed ({final_seeds[best_idx]}): {final_f1s[best_idx]:.4f} -- this is the saved model")

# %%
nn_summary = {
    "model": "Feedforward NN",
    "feature_set": "raw",
    "hidden_sizes": best_hidden_sizes,
    "dropout": best_reg["dropout"],
    "weight_decay": best_reg["weight_decay"],
    "batchnorm": best_batchnorm,
    "val_macro_f1_mean": final_val_f1_mean,
    "val_macro_f1_std": final_val_f1_std,
    "val_macro_f1_best_seed": final_f1s[best_idx],
}
print(nn_summary)

torch.save(final_model.state_dict(), "processed/nn_final_model.pt")
with open("processed/nn_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("processed/nn_results.pkl", "wb") as f:
    pickle.dump({
        "summary": nn_summary,
        "depth_sweep": depth_df,
        "reg_sweep": reg_df,
        "norm_sweep": norm_df,
        "final_history": final_history,
    }, f)

print("Saved NN model + scaler + results to processed/")