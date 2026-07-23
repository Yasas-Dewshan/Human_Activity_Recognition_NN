# %% [markdown]
# # HAR Competition — Part 7: Model Comparison (DT vs SVM vs NN)
#
# Loads the results already saved from each model's own script
# (processed/dt_results.pkl, svm_results.pkl, nn_results.pkl + nn_final_model.pt)
# and puts them side by side -- directly addresses the guideline requirement
# to "compare the behaviour and performance of the different approaches."
#
# Requires having already run the DT, SVM, and NN scripts (in that order) so
# processed/ contains their saved results.

# %%
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay

# %%
X_val = np.load("processed/X_val.npy")
y_val = np.load("processed/y_val.npy")
with open("processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# %%
# --- Decision Tree ---
with open("processed/dt_results.pkl", "rb") as f:
    dt_data = pickle.load(f)
dt_model = dt_data["model"]
y_pred_dt = dt_model.predict(X_val)

# --- SVM ---
with open("processed/svm_results.pkl", "rb") as f:
    svm_data = pickle.load(f)
svm_model = svm_data["model"]
y_pred_svm = svm_model.predict(X_val)

# --- Feedforward NN ---
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

with open("processed/nn_results.pkl", "rb") as f:
    nn_data = pickle.load(f)
nn_cfg = nn_data["summary"]

with open("processed/nn_scaler.pkl", "rb") as f:
    nn_scaler = pickle.load(f)

nn_model = MLP(X_val.shape[1], nn_cfg["hidden_sizes"], len(le.classes_), nn_cfg["dropout"], nn_cfg["batchnorm"])
nn_model.load_state_dict(torch.load("processed/nn_final_model.pt"))
nn_model.eval()

X_val_scaled = nn_scaler.transform(X_val)
with torch.no_grad():
    y_pred_nn = nn_model(torch.tensor(X_val_scaled, dtype=torch.float32)).argmax(dim=1).numpy()

# %% [markdown]
# ## 7.1 — Headline comparison table

# %%
comparison = pd.DataFrame([
    {"model": "Decision Tree", "val_macro_f1": f1_score(y_val, y_pred_dt, average="macro")},
    {"model": "SVM", "val_macro_f1": f1_score(y_val, y_pred_svm, average="macro")},
    {"model": "Feedforward NN", "val_macro_f1": f1_score(y_val, y_pred_nn, average="macro")},
]).sort_values("val_macro_f1", ascending=False)
print(comparison.to_string(index=False))

# %% [markdown]
# ## 7.2 — Per-class comparison
# Same validation set, three models -- shows exactly where each one struggles,
# not just an aggregate number.

# %%
per_class_rows = []
for model_name, y_pred in [("Decision Tree", y_pred_dt), ("SVM", y_pred_svm), ("Feedforward NN", y_pred_nn)]:
    report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
    for cls in le.classes_:
        per_class_rows.append({
            "model": model_name, "class": cls,
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1": report[cls]["f1-score"],
        })
per_class_df = pd.DataFrame(per_class_rows)
pivot_f1 = per_class_df.pivot(index="class", columns="model", values="f1")
pivot_f1 = pivot_f1[["Decision Tree", "SVM", "Feedforward NN"]].loc[le.classes_]
print("\nPer-class F1 score by model:")
print(pivot_f1.round(3).to_string())

# %% [markdown]
# ## 7.3 — Side-by-side confusion matrices

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (name, y_pred, cmap) in zip(
    axes,
    [("Decision Tree", y_pred_dt, "Blues"), ("SVM", y_pred_svm, "Greens"), ("Feedforward NN", y_pred_nn, "Purples")]
):
    ConfusionMatrixDisplay.from_predictions(
        y_val, y_pred, display_labels=le.classes_, xticks_rotation=45, ax=ax, cmap=cmap, colorbar=False
    )
    ax.set_title(name)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7.4 — Save comparison artifacts for the report

# %%
comparison.to_csv("processed/model_comparison.csv", index=False)
pivot_f1.to_csv("processed/per_class_f1_comparison.csv")
print("Saved processed/model_comparison.csv and processed/per_class_f1_comparison.csv")