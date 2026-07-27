# %% [markdown]
# # XGBoost Test: A Genuinely Different Algorithm Family
#
# Gradient boosting (sequentially-corrected trees) is a fundamentally
# different approach from DT/SVM/NN/RNN tried so far, and often the
# strongest performer on structured/tabular data like these 561 hand-crafted
# features. Tests raw vs. per-subject normalized features (mirroring the
# earlier DT/SVM checks), then a small hyperparameter sweep, all validated
# via the same GroupKFold protocol used throughout.

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

TRAIN_PATH = "train.csv"

# %%
train = pd.read_csv(TRAIN_PATH)
feature_cols = [c for c in train.columns if c not in ["Activity", "id", "subject"]]

le = LabelEncoder()
y_all = le.fit_transform(train["Activity"])
X_all = train[feature_cols].values
subjects_all = train["subject"].values
NUM_CLASSES = len(le.classes_)

# %%
def per_subject_normalize(X, subjects):
    X_norm = np.zeros_like(X, dtype=np.float64)
    for subj in np.unique(subjects):
        mask = subjects == subj
        subj_mean = X[mask].mean(axis=0)
        subj_std = X[mask].std(axis=0) + 1e-8
        X_norm[mask] = (X[mask] - subj_mean) / subj_std
    return X_norm

# %% [markdown]
# ## Step 1: raw vs per-subject normalized features, default-ish XGBoost params

# %%
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_all, y_all, groups=subjects_all))

default_params = dict(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42,
                       objective="multi:softmax", num_class=NUM_CLASSES, n_jobs=-1, eval_metric="mlogloss")

for feat_label, normalize in [("Raw features", False), ("Per-subject normalized", True)]:
    scores = []
    for tr_idx, val_idx in folds:
        if normalize:
            X_tr = per_subject_normalize(X_all[tr_idx], subjects_all[tr_idx])
            X_val = per_subject_normalize(X_all[val_idx], subjects_all[val_idx])
        else:
            X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        model = XGBClassifier(**default_params)
        model.fit(X_tr, y_all[tr_idx])
        pred = model.predict(X_val)
        scores.append(f1_score(y_all[val_idx], pred, average="macro"))
    scores = np.array(scores)
    print(f"{feat_label:25s} | GroupKFold mean = {scores.mean():.4f} +/- {scores.std():.4f} "
          f"(min {scores.min():.4f}, max {scores.max():.4f})")

# %% [markdown]
# ## Step 2: small hyperparameter sweep on the winning feature set
# Update USE_PERSUBJ below based on Step 1's result before running this.

# %%
USE_PERSUBJ = True  # set based on Step 1 output

param_grid = [
    dict(n_estimators=200, max_depth=4, learning_rate=0.1),
    dict(n_estimators=300, max_depth=6, learning_rate=0.1),
    dict(n_estimators=500, max_depth=6, learning_rate=0.05),
    dict(n_estimators=300, max_depth=8, learning_rate=0.05),
    dict(n_estimators=500, max_depth=4, learning_rate=0.1),
]

sweep_results = []
for params in param_grid:
    scores = []
    for tr_idx, val_idx in folds:
        if USE_PERSUBJ:
            X_tr = per_subject_normalize(X_all[tr_idx], subjects_all[tr_idx])
            X_val = per_subject_normalize(X_all[val_idx], subjects_all[val_idx])
        else:
            X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        model = XGBClassifier(**params, random_state=42, objective="multi:softmax",
                               num_class=NUM_CLASSES, n_jobs=-1, eval_metric="mlogloss")
        model.fit(X_tr, y_all[tr_idx])
        pred = model.predict(X_val)
        scores.append(f1_score(y_all[val_idx], pred, average="macro"))
    scores = np.array(scores)
    sweep_results.append({**params, "mean_f1": scores.mean(), "std_f1": scores.std()})
    print(f"{params} | GroupKFold mean = {scores.mean():.4f} +/- {scores.std():.4f}")

sweep_df = pd.DataFrame(sweep_results).sort_values("mean_f1", ascending=False)
print("\nBest config:")
print(sweep_df.iloc[0])

# %% [markdown]
# ## Comparison against the NN

# %%
nn_persubj_mean = 0.9731  # already confirmed
best_xgb_mean = sweep_df.iloc[0]["mean_f1"]
print(f"\nNN (per-subject norm) GroupKFold mean:      {nn_persubj_mean:.4f}")
print(f"XGBoost (best config) GroupKFold mean:       {best_xgb_mean:.4f}")
if best_xgb_mean > nn_persubj_mean:
    print("XGBoost BEATS the NN -- worth building into the final submission.")
else:
    print("XGBoost does not beat the NN -- stick with the NN as primary model.")