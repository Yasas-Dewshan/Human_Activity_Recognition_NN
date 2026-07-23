# %% [markdown]
# # HAR Competition — Full Notebook
# Part 1: Setup & Data Loading | Part 2: EDA | Part 3: Preprocessing | Part 4a: Decision Tree
#
# IMPORTANT: This is ONE notebook, run top to bottom in a single session — the
# competition requires one reproducible Kaggle Notebook, not several separate
# ones. Every later cell depends on variables created by earlier cells, so
# always "Save & Run All" / run cells in order, never skip ahead.

# %%
# ===== PART 1: SETUP & DATA LOADING =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# On Kaggle, update this to the actual mounted path, e.g.:
# TRAIN_PATH = "/kaggle/input/<your-competition-name>/train.csv"
TRAIN_PATH = "train.csv"

train = pd.read_csv(TRAIN_PATH)
print("Train shape:", train.shape)
train.head()

# %%
print("Missing values in train:", train.isnull().sum().sum())
print("Duplicate rows in train:", train.duplicated().sum())
print("\nData types summary:")
print(train.dtypes.value_counts())

# %%
# ===== PART 2: EDA =====
class_counts = train["Activity"].value_counts()
print(class_counts)

plt.figure(figsize=(8, 5))
sns.barplot(x=class_counts.values, y=class_counts.index, orient="h")
plt.title("Activity Class Distribution (train)")
plt.xlabel("Count")
plt.tight_layout()
plt.show()

# %%
# 'subject' is NOT a sensor feature -- exclude it from feature_cols, keep
# separately for group-aware splitting/cross-validation.
feature_cols = [c for c in train.columns if c not in ["Activity", "id", "subject"]]
X = train[feature_cols]

print("Feature value range across all 561 features:")
print("Min:", X.values.min(), " Max:", X.values.max())

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(train["Activity"])

pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X)

plt.figure(figsize=(9, 7))
sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=train["Activity"], palette="tab10", alpha=0.6, s=15)
plt.title("PCA (2 components) of 561 HAR features, colored by Activity")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
print(f"Variance explained by 2 PCs: {pca_2d.explained_variance_ratio_.sum()*100:.1f}%")

# %%
pca_full = PCA(random_state=42)
pca_full.fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
for threshold in [0.90, 0.95, 0.99]:
    n_needed = np.argmax(cumvar >= threshold) + 1
    print(f"Components needed for {threshold*100:.0f}% variance: {n_needed}")

# %%
# ===== PART 3: PREPROCESSING =====
X_train_full = train[feature_cols].copy()
y_train_full = train["Activity"].copy()
y_train_encoded = le.fit_transform(y_train_full)

print("Label mapping:")
for i, label in enumerate(le.classes_):
    print(f"  {i} -> {label}")

# %%
# Group-based split by subject -- no subject appears in both train and val.
from sklearn.model_selection import GroupShuffleSplit

subjects = train["subject"].values
print("Number of unique subjects:", len(np.unique(subjects)))

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(gss.split(X_train_full, y_train_encoded, groups=subjects))

X_tr, X_val = X_train_full.iloc[tr_idx], X_train_full.iloc[val_idx]
y_tr, y_val = y_train_encoded[tr_idx], y_train_encoded[val_idx]
subj_tr, subj_val = subjects[tr_idx], subjects[val_idx]  # saved for group-aware CV later

print("Train split:", X_tr.shape, " Validation split:", X_val.shape)
print("Subject overlap (should be empty):", set(subj_tr) & set(subj_val))

# %%
# PCA-reduced feature set, fit on X_tr ONLY (no leakage into the transform)
pca_95 = PCA(n_components=0.95, random_state=42)
X_tr_pca = pca_95.fit_transform(X_tr)
X_val_pca = pca_95.transform(X_val)
print(f"PCA-reduced shape — train: {X_tr_pca.shape}, val: {X_val_pca.shape}")
print(f"Components kept: {pca_95.n_components_} (95% variance target)")

# %%
os.makedirs("processed", exist_ok=True)
np.save("processed/X_tr.npy", X_tr.values)
np.save("processed/X_val.npy", X_val.values)
np.save("processed/y_tr.npy", y_tr)
np.save("processed/y_val.npy", y_val)
np.save("processed/X_tr_pca.npy", X_tr_pca)
np.save("processed/X_val_pca.npy", X_val_pca)
np.save("processed/subj_tr.npy", subj_tr)
np.save("processed/subj_val.npy", subj_val)

with open("processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("processed/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
with open("processed/pca_95.pkl", "wb") as f:
    pickle.dump(pca_95, f)

print("Saved processed arrays to ./processed/")

# %% [markdown]
# ## Part 4a: Decision Tree Baseline (upgraded)
#
# Two upgrades over the first version:
# 1. **Raw (561) vs PCA (65) features compared directly.** Decision trees split
#    on one feature at a time along axis-aligned thresholds. PCA components are
#    rotated linear combinations of all original features, so there's no reason
#    a single PCA axis lines up with a threshold that cleanly separates classes.
#    Trees generally don't benefit from PCA the way distance/margin-based models
#    (SVM) or NNs might -- we test this rather than assume it.
# 2. **Group-aware cross-validation for hyperparameter selection.** A single
#    train/val split is noisy with only 21 subjects -- which specific subjects
#    land in validation can shift the "best" max_depth by chance. GroupKFold
#    (grouped by subject) averages performance across multiple splits, giving
#    a more reliable hyperparameter choice before we touch the true val set.

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay

# %%
# 4a.1 — Quick raw-vs-PCA comparison at a fixed, reasonable depth, to decide
# which feature set the tree should actually use going forward.
for feat_name, Xtr_, Xval_ in [
    ("Raw (561 features)", X_tr.values, X_val.values),
    ("PCA (65 components)", X_tr_pca, X_val_pca),
]:
    dt_probe = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_probe.fit(Xtr_, y_tr)
    val_f1 = f1_score(y_val, dt_probe.predict(Xval_), average="macro")
    print(f"{feat_name:25s} | max_depth=10 | val Macro F1 = {val_f1:.4f}")

# %%
# 4a.2 — Group-aware hyperparameter search on whichever feature set wins above.
# Update FEATURE_SET below based on the printed comparison.
FEATURE_SET = "raw"  # "raw" or "pca" -- set based on 4a.1 results

if FEATURE_SET == "raw":
    X_tr_use, X_val_use = X_tr.values, X_val.values
else:
    X_tr_use, X_val_use = X_tr_pca, X_val_pca

param_grid = {
    "max_depth": [8, 10, 13, 16, None],
    "min_samples_leaf": [1, 5, 10],
    "criterion": ["gini", "entropy"],
}
# 5 x 3 x 2 = 30 combos x 5 folds = 150 fits total (vs 240 before) -- still a
# solid sweep, faster to complete, and parallelized via n_jobs=-1 below.

gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X_tr_use, y_tr, groups=subj_tr))

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv_splits,
    n_jobs=-1,       # use all available CPU cores in parallel
    verbose=2,        # prints progress for each fold/combo -- confirms it's running
)
grid_search.fit(X_tr_use, y_tr)

grid_df = pd.DataFrame(grid_search.cv_results_)[
    ["param_max_depth", "param_min_samples_leaf", "param_criterion", "mean_test_score", "std_test_score"]
].sort_values("mean_test_score", ascending=False)
grid_df.columns = ["max_depth", "min_samples_leaf", "criterion", "cv_macro_f1_mean", "cv_macro_f1_std"]
print(grid_df.head(10))
print("\nBest params:", grid_search.best_params_)
print("Best CV Macro F1:", grid_search.best_score_)

# %%
# 4a.3 — Train final tree with best CV params, evaluate on the true held-out val set
best_params = grid_search.best_params_
print("Best params from group-CV search:", best_params)

dt_final = DecisionTreeClassifier(random_state=42, **best_params)
dt_final.fit(X_tr_use, y_tr)
y_pred = dt_final.predict(X_val_use)

val_macro_f1 = f1_score(y_val, y_pred, average="macro")
print(f"\nFinal Decision Tree — feature set: {FEATURE_SET}, params: {best_params}")
print(f"Validation Macro F1: {val_macro_f1:.4f}\n")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# %%
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred, display_labels=le.classes_, xticks_rotation=45, ax=ax, cmap="Blues")
plt.title(f"Decision Tree Confusion Matrix ({FEATURE_SET}, {best_params})")
plt.tight_layout()
plt.show()

# %%
dt_summary = {
    "model": "Decision Tree",
    "feature_set": FEATURE_SET,
    "best_hyperparams": best_params,
    "val_macro_f1": val_macro_f1,
}
print(dt_summary)

with open("processed/dt_results.pkl", "wb") as f:
    pickle.dump({"summary": dt_summary, "grid": grid_df, "model": dt_final}, f)

# %% [markdown]
# ## Part 4b: SVM Baseline
#
# SVM finds the hyperplane that maximizes the margin between classes. Unlike
# the decision tree, it's a distance/margin-based method -- feature *scale*
# matters a lot (a feature ranging -100 to 100 would dominate the margin
# calculation over one ranging -1 to 1). Good news: we already confirmed all
# 561 features are cleanly bounded in [-1, 1], so heavy rescaling shouldn't be
# necessary here -- but we verify rather than assume, same as before.
#
# SVM is also expensive to train on this many samples/features, especially
# with an RBF kernel -- so this search is intentionally smaller (3-fold CV,
# fewer combos) than the decision tree's.

# %%
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# %%
# 4b.1 — Quick raw-vs-PCA comparison (same reasoning as the tree: check, don't
# assume, which feature set works better for this specific model).
# Also test whether StandardScaler (mean=0, std=1) helps despite features
# already being in [-1, 1].
for feat_name, Xtr_, Xval_ in [
    ("Raw (561 features)", X_tr.values, X_val.values),
    ("PCA (65 components)", X_tr_pca, X_val_pca),
]:
    svm_probe = SVC(kernel="rbf", C=1.0, random_state=42)
    svm_probe.fit(Xtr_, y_tr)
    val_f1 = f1_score(y_val, svm_probe.predict(Xval_), average="macro")
    print(f"{feat_name:25s} | kernel=rbf, C=1.0 | val Macro F1 = {val_f1:.4f}")

# %%
# 4b.2 — Instead of pre-deciding scaled-vs-unscaled from a single fixed-setting
# comparison (which can miss the true best combo -- e.g. unscaled+rbf might
# beat scaled+linear even if scaled+rbf loses to unscaled+rbf), fold "scaler
# or no scaler" INTO the hyperparameter search itself via a Pipeline. This way
# CV directly compares all (scaler, kernel, C) combinations on equal footing.
FEATURE_SET_SVM_FINAL = "raw"  # locked in from 4b.1 -- raw beat PCA there

if FEATURE_SET_SVM_FINAL == "raw":
    X_tr_svm, X_val_svm = X_tr.values, X_val.values
else:
    X_tr_svm, X_val_svm = X_tr_pca, X_val_pca

# %%
# 4b.3 — Group-aware hyperparameter search over scaler + kernel + C (+ gamma
# for rbf). C = regularization strength (lower = wider margin, more tolerant
# of misclassification; higher = fits training data more tightly). gamma
# (rbf only) = how far a single point's influence reaches (low = smoother
# boundary, high = tighter fit, risk of overfitting).
from sklearn.pipeline import Pipeline

pipe_svm = Pipeline([
    ("scaler", "passthrough"),   # will be swapped for StandardScaler() in some grid combos
    ("svm", SVC(random_state=42)),
])

param_grid_svm = [
    {  # linear kernel: no gamma parameter
        "scaler": [StandardScaler(), "passthrough"],
        "svm__kernel": ["linear"],
        "svm__C": [0.1, 1, 10],
    },
    {  # rbf kernel: gamma matters
        "scaler": [StandardScaler(), "passthrough"],
        "svm__kernel": ["rbf"],
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale"],
    },
]
# (2 scalers x 3 C) x 2 kernels = 12 combos x 3 folds = 36 fits total.

gkf_svm = GroupKFold(n_splits=3)
cv_splits_svm = list(gkf_svm.split(X_tr_svm, y_tr, groups=subj_tr))

grid_search_svm = GridSearchCV(
    estimator=pipe_svm,
    param_grid=param_grid_svm,
    scoring="f1_macro",
    cv=cv_splits_svm,
    n_jobs=-1,
    verbose=2,
)
grid_search_svm.fit(X_tr_svm, y_tr)

cv_res = pd.DataFrame(grid_search_svm.cv_results_)
cv_res["scaler_label"] = cv_res["param_scaler"].apply(
    lambda s: "StandardScaler" if hasattr(s, "fit") else "none"
)
grid_df_svm = cv_res[
    ["scaler_label", "param_svm__kernel", "param_svm__C", "mean_test_score", "std_test_score"]
].sort_values("mean_test_score", ascending=False)
grid_df_svm.columns = ["scaler", "kernel", "C", "cv_macro_f1_mean", "cv_macro_f1_std"]
print(grid_df_svm.to_string(index=False))
print("\nBest params:", grid_search_svm.best_params_)
print("Best CV Macro F1:", grid_search_svm.best_score_)

# %%
# 4b.4 — Final SVM: best_estimator_ already includes the winning scaler choice
# (or 'passthrough') baked into the pipeline, so no manual scaling needed here.
svm_final = grid_search_svm.best_estimator_
y_pred_svm = svm_final.predict(X_val_svm)

val_macro_f1_svm = f1_score(y_val, y_pred_svm, average="macro")
print(f"\nFinal SVM — feature set: {FEATURE_SET_SVM_FINAL}, params: {grid_search_svm.best_params_}")
print(f"Validation Macro F1: {val_macro_f1_svm:.4f}\n")
print(classification_report(y_val, y_pred_svm, target_names=le.classes_))

# %%
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred_svm, display_labels=le.classes_, xticks_rotation=45, ax=ax, cmap="Greens")
plt.title(f"SVM Confusion Matrix ({FEATURE_SET_SVM_FINAL}, {grid_search_svm.best_params_})")
plt.tight_layout()
plt.show()

# %%
svm_summary = {
    "model": "SVM",
    "feature_set": FEATURE_SET_SVM_FINAL,
    "best_params": grid_search_svm.best_params_,
    "val_macro_f1": val_macro_f1_svm,
}
print(svm_summary)

with open("processed/svm_results.pkl", "wb") as f:
    pickle.dump({"summary": svm_summary, "grid": grid_df_svm, "model": svm_final}, f)