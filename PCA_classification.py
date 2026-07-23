# %% [markdown]
# # HAR Competition — Part 1, 2, 3: Setup, Data Loading, EDA, Preprocessing
#
# NOTE: test.csv isn't available yet, so everything here runs off train.csv only.
# We carve our own validation set out of train.csv to evaluate models until the
# real test.csv is released — at that point we just point Part 4/5 scripts at it.
#
# Run each `# %%` block as a cell in VS Code (click "Run Cell" above each block,
# or use Shift+Enter). Requires the Jupyter + Python extensions in VS Code.

# %%
# ===== PART 1: SETUP & DATA LOADING =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Update this path to wherever you downloaded train.csv
TRAIN_PATH = "train.csv"

train = pd.read_csv(TRAIN_PATH)

print("Train shape:", train.shape)
train.head()

# %%
# Basic sanity checks
print("Train columns (first 10):", train.columns[:10].tolist())
print("Train columns (last 10):", train.columns[-10:].tolist())
print("\nDoes train have an 'id' column?", "id" in train.columns)
print("Target column present in train:", "Activity" in train.columns)

# %%
# Missing values check
print("Missing values in train:", train.isnull().sum().sum())

# Duplicate rows check
print("Duplicate rows in train:", train.duplicated().sum())

# Data types — should be almost entirely float64 (the 561 features) + object (Activity) + int (id, subject if present)
print("\nData types summary:")
print(train.dtypes.value_counts())

# %%
# ===== PART 2: EDA =====

# 2.1 — Class distribution (important since we're scored on Macro F1,
# which treats every class equally regardless of how many samples it has)
class_counts = train["Activity"].value_counts()
print(class_counts)

plt.figure(figsize=(8, 5))
sns.barplot(x=class_counts.values, y=class_counts.index, orient="h")
plt.title("Activity Class Distribution (train)")
plt.xlabel("Count")
plt.tight_layout()
plt.show()

# %%
# 2.2 — Feature scale check
# HAR features are typically pre-normalized to roughly [-1, 1] (not strictly true
# for every feature, as we saw — max hit 30.0 on some mad()/max() jerk features).
# IMPORTANT: 'subject' is NOT a sensor feature — it's an ID for which of the 30
# volunteers performed the activity. It must be excluded from feature_cols, and
# kept separately for grouping the train/val split (see Part 3.2).
feature_cols = [c for c in train.columns if c not in ["Activity", "id", "subject"]]
X = train[feature_cols]

print("Feature value range across all 561 features:")
print("Min:", X.values.min(), " Max:", X.values.max())
print("\nPer-feature min/max summary (first 5 features):")
print(X.describe().T[["min", "max"]].head())

# %%
# 2.3 — Dimensionality reduction visualization (PCA)
# With 561 features, a 2D projection helps sanity-check that classes are separable
# before we even train a model.
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(train["Activity"])

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(9, 7))
sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1],
    hue=train["Activity"], palette="tab10", alpha=0.6, s=15
)
plt.title("PCA (2 components) of 561 HAR features, colored by Activity")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

print(f"Variance explained by 2 PCs: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# %%
# 2.3b — How many PCs should we actually keep?
# Fit PCA with ALL components, then plot cumulative variance explained.
# This tells us how much of the original 561-dim structure we retain
# as we keep more components — the "scree plot" approach.
pca_full = PCA(random_state=42)
pca_full.fit(X)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(cumvar) + 1), cumvar, marker='.', markersize=3)
for threshold in [0.90, 0.95, 0.99]:
    n_needed = np.argmax(cumvar >= threshold) + 1
    plt.axhline(threshold, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(n_needed, color='gray', linestyle='--', linewidth=0.8)
    print(f"Components needed for {threshold*100:.0f}% variance: {n_needed}")

plt.xlabel("Number of principal components")
plt.ylabel("Cumulative variance explained")
plt.title("PCA: Cumulative Explained Variance vs. Number of Components")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Interpretation guide (fill in once you've run this):
# - If ~90-95% variance needs only a small fraction of 561 components,
#   that's a strong case for using PCA as a preprocessing/dimensionality
#   reduction step (faster SVM training, less risk of overfitting for the
#   decision tree, potentially fewer input weights for the NN).
# - If it takes close to 561 components to reach 90%, the features don't
#   compress well — variance is spread thinly across many dimensions, and
#   PCA reduction may lose meaningful information. In that case, PCA stays
#   useful for visualization/EDA but we'd train the actual models on all
#   561 original features rather than a reduced PC set.

# %%
# 2.4 — Quick look at whether static (LAYING/SITTING/STANDING) vs
# dynamic (WALKING*) activities separate cleanly — a known pattern in this dataset.
# Just eyeballing the PCA plot above should already hint at this:
# static activities tend to cluster together, dynamic activities cluster separately,
# with the most confusion typically between SITTING vs STANDING, and between
# WALKING_UPSTAIRS vs WALKING_DOWNSTAIRS.

# %%
# ===== PART 3: PREPROCESSING =====

# 3.1 — Separate features/target, encode labels
X_train_full = train[feature_cols].copy()
y_train_full = train["Activity"].copy()

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_full)

print("Label mapping:")
for i, label in enumerate(le.classes_):
    print(f"  {i} -> {label}")

# %%
# 3.2 — Group-based train/validation split, grouped by 'subject'.
#
# WHY NOT a plain random/stratified split: rows from the same subject are highly
# correlated (same person's sensor signal, just different time windows), so a
# random split leaks subject identity between train and val — the model can partly
# "recognize the person" instead of "recognize the activity", which inflates
# validation Macro F1 above what you'd realistically see on test.csv (which likely
# contains subjects never seen in train.csv, mirroring the original UCI HAR design
# of 21 train subjects / 9 test subjects).
#
# GroupShuffleSplit guarantees no subject appears in both X_tr and X_val.
from sklearn.model_selection import GroupShuffleSplit

subjects = train["subject"].values
print("Number of unique subjects:", len(np.unique(subjects)))

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(gss.split(X_train_full, y_train_encoded, groups=subjects))

X_tr, X_val = X_train_full.iloc[tr_idx], X_train_full.iloc[val_idx]
y_tr, y_val = y_train_encoded[tr_idx], y_train_encoded[val_idx]

print("Train split:", X_tr.shape, " Validation split:", X_val.shape)

# Confirm no subject overlap
train_subjects = set(subjects[tr_idx])
val_subjects = set(subjects[val_idx])
print("Subject overlap between train/val (should be empty set):", train_subjects & val_subjects)

# Class balance won't match as precisely as a stratified split would, since we're
# constrained by whole subjects — that's expected and fine, just worth checking
# it isn't wildly skewed.
print("\nClass balance (train split):")
print(pd.Series(y_tr).value_counts(normalize=True).sort_index())
print("\nClass balance (val split):")
print(pd.Series(y_val).value_counts(normalize=True).sort_index())

# %%
# 3.3 — Optional: PCA-reduced feature set (95% variance retained, ~67 components)
# for use alongside the full 561-feature set — useful for faster SVM training
# and as a point of comparison in your "compare approaches" write-up.
#
# CRITICAL: fit PCA on X_tr ONLY, then transform X_val with that same fitted
# transformer. Fitting PCA on X_tr+X_val combined (or worse, the full train.csv
# before splitting) would leak validation-set distribution info into training,
# the same category of mistake as the subject-leakage issue from before.
pca_95 = PCA(n_components=0.95, random_state=42)
X_tr_pca = pca_95.fit_transform(X_tr)
X_val_pca = pca_95.transform(X_val)  # transform only, no fit

print(f"PCA-reduced shape — train: {X_tr_pca.shape}, val: {X_val_pca.shape}")
print(f"Components kept: {pca_95.n_components_} (95% variance target)")

# %%
# 3.4 — Save processed splits (both full 561-feature and PCA-reduced versions)
# so later scripts (baselines, NN) don't need to redo this every time. When
# test.csv arrives, add a couple lines to load it, extract feature_cols, and
# np.save it the same way (both raw and pca_95.transform() versions).
import os
import pickle

os.makedirs("processed", exist_ok=True)
np.save("processed/X_tr.npy", X_tr.values)
np.save("processed/X_val.npy", X_val.values)
np.save("processed/y_tr.npy", y_tr)
np.save("processed/y_val.npy", y_val)
np.save("processed/X_tr_pca.npy", X_tr_pca)
np.save("processed/X_val_pca.npy", X_val_pca)

with open("processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("processed/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

with open("processed/pca_95.pkl", "wb") as f:
    pickle.dump(pca_95, f)  # save the fitted transformer so test.csv can reuse it later

print("Saved processed arrays (raw + PCA-reduced) to ./processed/")
print("(test.csv not yet available — rerun Part 3.4 with test data once released)")