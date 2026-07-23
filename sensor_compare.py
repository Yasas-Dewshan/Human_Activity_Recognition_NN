# %% [markdown]
# # Extended Task 4 — Sensor Contribution Investigation
#
# Question: how much performance is lost when only accelerometer or only
# gyroscope features are used? What does this reveal about the
# complementarity of the two sensors?
#
# Approach: split the 561 feature names into three groups based on naming
# convention (tBodyAcc-*, tGravityAcc-*, fBodyAcc-* etc. contain "Acc";
# tBodyGyro-*, fBodyGyro-* etc. contain "Gyro"). A handful of features (the
# angle(...) ones) combine information from both sensors or reference gravity
# directly -- these are kept as a separate "other" group rather than force-
# assigned to one sensor, since mislabeling them would corrupt the comparison.
#
# Uses the already-confirmed best SVM config (unscaled, rbf, C=10) since it's
# fast to retrain multiple times and already a strong, well-understood model
# -- appropriate for isolating the effect of feature *selection* rather than
# re-introducing model-choice as a variable.

# %%
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report

TRAIN_PATH = "train.csv"  # update on Kaggle

# %%
train = pd.read_csv(TRAIN_PATH)
feature_cols = [c for c in train.columns if c not in ["Activity", "id", "subject"]]

acc_features = [c for c in feature_cols if "Acc" in c and "Gyro" not in c]
gyro_features = [c for c in feature_cols if "Gyro" in c and "Acc" not in c]
other_features = [c for c in feature_cols if c not in acc_features and c not in gyro_features]

print(f"Total features: {len(feature_cols)}")
print(f"Accelerometer-derived: {len(acc_features)}")
print(f"Gyroscope-derived: {len(gyro_features)}")
print(f"Other (angle/combined, excluded from single-sensor subsets): {len(other_features)}")
print(f"\nExample 'other' features (for sanity-check): {other_features[:5]}")

# %%
le = LabelEncoder()
y_train_encoded = le.fit_transform(train["Activity"])
subjects = train["subject"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(gss.split(train[feature_cols], y_train_encoded, groups=subjects))

y_tr, y_val = y_train_encoded[tr_idx], y_train_encoded[val_idx]

# %%
BEST_SVM_PARAMS = {"kernel": "rbf", "C": 10, "gamma": "scale"}  # confirmed best from earlier search

def train_eval_subset(cols, label):
    X_tr = train[cols].iloc[tr_idx].values
    X_val = train[cols].iloc[val_idx].values
    model = SVC(random_state=42, **BEST_SVM_PARAMS)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, y_pred, average="macro")
    print(f"{label:30s} ({len(cols):3d} features) | val Macro F1 = {val_f1:.4f}")
    return val_f1, y_pred, model

# %%
print("--- Sensor contribution comparison ---")
full_f1, y_pred_full, _ = train_eval_subset(feature_cols, "All features (baseline)")
acc_f1, y_pred_acc, _ = train_eval_subset(acc_features, "Accelerometer-only")
gyro_f1, y_pred_gyro, _ = train_eval_subset(gyro_features, "Gyroscope-only")

# %%
print("\n--- Performance drop from full feature set ---")
print(f"Accelerometer-only: {full_f1:.4f} -> {acc_f1:.4f}  (drop: {full_f1 - acc_f1:.4f})")
print(f"Gyroscope-only:     {full_f1:.4f} -> {gyro_f1:.4f}  (drop: {full_f1 - gyro_f1:.4f})")

# %%
print("\n--- Per-class F1: full vs accelerometer-only vs gyroscope-only ---")
per_class_rows = []
for label, y_pred in [("Full", y_pred_full), ("Acc-only", y_pred_acc), ("Gyro-only", y_pred_gyro)]:
    report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
    for cls in le.classes_:
        per_class_rows.append({"feature_set": label, "class": cls, "f1": report[cls]["f1-score"]})
per_class_df = pd.DataFrame(per_class_rows)
pivot = per_class_df.pivot(index="class", columns="feature_set", values="f1")[["Full", "Acc-only", "Gyro-only"]].loc[le.classes_]
print(pivot.round(3).to_string())

# %%
sensor_comparison = pd.DataFrame([
    {"feature_set": "All features", "n_features": len(feature_cols), "val_macro_f1": full_f1},
    {"feature_set": "Accelerometer-only", "n_features": len(acc_features), "val_macro_f1": acc_f1},
    {"feature_set": "Gyroscope-only", "n_features": len(gyro_features), "val_macro_f1": gyro_f1},
])
sensor_comparison.to_csv("processed/sensor_contribution_comparison.csv", index=False)
pivot.to_csv("processed/sensor_contribution_per_class.csv")
print("\nSaved processed/sensor_contribution_comparison.csv and per_class breakdown")