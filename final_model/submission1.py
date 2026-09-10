# %% [markdown]
# # HAR Competition — Working Notebook (trimmed)
# Part 1: Load | Part 2: Preprocessing | Part 3: Decision Tree (required baseline)
# | Part 4: SVM (required baseline) | Part 5: Submission (SVM only, best model)
#
# NOTE: PCA was investigated and dropped (raw features won for both DT and SVM
# -- see prior results: DT raw 0.87 vs PCA 0.82, SVM raw 0.96 vs PCA 0.95).
# This version is stripped to essentials for fast iteration -- plots and full
# result tables will be added back in the final polished notebook for the
# report, since confusion matrices etc. are useful evidence there.
#
# DT stays in the notebook (required baseline per guidelines) but is NOT used
# for the submission file -- SVM is the stronger model and is what generates
# submission.csv.

# %%
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

def find_path(filename, default_dir="data"):
    for p in [os.path.join(default_dir, filename), filename, os.path.join("..", default_dir, filename), os.path.join("..", filename)]:
        if os.path.exists(p):
            return p
    return os.path.join(default_dir, filename)

def get_dir(dirname):
    if os.path.exists(dirname):
        return dirname
    parent_dir = os.path.join("..", dirname)
    if os.path.exists(parent_dir):
        return parent_dir
    os.makedirs(dirname, exist_ok=True)
    return dirname

PROCESSED_DIR = get_dir("processed")
SUBMISSIONS_DIR = get_dir("submissions")

TRAIN_PATH = find_path("train.csv")  # update on Kaggle

# %% [markdown]
# ## Part 1-2: Load & Preprocess

# %%
train = pd.read_csv(TRAIN_PATH)
feature_cols = [c for c in train.columns if c not in ["Activity", "id", "subject"]]

le = LabelEncoder()
y_train_encoded = le.fit_transform(train["Activity"])
X_train_full = train[feature_cols]
subjects = train["subject"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(gss.split(X_train_full, y_train_encoded, groups=subjects))

X_tr, X_val = X_train_full.iloc[tr_idx].values, X_train_full.iloc[val_idx].values
y_tr, y_val = y_train_encoded[tr_idx], y_train_encoded[val_idx]
subj_tr = subjects[tr_idx]

assert len(set(subjects[tr_idx]) & set(subjects[val_idx])) == 0  # subject leakage check

with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)
with open(os.path.join(PROCESSED_DIR, "feature_cols.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)

print(f"Loaded. Train: {X_tr.shape}, Val: {X_val.shape}, subjects OK.")

# %% [markdown]
# ## Part 3: Decision Tree (required baseline)

# %%
dt_grid = {
    "max_depth": [8, 10, 13, 16, None],
    "min_samples_leaf": [1, 5, 10],
    "criterion": ["gini", "entropy"],
}
cv_dt = list(GroupKFold(n_splits=5).split(X_tr, y_tr, groups=subj_tr))

dt_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_grid,
                          scoring="f1_macro", cv=cv_dt, n_jobs=-1, verbose=0)
dt_search.fit(X_tr, y_tr)

dt_final = dt_search.best_estimator_
dt_val_f1 = f1_score(y_val, dt_final.predict(X_val), average="macro")
print(f"Decision Tree — best params: {dt_search.best_params_} | val Macro F1: {dt_val_f1:.4f}")

# %% [markdown]
# ## Part 4: SVM (required baseline, also used for final submission)

# %%
svm_grid = [
    {"scaler": [StandardScaler(), "passthrough"], "svm__kernel": ["linear"], "svm__C": [0.1, 1, 10]},
    {"scaler": [StandardScaler(), "passthrough"], "svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10], "svm__gamma": ["scale"]},
]
pipe_svm = Pipeline([("scaler", "passthrough"), ("svm", SVC(random_state=42))])
cv_svm = list(GroupKFold(n_splits=3).split(X_tr, y_tr, groups=subj_tr))

svm_search = GridSearchCV(pipe_svm, svm_grid, scoring="f1_macro", cv=cv_svm, n_jobs=-1, verbose=0)
svm_search.fit(X_tr, y_tr)

svm_final = svm_search.best_estimator_
svm_val_f1 = f1_score(y_val, svm_final.predict(X_val), average="macro")
print(f"SVM — best params: {svm_search.best_params_} | val Macro F1: {svm_val_f1:.4f}")

# %% [markdown]
# ## Part 5: Submission (SVM only — best model)
# Uses X_val as a stand-in for test.csv until the real file is available.
# Swap instructions below once test.csv arrives.

# %%
best_svm_params = {"kernel": "rbf", "C": 10, "gamma": "scale"}  # from svm_search.best_params_

svm_submission_model = SVC(random_state=42, **best_svm_params)
svm_submission_model.fit(X_tr, y_tr)

# --- Replace this block once test.csv is available ---
X_test = X_val
mock_test_ids = np.arange(1, len(X_val) + 1)  # ids start at 1, matching sample_submission.csv
# test = pd.read_csv(TEST_PATH); X_test = test[feature_cols].values; mock_test_ids = test["id"]
# -------------------------------------------------------

preds = le.inverse_transform(svm_submission_model.predict(X_test))
submission = pd.DataFrame({"id": mock_test_ids, "Activity": preds})
submission.to_csv(os.path.join(SUBMISSIONS_DIR, "submission.csv"), index=False)

assert len(submission) == len(X_test)
assert set(submission["Activity"]) <= set(le.classes_)
assert submission["Activity"].isnull().sum() == 0
assert list(submission.columns) == ["id", "Activity"]  # no extra columns, correct order

sample_sub_file = find_path("sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_file)
assert list(submission.columns) == list(sample_sub.columns), \
    f"Column mismatch: {list(submission.columns)} vs {list(sample_sub.columns)}"

print(f"submission.csv saved. Rows: {len(submission)}. Mock Macro F1: {f1_score(y_val, svm_submission_model.predict(X_val), average='macro'):.4f}")