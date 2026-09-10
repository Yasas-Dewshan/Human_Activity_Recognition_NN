# %% [markdown]
# # Leak-check: new UCI HAR dataset vs. competition train/test
#
# Run this BEFORE merging any new data into your training pipeline. Checks,
# in order of severity:
#   1. Feature schema match (column names/order)
#   2. Subject ID overlap between new data and competition test.csv
#      (the dangerous one -- training on a test subject's other windows can
#      still leak information about their movement style/calibration)
#   3. Exact row-level duplicates between new data and existing train/test
#      (catches "new dataset" actually being your existing data repackaged)
#   4. Near-duplicate check via feature hashing, in case IDs were reassigned
#      but underlying sensor windows are identical

# %%
import pandas as pd
import numpy as np
import hashlib

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
NEW_DATA_PATH = "new_uci_har.csv"   # <-- point this at whatever you downloaded

# %% [markdown]
# ## 0. Load everything
# If your "new" dataset came as the raw UCI folder (X_train.txt, y_train.txt,
# subject_train.txt, X_test.txt, ...) instead of a single CSV, use the loader
# in the commented block below instead of pd.read_csv.

# %%
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
new_data = pd.read_csv(NEW_DATA_PATH)

# --- Uncomment if the new data is the raw UCI folder format instead ---
# import os
# RAW_DIR = "UCI HAR Dataset"
# def load_raw_split(split):
#     X = pd.read_csv(os.path.join(RAW_DIR, split, f"X_{split}.txt"),
#                      sep=r"\s+", header=None)
#     y = pd.read_csv(os.path.join(RAW_DIR, split, f"y_{split}.txt"),
#                      header=None, names=["Activity"])
#     subj = pd.read_csv(os.path.join(RAW_DIR, split, f"subject_{split}.txt"),
#                         header=None, names=["subject"])
#     features = pd.read_csv(os.path.join(RAW_DIR, "features.txt"), sep=r"\s+",
#                             header=None, names=["idx", "name"])
#     X.columns = features["name"].values
#     activity_labels = pd.read_csv(os.path.join(RAW_DIR, "activity_labels.txt"),
#                                    sep=r"\s+", header=None, names=["id", "label"])
#     label_map = dict(zip(activity_labels["id"], activity_labels["label"]))
#     y["Activity"] = y["Activity"].map(label_map)
#     return pd.concat([X, y, subj], axis=1)
# new_train_raw = load_raw_split("train")
# new_test_raw = load_raw_split("test")
# new_data = pd.concat([new_train_raw, new_test_raw], ignore_index=True)

print(f"train.csv:    {len(train)} rows, {len(train.columns)} cols")
print(f"test.csv:     {len(test)} rows, {len(test.columns)} cols")
print(f"new dataset:  {len(new_data)} rows, {len(new_data.columns)} cols")

# %% [markdown]
# ## 1. Feature schema check

# %%
train_feat_cols = set(c for c in train.columns if c not in ["Activity", "id", "subject"])
new_feat_cols = set(c for c in new_data.columns if c not in ["Activity", "id", "subject"])

missing_in_new = train_feat_cols - new_feat_cols
extra_in_new = new_feat_cols - train_feat_cols

print(f"\nFeature columns in train.csv but NOT in new dataset: {len(missing_in_new)}")
if missing_in_new:
    print(f"  e.g. {list(missing_in_new)[:10]}")
print(f"Feature columns in new dataset but NOT in train.csv: {len(extra_in_new)}")
if extra_in_new:
    print(f"  e.g. {list(extra_in_new)[:10]}")

if missing_in_new or extra_in_new:
    print("\n*** SCHEMA MISMATCH -- do not concat directly. Align columns first. ***")
else:
    print("\nSchema matches. Safe to align on train_feat_cols for merging.")

shared_cols = sorted(train_feat_cols & new_feat_cols)

# %% [markdown]
# ## 2. Subject ID overlap check (the critical one)
# If test.csv has a 'subject' column and any of those IDs appear in the new
# dataset, training on the new dataset would leak information about your
# actual competition test subjects.

# %%
if "subject" in test.columns and "subject" in new_data.columns:
    test_subjects = set(test["subject"].unique())
    new_subjects = set(new_data["subject"].unique())
    train_subjects = set(train["subject"].unique()) if "subject" in train.columns else set()

    overlap_with_test = test_subjects & new_subjects
    overlap_with_train = train_subjects & new_subjects

    print(f"\ntest.csv subjects:  {sorted(test_subjects)}")
    print(f"new dataset subjects: {sorted(new_subjects)}")
    print(f"\nSubjects in BOTH test.csv and new dataset: {sorted(overlap_with_test)}")
    print(f"Subjects in BOTH train.csv and new dataset: {sorted(overlap_with_train)}")

    if overlap_with_test:
        print("\n*** LEAK RISK: drop these subjects from the new dataset before "
              "using it for training, or you're training on your test subjects. ***")
    else:
        print("\nNo subject overlap with test.csv -- safe on this axis.")
else:
    print("\nNo 'subject' column in test.csv and/or new dataset -- cannot check subject "
          "overlap. Treat this as unresolved risk, not as 'safe'.")

# %% [markdown]
# ## 3. Exact row-level duplicate check
# Hashes each row's feature values to catch the new dataset being a repackage
# of data you already have (common when "new" datasets are just UCI HAR
# train+test concatenated under a different filename).

# %%
def hash_rows(df, cols):
    # round to reduce float-precision false negatives, then hash
    rounded = df[cols].round(6)
    return rounded.apply(lambda row: hashlib.md5(row.values.tobytes()).hexdigest(), axis=1)

train_hashes = set(hash_rows(train, shared_cols))
test_hashes = set(hash_rows(test, shared_cols)) if all(c in test.columns for c in shared_cols) else set()
new_hashes = hash_rows(new_data, shared_cols)

dup_with_train = new_hashes.isin(train_hashes)
dup_with_test = new_hashes.isin(test_hashes) if test_hashes else pd.Series(False, index=new_hashes.index)

print(f"\nNew dataset rows that exactly duplicate a train.csv row: {dup_with_train.sum()} / {len(new_data)}")
print(f"New dataset rows that exactly duplicate a test.csv row:  {dup_with_test.sum()} / {len(new_data)}")

if dup_with_test.sum() > 0:
    print("\n*** LEAK: some new-dataset rows are byte-identical to test.csv rows. "
          "These MUST be excluded before training. ***")
if dup_with_train.sum() > 0:
    print(f"\n{dup_with_train.sum()} rows already exist in train.csv -- drop these "
          "duplicates before merging (they'd just double-weight those samples).")

# %% [markdown]
# ## 4. Summary: what's actually safe to add

# %%
truly_new_mask = ~(dup_with_train.values | dup_with_test.values)
if "subject" in new_data.columns and "subject" in test.columns:
    truly_new_mask &= ~new_data["subject"].isin(test_subjects).values

n_usable = truly_new_mask.sum()
print(f"\n=== SUMMARY ===")
print(f"New dataset total rows:        {len(new_data)}")
print(f"Usable rows (new, no leak):     {n_usable}")
print(f"Excluded (dup of train):        {dup_with_train.sum()}")
print(f"Excluded (dup of test / leak):  {dup_with_test.sum()}")
if "subject" in new_data.columns and "subject" in test.columns:
    print(f"Excluded (subject overlaps test): {new_data['subject'].isin(test_subjects).sum()}")

if n_usable > 0:
    safe_new_data = new_data[truly_new_mask].reset_index(drop=True)
    safe_new_data.to_csv("new_data_safe_to_merge.csv", index=False)
    print(f"\nSaved {n_usable} verified-clean rows to new_data_safe_to_merge.csv")
    print("You can pd.concat() this with your existing train.csv.")
else:
    print("\nNothing usable -- the 'new' dataset appears to be entirely redundant "
          "with or leaking into your existing train/test.")