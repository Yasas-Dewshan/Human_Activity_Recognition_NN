# %% [markdown]
# # Extended Task 3, Step 1: Load raw signals + verify alignment with train.csv
#
# Before trusting that row N in the raw UCI data corresponds to row N in
# Kaggle's train.csv, we verify it directly: subject IDs and activity labels
# should match exactly, row for row, between the two sources. If they don't,
# we can't safely reuse the same subject-grouped split logic across both.

# %%
import numpy as np
import pandas as pd

UCI_DIR = "UCI HAR Dataset"  # update to wherever you extracted the zip
KAGGLE_TRAIN_PATH = "train.csv"  # your existing Kaggle train.csv

# %%
# Load raw inertial signals -- 9 channels, each (n_samples, 128 timesteps)
signal_names = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

def load_signals(split):
    signals = []
    for name in signal_names:
        path = f"{UCI_DIR}/{split}/Inertial Signals/{name}_{split}.txt"
        arr = np.loadtxt(path)  # shape (n_samples, 128)
        signals.append(arr)
    # Stack into (n_samples, 128 timesteps, 9 channels)
    return np.stack(signals, axis=-1)

X_raw_train = load_signals("train")
print(f"Raw train signals shape: {X_raw_train.shape}  (samples, timesteps, channels)")

# %%
y_raw = np.loadtxt(f"{UCI_DIR}/train/y_train.txt", dtype=int)  # 1-6
subject_raw = np.loadtxt(f"{UCI_DIR}/train/subject_train.txt", dtype=int)

activity_map = {}
with open(f"{UCI_DIR}/activity_labels.txt") as f:
    for line in f:
        idx, name = line.strip().split()
        activity_map[int(idx)] = name

y_raw_labels = np.array([activity_map[i] for i in y_raw])
print(f"Raw labels shape: {y_raw_labels.shape}, unique: {np.unique(y_raw_labels)}")
print(f"Raw subjects shape: {subject_raw.shape}, unique count: {len(np.unique(subject_raw))}")

# %% [markdown]
# ## Verification: does row order match Kaggle's train.csv?

# %%
kaggle_train = pd.read_csv(KAGGLE_TRAIN_PATH)
print(f"Kaggle train.csv shape: {kaggle_train.shape}")

# %%
same_length = len(kaggle_train) == len(y_raw_labels)
print(f"Same row count: {same_length} (Kaggle: {len(kaggle_train)}, raw: {len(y_raw_labels)})")

if same_length:
    labels_match = (kaggle_train["Activity"].values == y_raw_labels).all()
    subjects_match = (kaggle_train["subject"].values == subject_raw).all()
    print(f"Activity labels match row-for-row: {labels_match}")
    print(f"Subject IDs match row-for-row: {subjects_match}")

    if labels_match and subjects_match:
        print("\nALIGNED. Safe to reuse the same subject-grouped split logic "
              "across both train.csv and the raw signal arrays -- row N means "
              "the same thing in both.")
    else:
        n_label_mismatches = (kaggle_train["Activity"].values != y_raw_labels).sum()
        n_subject_mismatches = (kaggle_train["subject"].values != subject_raw).sum()
        print(f"\nNOT ALIGNED. {n_label_mismatches} label mismatches, "
              f"{n_subject_mismatches} subject mismatches out of {len(kaggle_train)} rows.")
        print("Do not assume row N means the same thing in both sources -- "
              "would need to re-derive a mapping (e.g. by matching each row's "
              "561 hand-crafted features against a recomputation from raw "
              "signals) before proceeding.")
else:
    print("\nRow counts differ -- train.csv and the raw UCI data are not the "
          "same partition. Cannot assume alignment without further work.")

# %%
# Extra sanity check regardless of the above: does a statistic computed from
# the raw signal roughly match the corresponding hand-crafted feature? E.g.
# tBodyAcc-mean()-X should be close to the mean of body_acc_x's first window.
if same_length and "tBodyAcc-mean()-X" in kaggle_train.columns:
    computed_mean = X_raw_train[0, :, 0].mean()  # body_acc_x, row 0
    reported_mean = kaggle_train["tBodyAcc-mean()-X"].iloc[0]
    print(f"\nCross-check row 0: computed body_acc_x mean = {computed_mean:.4f}, "
          f"reported tBodyAcc-mean()-X = {reported_mean:.4f}")
    print("(Note: the hand-crafted feature is normalized/scaled differently than "
          "raw signal, so these won't match exactly -- just check they're in a "
          "similar ballpark and correlate in sign/direction, not identical.)")