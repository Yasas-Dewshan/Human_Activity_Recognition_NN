# %% [markdown]
# # HAR Competition — Subject Leakage Experiment
#
# This experiment intentionally allows subject information into the model.
#
# Differences from the best model:
#
# 1. Subject column is included as an input feature.
# 2. No per-subject normalization is applied.
# 3. 5-seed ensemble is retained.
#
# Purpose:
# Test whether subject identity information improves leaderboard performance.

# %%

import numpy as np
import pandas as pd
import pickle
import os
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"


NN_CONFIG = {
    "hidden_sizes": [256, 128],
    "dropout": 0.5,
    "weight_decay": 0.0,
    "batchnorm": True,
    "lr": 1e-3,
    "batch_size": 64,
}


ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)


# %%
train = pd.read_csv(TRAIN_PATH)


# IMPORTANT:
# Subject is intentionally NOT removed.
# This is the leakage experiment.

feature_cols = [
    c for c in train.columns
    if c not in ["Activity", "id"]
]


le = LabelEncoder()

y_train_encoded = le.fit_transform(train["Activity"])

X_train_full = train[feature_cols].values.astype(np.float32)

subjects = train["subject"].values


NUM_CLASSES = len(le.classes_)
INPUT_DIM = X_train_full.shape[1]


os.makedirs("processed", exist_ok=True)



# %%
class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_sizes,
        num_classes,
        dropout=0.0,
        batchnorm=False
    ):
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
def make_loader(X, y, batch_size, seed):

    generator = torch.Generator().manual_seed(seed)

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )



# %%
def train_nn(
    X_train,
    y_train,
    class_weights,
    epochs,
    seed,
    X_val=None,
    y_val=None,
    early_stop=False,
    patience=12
):

    torch.manual_seed(seed)


    model = MLP(
        INPUT_DIM,
        NN_CONFIG["hidden_sizes"],
        NUM_CLASSES,
        NN_CONFIG["dropout"],
        NN_CONFIG["batchnorm"]
    ).to(device)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=NN_CONFIG["lr"],
        weight_decay=NN_CONFIG["weight_decay"]
    )


    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    scheduler = None

    if early_stop:

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4
        )


    loader = make_loader(
        X_train,
        y_train,
        NN_CONFIG["batch_size"],
        seed
    )


    best_f1 = -1
    best_epoch = 0
    best_state = None
    counter = 0



    for epoch in range(epochs):

        model.train()


        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            output = model(xb)

            loss = criterion(output, yb)

            loss.backward()

            optimizer.step()



        if early_stop:

            model.eval()

            with torch.no_grad():

                preds = model(X_val).argmax(dim=1)
                preds = preds.cpu().numpy()


            val_f1 = f1_score(
                y_val,
                preds,
                average="macro"
            )


            scheduler.step(val_f1)


            if val_f1 > best_f1:

                best_f1 = val_f1
                best_epoch = epoch

                best_state = {
                    k:v.clone()
                    for k,v in model.state_dict().items()
                }

                counter = 0

            else:

                counter += 1

                if counter >= patience:
                    break



    if early_stop:

        model.load_state_dict(best_state)

        return model, best_f1, best_epoch


    return model, None, epochs-1

# %% [markdown]
# ## Phase 1: Find optimal epoch count
#
# Uses subject-grouped validation split.
# Subject is still included as a feature intentionally.

# %%

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)


tr_idx, val_idx = next(
    gss.split(
        X_train_full,
        y_train_encoded,
        groups=subjects
    )
)


X_tr = X_train_full[tr_idx]
X_val = X_train_full[val_idx]

y_tr = y_train_encoded[tr_idx]
y_val = y_train_encoded[val_idx]


# Confirm no subject overlap

assert len(
    set(subjects[tr_idx]) &
    set(subjects[val_idx])
) == 0



X_val_t = torch.tensor(
    X_val,
    dtype=torch.float32
).to(device)



class_weights_tr = torch.tensor(
    np.bincount(
        y_tr,
        minlength=NUM_CLASSES
    ).sum()
    /
    (
        NUM_CLASSES *
        np.bincount(
            y_tr,
            minlength=NUM_CLASSES
        )
    ),
    dtype=torch.float32
).to(device)



_, best_val_f1, best_epoch = train_nn(
    X_tr,
    y_tr,
    class_weights_tr,
    epochs=100,
    seed=42,
    X_val=X_val_t,
    y_val=y_val,
    early_stop=True,
    patience=12
)


n_final_epochs = best_epoch + 1


print(
    f"Phase 1 done. "
    f"Best epoch: {best_epoch} | "
    f"val Macro F1: {best_val_f1:.4f}"
)



# %% [markdown]
# ## Phase 2: Train 5-seed ensemble on all training data
#
# The final models see:
#
# - All training samples
# - Subject ID feature
# - No normalization
#
# Five independent models are trained and their probabilities are averaged.

# %%


class_weights_full = torch.tensor(
    np.bincount(
        y_train_encoded,
        minlength=NUM_CLASSES
    ).sum()
    /
    (
        NUM_CLASSES *
        np.bincount(
            y_train_encoded,
            minlength=NUM_CLASSES
        )
    ),
    dtype=torch.float32
).to(device)



ensemble_models = []


for seed in ENSEMBLE_SEEDS:

    model, _, _ = train_nn(
        X_train_full,
        y_train_encoded,
        class_weights_full,
        epochs=n_final_epochs,
        seed=seed
    )


    ensemble_models.append(model)


    torch.save(
        model.state_dict(),
        f"processed/nn_subject_leak_seed_{seed}.pt"
    )


    print(
        f"Finished seed {seed} "
        f"({n_final_epochs} epochs)"
    )



with open(
    "processed/label_encoder.pkl",
    "wb"
) as f:

    pickle.dump(
        le,
        f
    )


with open(
    "processed/feature_cols_subject_leak.pkl",
    "wb"
) as f:

    pickle.dump(
        feature_cols,
        f
    )



print(
    f"Saved {len(ensemble_models)} subject leakage models."
)



# %% [markdown]
# ## Generate submission using subject leakage model

# %%


if os.path.exists(TEST_PATH):

    test = pd.read_csv(TEST_PATH)


    X_test = test[
        feature_cols
    ].values.astype(np.float32)



    X_test_t = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(device)



    all_probs = []


    for model in ensemble_models:

        model.eval()


        with torch.no_grad():

            probs = torch.softmax(
                model(X_test_t),
                dim=1
            ).cpu().numpy()


        all_probs.append(
            probs
        )



    avg_probs = np.mean(
        all_probs,
        axis=0
    )


    preds = avg_probs.argmax(
        axis=1
    )


    preds_labels = le.inverse_transform(
        preds
    )



    test_ids = (
        test["id"]
        if "id" in test.columns
        else np.arange(
            1,
            len(test)+1
        )
    )



    submission = pd.DataFrame(
        {
            "id": test_ids,
            "Activity": preds_labels
        }
    )


    submission.to_csv(
        "submission_subject_leakage.csv",
        index=False
    )



    assert len(submission) == len(test)

    assert set(
        submission["Activity"]
    ) <= set(
        le.classes_
    )


    assert submission[
        "Activity"
    ].isnull().sum() == 0



    print(
        "Submission created:"
    )

    print(
        "submission_subject_leakage.csv"
    )


    print(
        f"Rows: {len(submission)}"
    )


    print(
        "\nPrediction distribution:"
    )


    print(
        submission["Activity"].value_counts()
    )



else:

    print(
        f"test.csv not found at {TEST_PATH}"
    )