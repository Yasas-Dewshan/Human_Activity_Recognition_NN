# HAR Competition — Larger MLP + Label Smoothing + Per Subject Normalization

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
    "hidden_sizes": [512,256,128],
    "dropout": 0.4,
    "weight_decay": 0.0,
    "batchnorm": True,
    "lr": 1e-3,
    "batch_size": 64
}

LABEL_SMOOTHING = 0.05
ENSEMBLE_SEEDS = [42,43,44,45,46]

def per_subject_normalize(X, subjects):
    X_norm = np.zeros_like(X, dtype=np.float64)

    for subj in np.unique(subjects):
        mask = subjects == subj
        mean = X[mask].mean(axis=0)
        std = X[mask].std(axis=0) + 1e-8
        X_norm[mask] = (X[mask] - mean) / std

    return X_norm.astype(np.float32)

train = pd.read_csv(TRAIN_PATH)

feature_cols = [
    c for c in train.columns
    if c not in ["Activity","id","subject"]
]

le = LabelEncoder()

y_train_encoded = le.fit_transform(
    train["Activity"]
)

X_train_full = train[feature_cols].values
subjects = train["subject"].values

NUM_CLASSES = len(le.classes_)
INPUT_DIM = X_train_full.shape[1]

os.makedirs("processed", exist_ok=True)


class MLP(nn.Module):
    def __init__(self,input_dim,hidden_sizes,num_classes,dropout,batchnorm):
        super().__init__()

        layers=[]
        prev=input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev,h))

            if batchnorm:
                layers.append(nn.BatchNorm1d(h))

            layers.append(nn.ReLU())

            if dropout>0:
                layers.append(nn.Dropout(dropout))

            prev=h

        layers.append(nn.Linear(prev,num_classes))
        self.net=nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)


def make_loader(X,y,batch_size,seed):
    generator=torch.Generator().manual_seed(seed)

    ds=TensorDataset(
        torch.tensor(X,dtype=torch.float32),
        torch.tensor(y,dtype=torch.long)
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

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
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
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

        for xb,yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            loss = criterion(
                model(xb),
                yb
            )

            loss.backward()
            optimizer.step()

        if early_stop:

            model.eval()

            with torch.no_grad():
                preds = model(X_val).argmax(dim=1).cpu().numpy()

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
        return model,best_f1,best_epoch

    return model,None,epochs-1



# Phase 1: subject grouped validation split

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

tr_idx,val_idx = next(
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

subj_tr = subjects[tr_idx]
subj_val = subjects[val_idx]

assert len(set(subj_tr)&set(subj_val)) == 0


X_tr_norm = per_subject_normalize(
    X_tr,
    subj_tr
)

X_val_norm = per_subject_normalize(
    X_val,
    subj_val
)


X_val_t = torch.tensor(
    X_val_norm,
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


_,best_val_f1,best_epoch = train_nn(
    X_tr_norm,
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
    f"Phase 1 done. Best epoch: {best_epoch} | Validation Macro F1: {best_val_f1:.4f}"
)

# Phase 2: train final ensemble on all training data

X_full_norm = per_subject_normalize(
    X_train_full,
    subjects
)

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

    model,_,_ = train_nn(
        X_full_norm,
        y_train_encoded,
        class_weights_full,
        epochs=n_final_epochs,
        seed=seed
    )

    ensemble_models.append(model)

    torch.save(
        model.state_dict(),
        f"processed/nn_large_labelsmoothing_seed_{seed}.pt"
    )

    print(
        f"Finished seed {seed} ({n_final_epochs} epochs)"
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
    "processed/feature_cols.pkl",
    "wb"
) as f:
    pickle.dump(
        feature_cols,
        f
    )


print(
    f"Trained and saved {len(ensemble_models)} models."
)

# Generate submission

if os.path.exists(TEST_PATH):

    test = pd.read_csv(TEST_PATH)

    assert "subject" in test.columns, (
        "test.csv must contain subject column "
        "for per-subject normalization"
    )


    X_test = test[
        feature_cols
    ].values


    test_subjects = test[
        "subject"
    ].values


    X_test_norm = per_subject_normalize(
        X_test,
        test_subjects
    )


    X_test_t = torch.tensor(
        X_test_norm,
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
        "submission_large_labelsmoothing.csv",
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


    try:

        sample_sub = pd.read_csv(
            SAMPLE_SUB_PATH
        )

        assert list(
            submission.columns
        ) == list(
            sample_sub.columns
        )

        print(
            "Column structure matches sample_submission.csv."
        )

    except FileNotFoundError:

        print(
            f"Could not find {SAMPLE_SUB_PATH}."
        )


    print(
        "submission_large_labelsmoothing.csv saved."
    )

    print(
        f"Rows: {len(submission)}"
    )


    print(
        "\nPredicted class distribution:"
    )

    print(
        submission["Activity"].value_counts()
    )


else:

    print(
        f"test.csv not found at {TEST_PATH}"
    )