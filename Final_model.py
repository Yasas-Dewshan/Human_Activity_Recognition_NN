import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 43

CONFIG = {
    "hidden_sizes": [256, 128],
    "dropout": 0.5,
    "batchnorm": True,
    "weight_decay": 0.0,
    "lr": 1e-3,
    "batch_size": 64,
}

torch.manual_seed(SEED)
np.random.seed(SEED)

# Load data
X_tr = np.load("processed/X_tr.npy")
X_val = np.load("processed/X_val.npy")
y_tr = np.load("processed/y_tr.npy")
y_val = np.load("processed/y_val.npy")

with open("processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

NUM_CLASSES = len(le.classes_)
INPUT_DIM = X_tr.shape[1]

# Scale features
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)

X_tr = torch.tensor(X_tr, dtype=torch.float32)
y_tr = torch.tensor(y_tr, dtype=torch.long)

X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

train_ds = TensorDataset(X_tr, y_tr)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, num_classes, dropout, batchnorm):
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

model = MLP(
    INPUT_DIM,
    CONFIG["hidden_sizes"],
    NUM_CLASSES,
    CONFIG["dropout"],
    CONFIG["batchnorm"],
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"],
)

criterion = nn.CrossEntropyLoss()

loader = DataLoader(
    train_ds,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)

best_f1 = -1
best_state = None
patience = 12
counter = 0
max_epochs = 100

history_train = []
history_val = []

for epoch in range(max_epochs):

    model.train()

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():

        train_pred = model(X_tr.to(device)).argmax(1).cpu().numpy()
        val_pred = model(X_val_t).argmax(1).cpu().numpy()

    train_f1 = f1_score(y_tr.numpy(), train_pred, average="macro")
    val_f1 = f1_score(y_val, val_pred, average="macro")

    history_train.append(train_f1)
    history_val.append(val_f1)

    print(
        f"Epoch {epoch+1:3d} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

model.load_state_dict(best_state)

print(f"\nBest Validation Macro F1 = {best_f1:.4f}")

model.eval()

with torch.no_grad():
    y_pred = model(X_val_t).argmax(1).cpu().numpy()

print(classification_report(y_val, y_pred, target_names=le.classes_))

ConfusionMatrixDisplay.from_predictions(
    y_val,
    y_pred,
    display_labels=le.classes_,
    xticks_rotation=45,
    cmap="Purples",
)

plt.title("Feedforward Neural Network Confusion Matrix")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history_train, label="Train Macro F1")
plt.plot(history_val, label="Validation Macro F1")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("Training History")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

torch.save(model.state_dict(), "processed/nn_best_model.pt")

with open("processed/nn_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Best model and scaler saved.")