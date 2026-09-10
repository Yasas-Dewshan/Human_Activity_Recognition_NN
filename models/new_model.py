# %% [markdown]
# # Margin-based SITTING/STANDING cascade — layered on top of the actual champion pipeline
#
# This reproduces your notebook's EXACT preprocessing (engineered physics
# features, per-subject normalization, RobustScaler, AdaBN ensemble
# prediction) so the validation number here is directly comparable to the
# notebook's own 0.9865 champion score.
#
# Unlike the notebook's cascade (which REPLACED the flat pipeline and lost,
# 0.9278 < 0.9865), this only overrides predictions for genuinely ambiguous
# SITTING/STANDING cases -- everywhere else, the flat ensemble's answer is
# kept untouched. It prints a clear before/after so you can see if it's
# worth adding as a real Stage 2.5 in the notebook.

# %%
import os, sys, time, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 64
LR = 1e-3
LABEL_SMOOTHING = 0.04
AMBIGUITY_MARGIN = 0.25   # tune this: lower = fewer, more-confident overrides

# %% [markdown]
# ## Load data (adjust find_data_file paths if your local layout differs)

# %%
def find_data_file(filename):
    paths_to_check = [
        os.path.join(".", filename),
        os.path.join("co5420-human-activity-recognition", filename),
        os.path.join("..", filename),
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not locate '{filename}'. Edit find_data_file() paths.")

train_df = pd.read_csv(find_data_file("train.csv"))
test_df = pd.read_csv(find_data_file("test.csv"))
print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

raw_feature_cols = [c for c in train_df.columns if c not in ["Activity", "id", "subject"]]

# %% [markdown]
# ## Reproduce the notebook's exact feature engineering (physics features)

# %%
REQUIRED_PHYSICS_COLS = [
    "tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z",
    "tGravityAcc-mean()-X", "tGravityAcc-mean()-Y", "tGravityAcc-mean()-Z",
]

def engineer_defensible_features(df, feature_cols):
    for col in REQUIRED_PHYSICS_COLS:
        assert col in df.columns, f"Required feature column '{col}' missing!"
    X = df[feature_cols].copy()

    bx, by, bz = df["tBodyAcc-mean()-X"], df["tBodyAcc-mean()-Y"], df["tBodyAcc-mean()-Z"]
    gx, gy, gz = df["tGravityAcc-mean()-X"], df["tGravityAcc-mean()-Y"], df["tGravityAcc-mean()-Z"]
    bmag = np.sqrt(bx**2 + by**2 + bz**2)
    gmag = np.sqrt(gx**2 + gy**2 + gz**2)
    dot_prod = (bx * gx + by * gy + bz * gz)
    cos_angle = dot_prod / (bmag * gmag + 1e-6)

    X["feat_body_grav_ratio"] = bmag / (gmag + 1e-6)
    X["feat_body_mag"] = bmag
    X["feat_grav_mag"] = gmag
    X["feat_body_grav_product"] = bmag * gmag
    X["feat_dot_prod"] = dot_prod
    X["feat_cos_angle"] = cos_angle

    if "angle(Y,gravityMean)" in df and "tGravityAcc-mean()-Y" in df:
        X["feat_y_angle_ratio"] = df["angle(Y,gravityMean)"] / (df["tGravityAcc-mean()-Y"].abs() + 1e-6)
    if "tBodyAccJerkMag-std()" in df and "tBodyAccMag-std()" in df:
        X["feat_stair_jerk_ratio"] = df["tBodyAccJerkMag-std()"] / (df["tBodyAccMag-std()"].abs() + 1e-6)

    return X.values

X_train_engineered = engineer_defensible_features(train_df, raw_feature_cols)
X_test_engineered = engineer_defensible_features(test_df, raw_feature_cols)

label_encoder = LabelEncoder()
y_train_all = label_encoder.fit_transform(train_df["Activity"])
subjects_train_all = train_df["subject"].values
subjects_test = test_df["subject"].values

NUM_CLASSES = len(label_encoder.classes_)
INPUT_DIM = X_train_engineered.shape[1]
SITTING_IDX = list(label_encoder.classes_).index("SITTING")
STANDING_IDX = list(label_encoder.classes_).index("STANDING")
print(f"Input dim: {INPUT_DIM} | Classes: {list(label_encoder.classes_)}")

# %% [markdown]
# ## Reproduce per-subject normalization + RobustScaler (exact notebook logic)

# %%
def per_subject_normalize(X, subjects):
    X_norm = np.zeros_like(X, dtype=np.float64)
    for s in np.unique(subjects):
        mask = subjects == s
        mean = X[mask].mean(axis=0)
        std = X[mask].std(axis=0) + 1e-8
        X_norm[mask] = (X[mask] - mean) / std
    return X_norm.astype(np.float32)

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_idx, val_idx = next(gss.split(X_train_engineered, y_train_all, groups=subjects_train_all))

X_tr_raw, X_val_raw = X_train_engineered[train_idx], X_train_engineered[val_idx]
y_tr, y_val = y_train_all[train_idx], y_train_all[val_idx]
subj_tr, subj_val = subjects_train_all[train_idx], subjects_train_all[val_idx]

X_tr_norm = per_subject_normalize(X_tr_raw, subj_tr)
X_val_norm = per_subject_normalize(X_val_raw, subj_val)

scaler = RobustScaler()
X_tr_scaled = scaler.fit_transform(X_tr_norm)
X_val_scaled = scaler.transform(X_val_norm)

print(f"Train fold: {len(train_idx)} samples ({len(np.unique(subj_tr))} subjects) | "
      f"Val fold: {len(val_idx)} samples ({len(np.unique(subj_val))} subjects)")

# %% [markdown]
# ## Reproduce the notebook's model zoo + AdaBN ensemble mechanism

# %%
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim))
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.block(x))

class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_blocks=2, dropout=0.35):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)

class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, num_classes, dropout, act_type="gelu"):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       Swish() if act_type == "swish" else nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

MODEL_CONFIGS = [
    {"type": "res", "name": "ResMLP_512",    "hidden": 512, "blocks": 2, "dropout": 0.35},
    {"type": "res", "name": "ResMLP_768",    "hidden": 768, "blocks": 3, "dropout": 0.30},
    {"type": "mlp", "name": "512_256_128",   "hidden_sizes": [512, 256, 128], "dropout": 0.40, "act": "gelu"},
    {"type": "mlp", "name": "1024_512_256",  "hidden_sizes": [1024, 512, 256], "dropout": 0.45, "act": "swish"},
    {"type": "mlp", "name": "512_128",       "hidden_sizes": [512, 128], "dropout": 0.38, "act": "swish"},
]
SEEDS = [42, 43, 44]  # 5 configs x 3 seeds = 15 NN models -- matches notebook

def make_loader(X, y, seed):
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen)

def class_weights_for(y):
    counts = np.bincount(y, minlength=NUM_CLASSES)
    return torch.tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float32).to(device)

def train_nn_model(X_tr_, y_tr_, class_weights, cfg, epochs, seed, mixup_alpha=0.15):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if cfg["type"] == "res":
        model = ResMLP(INPUT_DIM, cfg["hidden"], NUM_CLASSES, num_blocks=cfg["blocks"], dropout=cfg["dropout"]).to(device)
    else:
        model = DeepMLP(INPUT_DIM, cfg["hidden_sizes"], NUM_CLASSES, cfg["dropout"], act_type=cfg.get("act", "gelu")).to(device)

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    loader = make_loader(X_tr_, y_tr_, seed)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb + torch.randn_like(xb) * 0.008
            if mixup_alpha > 0 and np.random.rand() < 0.4:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                index = torch.randperm(xb.size(0)).to(device)
                xb_mixed = lam * xb + (1 - lam) * xb[index]
                yb_a, yb_b = yb, yb[index]
                opt.zero_grad()
                out = model(xb_mixed)
                loss = lam * criterion(out, yb_a) + (1 - lam) * criterion(out, yb_b)
            else:
                opt.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def train_ensemble(X_tr_, y_tr_, class_weights, epochs, seed_offset=0):
    models = []
    for cfg in MODEL_CONFIGS:
        for seed in SEEDS:
            m = train_nn_model(X_tr_, y_tr_, class_weights, cfg, epochs=epochs, seed=seed + seed_offset)
            models.append(m)
    return models

def adabn_predict_proba(trained_model, X_subj_scaled):
    model = copy.deepcopy(trained_model)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.momentum = 1.0
    X_subj_t = torch.tensor(X_subj_scaled, dtype=torch.float32).to(device)
    model.train()
    with torch.no_grad():
        _ = model(X_subj_t)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(X_subj_t), dim=1).cpu().numpy()

def ensemble_predict_proba_by_subject(models, X_scaled, subjects):
    probs = np.zeros((len(X_scaled), NUM_CLASSES))
    for s in np.unique(subjects):
        idx = np.where(subjects == s)[0]
        s_probs = np.zeros((len(idx), NUM_CLASSES))
        for m in models:
            s_probs += adabn_predict_proba(m, X_scaled[idx])
        probs[idx] = s_probs / len(models)
    return probs

# %% [markdown]
# ## Train the base ensemble on the train fold (matches notebook's Ablation A exactly)

# %%
cw_tr = class_weights_for(y_tr)
start = time.time()
base_nn_models = train_ensemble(X_tr_scaled, y_tr, cw_tr, epochs=30, seed_offset=0)
val_base_probs = ensemble_predict_proba_by_subject(base_nn_models, X_val_scaled, subj_val)
baseline_f1 = f1_score(y_val, val_base_probs.argmax(axis=1), average="macro")
print(f"\nBase ensemble (matches notebook's champion mechanism) trained in {time.time()-start:.1f}s")
print(f"Baseline validation macro F1: {baseline_f1:.4f}  (should be close to the notebook's 0.9865)")

# %% [markdown]
# ## Train the SITTING/STANDING specialist on the SAME train fold, apply cascade only
# to ambiguous validation predictions

# %%
ss_mask_tr = np.isin(y_tr, [SITTING_IDX, STANDING_IDX])
specialist = LogisticRegression(max_iter=2000, C=1.0)
specialist.fit(X_tr_scaled[ss_mask_tr], y_tr[ss_mask_tr])

sorted_probs = np.sort(val_base_probs, axis=1)
margin = sorted_probs[:, -1] - sorted_probs[:, -2]
top2_idx = np.argsort(-val_base_probs, axis=1)[:, :2]
is_ss_pair = np.array([set(row) == {SITTING_IDX, STANDING_IDX} for row in top2_idx])
ambiguous = is_ss_pair & (margin < AMBIGUITY_MARGIN)

cascade_preds = val_base_probs.argmax(axis=1).copy()
if ambiguous.sum() > 0:
    cascade_preds[ambiguous] = specialist.predict(X_val_scaled[ambiguous])
cascade_f1 = f1_score(y_val, cascade_preds, average="macro")

print(f"\nRouted {ambiguous.sum()} / {len(val_idx)} validation samples to the SITTING/STANDING specialist")
print(f"Baseline (flat ensemble) validation macro F1: {baseline_f1:.4f}")
print(f"Cascade  (margin-based override) validation macro F1: {cascade_f1:.4f}")
print(f"Delta: {cascade_f1 - baseline_f1:+.4f}")

print("\n--- Baseline classification report ---")
print(classification_report(y_val, val_base_probs.argmax(axis=1), target_names=label_encoder.classes_, zero_division=0))
print("\n--- Cascade classification report ---")
print(classification_report(y_val, cascade_preds, target_names=label_encoder.classes_, zero_division=0))

# %% [markdown]
# ## Decision: only proceed to full-dataset retrain + test prediction if the cascade
# actually beat the baseline here. This mirrors the notebook's own validation-gated
# pattern -- don't ship something that didn't prove itself.

# %%
USE_CASCADE = cascade_f1 > baseline_f1
print(f"\n{'='*60}")
if USE_CASCADE:
    print(f"DECISION: Margin-based cascade WINS on validation ({cascade_f1:.4f} > {baseline_f1:.4f}).")
    print("Proceeding to retrain on full train.csv and apply to test.csv.")
else:
    print(f"DECISION: Margin-based cascade did NOT beat baseline ({cascade_f1:.4f} <= {baseline_f1:.4f}).")
    print("Stopping here -- not worth adding to the submission. Try adjusting")
    print("AMBIGUITY_MARGIN (currently {:.2f}) and re-running.".format(AMBIGUITY_MARGIN))
print("=" * 60)

# %% [markdown]
# ## Final: retrain on ALL of train.csv, predict on test.csv, save submission
# (only runs if the cascade validated above)

# %%
if USE_CASCADE:
    X_train_all_norm = per_subject_normalize(X_train_engineered, subjects_train_all)
    scaler_full = RobustScaler()
    X_train_all_scaled = scaler_full.fit_transform(X_train_all_norm)

    X_test_norm = per_subject_normalize(X_test_engineered, subjects_test)
    X_test_scaled = scaler_full.transform(X_test_norm)

    cw_all = class_weights_for(y_train_all)
    print("\nTraining final ensemble on full train.csv...")
    final_models = train_ensemble(X_train_all_scaled, y_train_all, cw_all, epochs=35, seed_offset=0)
    final_test_probs = ensemble_predict_proba_by_subject(final_models, X_test_scaled, subjects_test)

    print("Training final SITTING/STANDING specialist on full train.csv...")
    ss_mask_full = np.isin(y_train_all, [SITTING_IDX, STANDING_IDX])
    specialist_full = LogisticRegression(max_iter=2000, C=1.0)
    specialist_full.fit(X_train_all_scaled[ss_mask_full], y_train_all[ss_mask_full])

    sorted_test_probs = np.sort(final_test_probs, axis=1)
    test_margin = sorted_test_probs[:, -1] - sorted_test_probs[:, -2]
    test_top2_idx = np.argsort(-final_test_probs, axis=1)[:, :2]
    test_is_ss_pair = np.array([set(row) == {SITTING_IDX, STANDING_IDX} for row in test_top2_idx])
    test_ambiguous = test_is_ss_pair & (test_margin < AMBIGUITY_MARGIN)

    final_preds_numeric = final_test_probs.argmax(axis=1)
    if test_ambiguous.sum() > 0:
        final_preds_numeric[test_ambiguous] = specialist_full.predict(X_test_scaled[test_ambiguous])
    final_preds_labels = label_encoder.inverse_transform(final_preds_numeric)

    print(f"Routed {test_ambiguous.sum()} / {len(test_df)} test samples to the specialist.")

    test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(1, len(test_df) + 1)
    submission_df = pd.DataFrame({"id": test_ids, "Activity": final_preds_labels})

    assert len(submission_df) == len(test_df)
    assert submission_df["id"].isna().sum() == 0
    assert set(submission_df["Activity"].unique()).issubset(set(label_encoder.classes_))
    assert list(submission_df.columns) == ["id", "Activity"]

    submission_df.to_csv("submission_margin_cascade.csv", index=False)
    print(f"\nSaved submission_margin_cascade.csv ({len(submission_df)} rows).")
    print("\nPredicted class distribution:")
    print(submission_df["Activity"].value_counts())
else:
    print("\nSkipped final training -- cascade did not validate. No submission file written.")