# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 11:45:06 2025

@author: 72458991
"""

import pandas as pd
import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Load the dataset (adjust path as needed)
data = pd.read_excel('./allExcels_negatiu.xlsx')
data.fillna(0, inplace=True)

# View the first 10 rows
print(data.head(10))

# List all column names
print("\nColumns in the dataset:")
for col in data.columns:
    print(f"- {col}")

data_original = copy.deepcopy(data)

data["NGO Allocation to Country, Previous Year"] = (data["NGO Allocation to Country ($), Previous Year"] > 0).astype(int)


data['GDP Per Capita']                = np.log1p(data['GDP Per Capita'])
data['NGO Public Grant']       = np.log1p(data['NGO Public Grant'])
data['ODA Grants Spain']   = np.log1p(data['ODA Grants Spain'])



avoid_cols = ["Country_Year", 'NGO', 'Country', 'Year', 'Visitado','RuleofLaw','RegulatoryQuality','GovernmentEffectiveness','VoiceandAccountability','generic','cumulative_path_dependence',"NGO Allocation to Country ($), Previous Year"]
include_cols = [c for c in data.columns if c not in avoid_cols and c not in ['Budget_Previous_Year']]
#include_cols = [c for c in data.columns if c not in avoid_cols and c not in ['Project_Last_Year']]

print("\nColumns included:")
for col in include_cols:
    print(f"- {col}")


scaler = StandardScaler()
scaler.fit(data[include_cols].values)


training_LSTM = {}
y_LSTM = {}

pos = 0
for index, row in data.iterrows():
    
    if row["NGO"] not in training_LSTM:
        training_LSTM[row["NGO"]] = {}
        y_LSTM[row["NGO"]] = {}

    if row["Country"] not in training_LSTM[row["NGO"]]:
        training_LSTM[row["NGO"]][row["Country"]] = {}
        y_LSTM[row["NGO"]][row["Country"]]={}
    y_LSTM[row["NGO"]][row["Country"]][row["Year"]] = row["Visitado"]

    raw_feats    = row[include_cols]                                        # Series
    scaled_arr   = scaler.transform(raw_feats.values.reshape(1, -1))[0]     # array(F,)
    scaled_series= pd.Series(scaled_arr, index=include_cols)               # Series(F,)


    training_LSTM[row["NGO"]][row["Country"]][row["Year"]]= scaled_series




# Suppose these already exist:
#   training_LSTM[ngo][country][year] = pd.Series(..., index=include_cols)
#   y_LSTM       [ngo][country][year] = 0 or 1
import numpy as np

# 1) Gather and sort all years
all_years = sorted({
    year
    for ngo_dict in training_LSTM.values()
    for country_dict in ngo_dict.values()
    for year in country_dict.keys()
})
T = len(all_years)    # number of timesteps
F = len(include_cols) # number of features

# 2) Prepare outputs
X_list      = []   # flat list of sequence matrices
y_list      = []   # flat list of labels
lookup_list = []   # flat list of dicts, each carrying ngo, country, AND its full matrix
sequences   = {}   # nested dict: sequences[ngo][country] = matrix (T×F)

# 3) Build everything in one pass
for ngo, country_dict in training_LSTM.items():
    sequences.setdefault(ngo, {})
    for country, year_dict in country_dict.items():
        # a) build the T×F matrix, zero‐filled
        mat = np.zeros((T, F), dtype=np.float32)
        for t, year in enumerate(all_years):
            if year in year_dict:
                mat[t] = year_dict[year].values

        # b) store it in a nested dict (just like training_LSTM but leaves are matrices)
        sequences[ngo][country] = mat

        # c) append to the flat lists
        X_list.append(mat)
        y_list.append(y_LSTM[ngo][country].get(all_years[-1], 0))

        # d) in lookup, carry everything: ngo, country, the matrix, and the years
        lookup_list.append({
            'ngo':     ngo,
            'country': country,
            'years':   all_years,  # so you know row 0 → all_years[0], etc.
            'matrix':  mat
        })

# 4) stack into numpy arrays for training
X = np.stack(X_list, axis=0)      # (N, T, F)
y = np.array(y_list, dtype=np.int64)

np.sum(y==1)

import numpy as np

# Suppose X has shape (N, T, F)
# Map feature names to indices
feat_map = {name: i for i, name in enumerate(include_cols)}

# Which features to zero
feats_to_zero = [
    "Delegation",
    "NGO Allocation to Country, Previous Year"
]
feat_idx = [feat_map[f] for f in feats_to_zero if f in feat_map]

# Map years to row indices
year_map = {yr: i for i, yr in enumerate(all_years)}
years_to_zero = [2014, 2015, 2016]
year_idx = [year_map[y] for y in years_to_zero if y in year_map]

# Modify X in place
for yi in year_idx:
    X[:, yi, feat_idx] = 0



# ============================================================
# HYPERPARAMETERS - MODIFY THESE FOR MANUAL SEARCH
# ============================================================
SEARCH_SPACE = dict(
    HIDDEN_SIZE   =[64],
    NUM_LAYERS    =[2],
    DROPOUT       =[0],
    BATCH_SIZE    =[64],
    LEARNING_RATE =[1e-4],
    WEIGHT_DECAY  =[0],
    GRADIENT_CLIP =[2],
)

# Fixed training controls
MAX_EPOCHS = 1000
EARLY_STOP_PATIENCE = 100
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
USE_CLASS_WEIGHTS = False
RANDOM_STATE = 13
ATTENTION_HIDDEN_RATIO = 2  # hidden_size // ratio
N_SPLITS = 4
PRIMARY_METRIC = "ap"  # choose: 'roc_auc', 'f1', 'ap', etc.
THRESHOLD = 0.5




# ============================================================
# MODEL DEFINITION
# ============================================================
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, attention_ratio):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        # Attention
        if attention_ratio and attention_ratio > 0:
            self.attn = nn.Sequential(
                nn.Linear(hidden_size, max(1, hidden_size // attention_ratio)),
                nn.Tanh(),
                nn.Linear(max(1, hidden_size // attention_ratio), 1),
            )
        else:
            self.attn = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, return_attention=False):
        lstm_out, _ = self.lstm(x)                  # (B, T, H)
        scores = self.attn(lstm_out)                # (B, T, 1)
        weights = torch.softmax(scores, dim=1)      # (B, T, 1)
        context = torch.sum(weights * lstm_out, dim=1)  # (B, H)
        context = self.dropout(context)
        logits = self.fc(context)                   # (B, 1)
        if return_attention:
            return logits, weights
        return logits


# ============================================================
# UTILITIES
# ============================================================

import math
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, average_precision_score,
    precision_recall_fscore_support
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from typing import Dict, Any, Tuple

def set_seed(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_loaders(X_np, y_np, idx_train, idx_val, batch_size) -> Tuple[DataLoader, DataLoader]:
    X_train = torch.from_numpy(X_np[idx_train]).float()
    y_train = torch.from_numpy(y_np[idx_train]).float()
    X_val   = torch.from_numpy(X_np[idx_val]).float()
    y_val   = torch.from_numpy(y_np[idx_val]).float()

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def compute_class_weight(y_train_np):
    pos = (y_train_np == 1).sum()
    neg = (y_train_np == 0).sum()
    if pos == 0:  # avoid div by zero; no positives in fold
        return torch.tensor([1.0])
    return torch.tensor([neg / max(1, pos)], dtype=torch.float32)

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5) -> Dict[str, float]:
    model.eval()
    all_y = []
    all_p = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()
        logits = model(xb)
        probs = torch.sigmoid(logits).squeeze(1)
        all_y.append(yb.detach().cpu().numpy())
        all_p.append(probs.detach().cpu().numpy())
    y_true = np.concatenate(all_y)
    y_proba = np.concatenate(all_p)
    y_pred = (y_proba > threshold).astype(int)

    # Basic metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ROC / AP (guard against single-class val folds)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
    except Exception:
        roc_auc = np.nan
    try:
        ap = average_precision_score(y_true, y_proba)
    except Exception:
        ap = np.nan

    return dict(
        accuracy=acc, precision=prec, recall=rec, f1=f1,
        specificity=spec, roc_auc=roc_auc, ap=ap
    )

def train_one_fold(
    X_np, y_np, train_idx, val_idx, cfg, device
):
    """Train with early stopping on validation BCE (minimize). Scheduler also on val loss."""
    # Data
    train_loader, val_loader = build_loaders(
        X_np, y_np, train_idx, val_idx, batch_size=cfg["BATCH_SIZE"]
    )

    # Model
    model = LSTMWithAttention(
        input_size=X_np.shape[2],
        hidden_size=cfg["HIDDEN_SIZE"],
        num_layers=cfg["NUM_LAYERS"],
        dropout=cfg["DROPOUT"],
        attention_ratio=ATTENTION_HIDDEN_RATIO,
    ).to(device)

    # Loss (train objective)
    if USE_CLASS_WEIGHTS:
        pos_weight = compute_class_weight(y_np[train_idx]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optim & scheduler (both keyed to val loss)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["LEARNING_RATE"], weight_decay=cfg["WEIGHT_DECAY"]
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR
        ) if USE_SCHEDULER else None
    )

    # Early stopping (by val loss)
    best_epoch = 0
    best_state = None
    best_val_loss = float("inf")
    patience = 0
    eps_loss = 1e-6

    for epoch in range(1, MAX_EPOCHS + 1):
        # ----------------- Train -----------------
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["GRADIENT_CLIP"])
            optimizer.step()

            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / max(1, n)

        # ----------------- Validate -----------------
        model.eval()
        val_loss_accum, nv = 0.0, 0
        all_y, all_p = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).float()
                logits = model(xb)
                vloss = criterion(logits, yb.unsqueeze(1))
                val_loss_accum += vloss.item() * xb.size(0)
                nv += xb.size(0)
                # collect probs for logging AP (not used for early stopping)
                probs = torch.sigmoid(logits).squeeze(1)
                all_y.append(yb.cpu().numpy())
                all_p.append(probs.cpu().numpy())

        val_loss = val_loss_accum / max(1, nv)

        # optional logging metric (threshold-free, not used for stopping)
        try:
            y_true = np.concatenate(all_y)
            y_proba = np.concatenate(all_p)
            val_ap = average_precision_score(y_true, y_proba)
        except Exception:
            val_ap = float("nan")

        # LR scheduler on val loss
        if scheduler is not None:
            scheduler.step(val_loss)

        # -------- Early stopping on val loss (minimize) --------
        if val_loss < best_val_loss - eps_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            ap_txt = f"{val_ap:.4f}" if not np.isnan(val_ap) else "nan"
            print(f"Epoch {epoch:04d} | Train {train_loss:.4f} | ValLoss {val_loss:.4f} | (AP {ap_txt})")

        if patience >= EARLY_STOP_PATIENCE:
            break

    # Restore best (by lowest val loss) and evaluate fold metrics
    model.load_state_dict(best_state)
    fold_metrics = evaluate(model, val_loader, device, threshold=THRESHOLD)
    # (Optional) record the best loss epoch for analysis
    fold_metrics["early_stop_val_loss"] = float(best_val_loss)
    return best_epoch, fold_metrics



def config_iter(search_space: Dict[str, list]):
    keys = list(search_space.keys())
    for values in itertools.product(*[search_space[k] for k in keys]):
        yield dict(zip(keys, values))

def cfg_to_str(cfg: Dict[str, Any]) -> str:
    return (
        f"HS={cfg['HIDDEN_SIZE']}, L={cfg['NUM_LAYERS']}, DO={cfg['DROPOUT']}, "
        f"BS={cfg['BATCH_SIZE']}, LR={cfg['LEARNING_RATE']}, WD={cfg['WEIGHT_DECAY']}, "
        f"GC={cfg['GRADIENT_CLIP']}"
    )



# ============================================================
# RUN GRID SEARCH
# ============================================================
set_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

# Expect X, y already defined as numpy arrays: X: (N, T, F), y: (N,)
X_np = X.astype(np.float32, copy=False)
y_np = y.astype(np.int64, copy=False)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

results_rows = []
config_index = 0

for cfg in config_iter(SEARCH_SPACE):
    config_index += 1
    print("=" * 80)
    print(f"Config {config_index}: {cfg_to_str(cfg)}")
    print("-" * 80)

    fold_metrics_list = []
    fold_epochs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np), start=1):
        # Guard: if a fold has no positive or negative examples in train, still proceed (loss has pos_weight protection)
        best_epoch, fold_metrics = train_one_fold(X_np, y_np, train_idx, val_idx, cfg, device)
        fold_epochs.append(best_epoch)
        fold_metrics_list.append(fold_metrics)

        print(f"  Fold {fold} | best_epoch={best_epoch:>4} | "
              f"AUC={fold_metrics['roc_auc']:.3f} | F1={fold_metrics['f1']:.3f} | AP={fold_metrics['ap']:.3f}")

    # Aggregate across folds
    df_folds = pd.DataFrame(fold_metrics_list)
    means = df_folds.mean(numeric_only=True).to_dict()
    stds  = df_folds.std(numeric_only=True).to_dict()
    mean_epoch = float(np.mean(fold_epochs))

    row = dict(
        config_index=config_index,
        **cfg,
        mean_epoch=mean_epoch,
        **{f"mean_{k}": float(v) for k, v in means.items()},
        **{f"std_{k}": float(v) for k, v in stds.items()},
    )
    results_rows.append(row)

    print("  ─ Averages over 4 folds ─")
    print(f"    Epochs: {mean_epoch:.1f}")
    print(f"    Acc={means['accuracy']:.3f} ± {stds['accuracy']:.3f} | "
          f"Prec={means['precision']:.3f} ± {stds['precision']:.3f} | "
          f"Rec={means['recall']:.3f} ± {stds['recall']:.3f} | "
          f"F1={means['f1']:.3f} ± {stds['f1']:.3f}")
    print(f"    Spec={means['specificity']:.3f} ± {stds['specificity']:.3f} | "
          f"ROC-AUC={means['roc_auc']:.3f} ± {stds['roc_auc']:.3f} | "
          f"AP={means['ap']:.3f} ± {stds['ap']:.3f}")

# ============================================================
# SUMMARY & PLOT
# ============================================================
results_df = pd.DataFrame(results_rows)

# Choose best by PRIMARY_METRIC
best_row = results_df.sort_values(by=f"mean_{PRIMARY_METRIC}", ascending=False).iloc[0].to_dict()

print("\n" + "#" * 80)
print("GRID SEARCH SUMMARY")
print("#" * 80)
print(f"Total configurations explored: {len(results_df)}")
print(f"Primary selection metric: mean_{PRIMARY_METRIC}")
print("\nBEST CONFIGURATION:")
print(cfg_to_str(best_row))
print(
    f"\nMean metrics (± std) over {N_SPLITS} folds for best config:\n"
    f"  Acc: {best_row['mean_accuracy']:.3f} ± {best_row['std_accuracy']:.3f}\n"
    f"  Prec:{best_row['mean_precision']:.3f} ± {best_row['std_precision']:.3f}\n"
    f"  Rec: {best_row['mean_recall']:.3f} ± {best_row['std_recall']:.3f}\n"
    f"  F1:  {best_row['mean_f1']:.3f} ± {best_row['std_f1']:.3f}\n"
    f"  Spec:{best_row['mean_specificity']:.3f} ± {best_row['std_specificity']:.3f}\n"
    f"  AUC: {best_row['mean_roc_auc']:.3f} ± {best_row['std_roc_auc']:.3f}\n"
    f"  AP:  {best_row['mean_ap']:.3f} ± {best_row['std_ap']:.3f}\n"
    f"  Mean best epoch: {best_row['mean_epoch']:.1f}"
)

# Save CSV
results_df.to_csv("grid_search_results.csv", index=False)
print("\nSaved: grid_search_results.csv")

# Plot: show top K by primary metric
TOP_K = min(20, len(results_df))
plot_df = results_df.nlargest(TOP_K, f"mean_{PRIMARY_METRIC}").copy()
plot_df["label"] = plot_df.apply(
    lambda r: f"#{int(r['config_index'])} | HS={int(r['HIDDEN_SIZE'])},L={int(r['NUM_LAYERS'])},DO={r['DROPOUT']},"
              f"BS={int(r['BATCH_SIZE'])},LR={r['LEARNING_RATE']},WD={r['WEIGHT_DECAY']},GC={r['GRADIENT_CLIP']}",
    axis=1
)

plt.figure(figsize=(12, max(6, TOP_K * 0.45)))
sns.barplot(
    data=plot_df,
    y="label",
    x=f"mean_{PRIMARY_METRIC}",
    xerr=plot_df[f"std_{PRIMARY_METRIC}"] if f"std_{PRIMARY_METRIC}" in plot_df else None,
    orient="h"
)
plt.xlabel(f"Mean {PRIMARY_METRIC.upper()} (± std) across {N_SPLITS} folds")
plt.ylabel("Configuration")
plt.title(f"Top {TOP_K} Configurations by Mean {PRIMARY_METRIC.upper()}")
plt.tight_layout()
plt.savefig("grid_search_summary.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: grid_search_summary.png")








# ============================================================
# PHASE 2: TRAIN FINAL MODEL ON 100% DATA (using best config)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

print("\n" + "=" * 60)
print("PHASE 2: Training final model on 100% data (best config from grid search)")
print("=" * 60)

# ---- Best configuration from your GRID SEARCH SUMMARY ----
BEST_HIDDEN_SIZE = 64
BEST_NUM_LAYERS  = 2
BEST_DROPOUT     = 0
BEST_BATCH_SIZE  = 64
BEST_LR          = 0.0001
BEST_WEIGHT_DECAY= 0
BEST_GRAD_CLIP   = 2.0
# From "Mean best epoch: 23.5" -> use median/rounded epochs
E_STAR = 50

print(f"Using best config: HS={BEST_HIDDEN_SIZE}, L={BEST_NUM_LAYERS}, "
      f"DO={BEST_DROPOUT}, BS={BEST_BATCH_SIZE}, LR={BEST_LR}, "
      f"WD={BEST_WEIGHT_DECAY}, GC={BEST_GRAD_CLIP}, E*={E_STAR}")

# ---- Rebuild full dataset loader with BEST_BATCH_SIZE ----
X_full = torch.from_numpy(X).float() if not isinstance(X, torch.Tensor) else X.float()
y_full = torch.from_numpy(y).float() if not isinstance(y, torch.Tensor) else y.float()

full_dataset = TensorDataset(X_full, y_full)
full_loader  = DataLoader(full_dataset, batch_size=BEST_BATCH_SIZE, shuffle=True)

# ---- Reinitialize model with best architecture ----
final_model = LSTMWithAttention(
    input_size=X_full.shape[2],
    hidden_size=BEST_HIDDEN_SIZE,
    num_layers=BEST_NUM_LAYERS,
    dropout=BEST_DROPOUT,
    attention_ratio=ATTENTION_HIDDEN_RATIO
).to(device)

# ---- Criterion with class weights computed on FULL data ----
if USE_CLASS_WEIGHTS:
    # pos_weight = (#neg / #pos) computed on full labels
    y_np = y if isinstance(y, np.ndarray) else y_full.cpu().numpy()
    n_pos = float((y_np == 1).sum())
    n_neg = float((y_np == 0).sum())
    pos_weight_full = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)
    criterion_full = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
    print(f"Class weights enabled on full data: pos_weight={pos_weight_full.item():.3f}")
else:
    criterion_full = nn.BCEWithLogitsLoss()

# ---- Optimizer (no val scheduler in full-data refit) ----
optimizer_full = torch.optim.AdamW(
    final_model.parameters(),
    lr=BEST_LR,
    weight_decay=BEST_WEIGHT_DECAY
)

# ---- Fixed-epoch training for E* epochs ----
loss_history = []
for epoch in range(1, E_STAR + 1):
    final_model.train()
    running = 0.0
    n_seen = 0
    for xb, yb in full_loader:
        xb = xb.to(device)
        yb = yb.to(device).float()

        optimizer_full.zero_grad()
        logits = final_model(xb)
        loss = criterion_full(logits, yb.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=BEST_GRAD_CLIP)
        optimizer_full.step()

        running += loss.item() * xb.size(0)
        n_seen  += xb.size(0)

    avg_loss = running / max(1, n_seen)
    loss_history.append(avg_loss)
    if epoch % 5 == 0 or epoch == 1 or epoch == E_STAR:
        print(f"[FULL] Epoch {epoch:02d}/{E_STAR} — TrainLoss: {avg_loss:.4f}")

print("\n✓ Final model trained on 100% of data")

# ---- Save with config in filename for traceability ----
final_ckpt = (
    f"final_full_LSTM_h{BEST_HIDDEN_SIZE}_L{BEST_NUM_LAYERS}_do{BEST_DROPOUT}_"
    f"bs{BEST_BATCH_SIZE}_lr{BEST_LR}_wd{BEST_WEIGHT_DECAY}_gc{BEST_GRAD_CLIP}_E{E_STAR}.pth"
)
torch.save(final_model.state_dict(), final_ckpt)
print(f"Model saved to: {final_ckpt}")

# ============================================================
# RESULTS SUMMARY
# ============================================================




device = torch.device("cpu")
final_model = final_model.to(device)
final_model.eval()

def predict_fn(x_flat):
    batch_size = x_flat.shape[0]
    x_reshaped = x_flat.reshape(batch_size, T, F)
    x_tensor = torch.FloatTensor(x_reshaped)  # CPU tensor
    
    with torch.no_grad():
        logits = final_model(x_tensor, return_attention=False)
        probs = torch.sigmoid(logits).squeeze().numpy()
    
    if probs.ndim == 0:
        probs = np.array([probs])
    
    return probs





# ============================================================
# CONFUSION MATRIX
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# Get predictions for all data
X_flat = X.reshape(len(X), -1)
predictions_proba = predict_fn(X_flat)
predictions = (predictions_proba > 0.5).astype(int)

# Create confusion matrix
cm = confusion_matrix(y, predictions)

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Confusion Matrix Heatmap
ax = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted No Visit', 'Predicted Visit'],
            yticklabels=['Actual No Visit', 'Actual Visit'],
            ax=ax, cbar_kws={'label': 'Count'})

# Add percentages
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / cm.sum() * 100
        ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                ha='center', va='center', fontsize=9, color='gray')

ax.set_title('Confusion Matrix')

# Plot 2: Normalized Confusion Matrix
ax = axes[0, 1]
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd',
            xticklabels=['Predicted No Visit', 'Predicted Visit'],
            yticklabels=['Actual No Visit', 'Actual Visit'],
            ax=ax, cbar_kws={'label': 'Percentage'})
ax.set_title('Normalized Confusion Matrix (by actual class)')

# Plot 3: ROC Curve
ax = axes[1, 0]
fpr, tpr, thresholds = roc_curve(y, predictions_proba)
roc_auc = auc(fpr, tpr)

ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# Plot 4: Metrics Summary
ax = axes[1, 1]
ax.axis('off')

metrics_text = f"""
CLASSIFICATION METRICS SUMMARY

Confusion Matrix Breakdown:
  • True Negatives (TN):  {tn:,} - Correctly predicted no visit
  • False Positives (FP): {fp:,} - Incorrectly predicted visit
  • False Negatives (FN): {fn:,} - Incorrectly predicted no visit
  • True Positives (TP):  {tp:,} - Correctly predicted visit

Performance Metrics:
  • Accuracy:    {accuracy:.3f} - Overall correct predictions
  • Precision:   {precision:.3f} - Of predicted visits, how many were correct
  • Recall:      {recall:.3f} - Of actual visits, how many were found
  • F1 Score:    {f1:.3f} - Harmonic mean of precision and recall
  • Specificity: {specificity:.3f} - Of actual no-visits, how many were correct
  • AUC-ROC:     {roc_auc:.3f} - Area under ROC curve

Model Performance Assessment:
  • {'✓ Good accuracy' if accuracy > 0.7 else '⚠ Low accuracy'} ({accuracy:.1%})
  • {'✓ Good precision' if precision > 0.7 else '⚠ Low precision'} ({precision:.1%})
  • {'✓ Good recall' if recall > 0.7 else '⚠ Low recall'} ({recall:.1%})
  • {'✓ Balanced' if abs(precision - recall) < 0.15 else '⚠ Imbalanced precision/recall'}
"""

ax.text(0.05, 0.5, metrics_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

plt.suptitle('Model Performance Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional analysis by threshold
print("="*60)
print("PERFORMANCE ANALYSIS")
print("="*60)
print(f"\nCurrent threshold: 0.5")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

# Test different thresholds
print("\nPerformance at different thresholds:")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    preds_at_threshold = (predictions_proba > threshold).astype(int)
    cm_temp = confusion_matrix(y, preds_at_threshold)
    tn_t, fp_t, fn_t, tp_t = cm_temp.ravel()
    acc_t = (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t)
    prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    print(f"  Threshold {threshold}: Acc={acc_t:.3f}, Prec={ prec_t:.3f}, Rec={rec_t:.3f}")

# Print classification report
print("\nDetailed Classification Report:")
print(classification_report(y, predictions, target_names=['No Visit', 'Visit']))





import shap
import numpy as np
import torch

# ============================================================
# PREPARE DATA FOR SHAP
# ============================================================
X_full = X  # Shape: (N, 8, F)
y_full = y  

N = X_full.shape[0]
T = X_full.shape[1]  # 8 years
F = X_full.shape[2]  # features

print(f"Dataset: {N} samples, {T} timesteps, {F} features")

# Flatten for KernelExplainer
X_flat = X_full.reshape(N, T*F)

# ============================================================
# SETUP MODEL AND PREDICTION FUNCTION
# ============================================================
# SHAP doesn't work well with CUDA - must use CPU
device = torch.device("cpu")
final_model = final_model.to(device)
final_model.eval()








# ============================================================
# COMPUTE SHAP VALUES - ALL AT ONCE
# ============================================================

import shap
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Use a subset as background (100-200 samples is usually enough)
# Stratified sampling to maintain class balance
background_idx, _ = train_test_split(
    range(len(X)), 
    test_size=0.95,  # Keep 10% as background
    stratify=y,
    random_state=42
)

X_background = X[background_idx]  # Much smaller background
X_background_flat = X_background.reshape(len(X_background), T*F)

print(f"Background samples: {len(X_background)} (instead of {len(X)})")

# Setup model
device = torch.device("cpu")
final_model = final_model.to(device)
final_model.eval()

# Create explainer with SUBSET as background
explainer = shap.KernelExplainer(predict_fn, X_background_flat)

# Still explain ALL samples
X_flat = X.reshape(len(X), T*F)
shap_values = explainer.shap_values(X_flat, nsamples=1024)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values_3d = shap_values.reshape(len(X), T, F)




import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#global_min = shap_values_3d.min()
#global_max = shap_values_3d.max()


global_min = np.abs(shap_values_3d).mean(axis=0).min()
global_max = np.abs(shap_values_3d).mean(axis=0).max()

print(f"Global SHAP min: {global_min:.4f}")
print(f"Global SHAP max: {global_max:.4f}")


# ============================================================
# GENERIC IMAGE with FIXED COLOR SCALE
# ============================================================

# Load SHAP values
N, T, F = shap_values_3d.shape

# Calculate importance for each feature at each year
feature_time_importance = np.abs(shap_values_3d).mean(axis=0)  # Shape: (T, F)

# Create feature names
if len(include_cols) != F:
    feature_names = [f"Feature_{i}" for i in range(F)]
else:
    feature_names = include_cols

years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]

plt.figure(figsize=(12, 8))

sns.heatmap(
    feature_time_importance.T, 
    xticklabels=years,
    yticklabels=feature_names,
    cmap='RdBu_r',
    vmin=global_min, vmax=global_max,   # fixed global scale
    cbar_kws={'label': 'Mean |SHAP value|'},
    annot=np.round(feature_time_importance.T, 3),   # add SHAP values inside cells
    fmt=".3f",
    annot_kws={"color": "white", "fontsize": 8}     # white text inside
)

plt.xlabel('Year', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance Across Years (SHAP Analysis)', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance_temporal.png', dpi=300, bbox_inches='tight')
plt.show()






# ============================================================
# FEATURE IMPORTANCE for POSITIVE y (with FIXED COLOR SCALE)
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Compute mean SHAP values restricted to y=1
mask_pos = (y == 1)
shap_values_pos = shap_values_3d[mask_pos]  # Shape: (N_pos, T, F)
feature_time_importance_pos = np.abs(shap_values_pos).mean(axis=0)  # (T, F)

global_min_pos = np.abs(shap_values_pos).mean(axis=0).min()
global_max_pos = np.abs(shap_values_pos).mean(axis=0).max()

# 2) Feature names
if len(include_cols) != shap_values_pos.shape[2]:
    feature_names = [f"Feature_{i}" for i in range(shap_values_pos.shape[2])]
else:
    feature_names = include_cols

years = [2009,2010,2011,2012,2013,2014,2015,2016]
data_mat = feature_time_importance_pos.T   # (F, T)

# 3) Plot heatmap
plt.figure(figsize=(12, 8))
cmap = plt.get_cmap('RdBu_r')
ax = sns.heatmap(data_mat,
                 xticklabels=years,
                 yticklabels=feature_names,
                 cmap=cmap,
                 vmin=global_max_pos, vmax=global_min_pos,
                 cbar_kws={'label': 'Mean |SHAP value| (y=1 only)'},
                 annot=False)

# 4) Add dynamic text annotations
norm = plt.Normalize(vmin=global_min_pos, vmax=global_max_pos)
for i in range(data_mat.shape[0]):    # features
    for j in range(data_mat.shape[1]):  # years
        val = data_mat[i, j]
        if not np.isfinite(val): 
            continue
        r, g, b, _ = cmap(norm(val))
        luminance = 0.2126*r + 0.7152*g + 0.0722*b
        text_color = "white" if luminance < 0.5 else "black"
        ax.text(j+0.5, i+0.5, f"{val:.3f}",
                ha="center", va="center",
                color=text_color, fontsize=8)

plt.xlabel('Year', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance Across Years (SHAP Analysis, y=1 only)', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance_temporal_y1.png', dpi=300, bbox_inches='tight')
plt.show()






# ============================================================
# FEATURE IMPORTANCE for y = 0 (with FIXED COLOR SCALE)
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Compute mean SHAP values restricted to y=1
mask_neg = (y == 0)
shap_values_neg = shap_values_3d[mask_neg]  # Shape: (N_pos, T, F)
feature_time_importance_neg = np.abs(shap_values_neg).mean(axis=0)  # (T, F)

global_min_neg = np.abs(shap_values_neg).mean(axis=0).min()
global_max_neg = np.abs(shap_values_neg).mean(axis=0).max()

# 2) Feature names
if len(include_cols) != shap_values_neg.shape[2]:
    feature_names = [f"Feature_{i}" for i in range(shap_values_neg.shape[2])]
else:
    feature_names = include_cols

years = [2009,2010,2011,2012,2013,2014,2015,2016]
data_mat = feature_time_importance_neg.T   # (F, T)

# 3) Plot heatmap
plt.figure(figsize=(12, 8))
cmap = plt.get_cmap('RdBu_r')
ax = sns.heatmap(data_mat,
                 xticklabels=years,
                 yticklabels=feature_names,
                 cmap=cmap,
                 vmin=global_min_neg, vmax=global_max_neg,
                 cbar_kws={'label': 'Mean |SHAP value| (y=1 only)'},
                 annot=False)

# 4) Add dynamic text annotations
norm = plt.Normalize(vmin=global_min_pos, vmax=global_max_pos)
for i in range(data_mat.shape[0]):    # features
    for j in range(data_mat.shape[1]):  # years
        val = data_mat[i, j]
        if not np.isfinite(val): 
            continue
        r, g, b, _ = cmap(norm(val))
        luminance = 0.2126*r + 0.7152*g + 0.0722*b
        text_color = "white" if luminance < 0.5 else "black"
        ax.text(j+0.5, i+0.5, f"{val:.3f}",
                ha="center", va="center",
                color=text_color, fontsize=8)

plt.xlabel('Year', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance Across Years (SHAP Analysis, y=0 only)', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance_temporal_y1.png', dpi=300, bbox_inches='tight')
plt.show()




# ============================================================
# pearson with visit
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Inputs from your pipeline:
# X: (N, T, F)
# y: (N,)
# include_cols: list of feature names

years = ["2009","2010","2011","2012","2013","2014","2015","2016"]
N, T, F = X.shape
assert T == len(years), f"X has {T} timesteps, expected {len(years)}"
assert F == len(include_cols), f"X has {F} features, but include_cols has {len(include_cols)}"

def safe_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    try:
        r, _ = pearsonr(x, y)
        return float(np.clip(r, -1, 1))
    except Exception:
        return np.nan

# Build correlation matrix (F x T)
corr_matrix = np.zeros((F, T))
corr_matrix[:] = np.nan
for t in range(T):
    Xt = X[:, t, :]   # features at year t across all samples
    for f in range(F):
        corr_matrix[f, t] = safe_corr(Xt[:, f], y)

# Plot heatmap with annotations
plt.figure(figsize=(14, 9))
sns.heatmap(
    corr_matrix,
    xticklabels=years,
    yticklabels=include_cols,
    cmap="RdBu_r",
    vmin=-1, vmax=1,
    center=0,
    annot=True, fmt=".2f",  # << add correlation numbers
    cbar_kws={'label': 'Pearson correlation with target'}
)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Feature–Target Correlation Across Years", fontsize=14)
plt.tight_layout()
plt.show()







# ============================================================
# VIOLIN PLOTS
# ============================================================
# Unified split-violin of SHAP for Project_Last_Year across years
# Horizontal layout (wide, short) and corrected y-axis label.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

sns.set_style("whitegrid")

feat = "NGO Allocation to Country, Previous Year"
feat_idx = include_cols.index(feat)

# Slice SHAP + feature values
shap_feat = shap_values_3d[:, :, feat_idx]   # (N, T)
vals_raw  = X[:, :, feat_idx]                # (N, T)

# Rebuild binary 0/1 per year (robust to scaling/standardization)
vals_bin = np.zeros_like(vals_raw, dtype=int)
for t in range(vals_raw.shape[1]):
    col = vals_raw[:, t]
    u = np.unique(np.round(col, 6))
    if u.size == 2:
        vals_bin[:, t] = (col == col.max()).astype(int)
    else:
        thr = 0.5 * (col.min() + col.max())
        vals_bin[:, t] = (col > thr).astype(int)

# Tidy DF
records = []
for t, y in enumerate(years):
    records.append(pd.DataFrame({
        "Year": y,
        "SHAP": shap_feat[:, t],
        feat:  vals_bin[:, t].astype(int)
    }))
df = pd.concat(records, ignore_index=True)
df["Year"] = pd.Categorical(df["Year"], categories=years, ordered=True)

# --- Colors (reuse for violins, legend, and colorful counts)
PALETTE = {0: "#4C78A8", 1: "#F58518"}

# --- Figure with extra margins for legend (top) and counts (bottom)
fig, ax = plt.subplots(figsize=(14, 3.6))
# Slightly larger bottom margin to fit two colored lines of counts
plt.subplots_adjust(top=0.78, bottom=0.34)

# Split violin
vp = sns.violinplot(
    data=df, x="Year", y="SHAP",
    hue=feat, hue_order=[0, 1],
    split=True, inner="quartile", cut=0, linewidth=0.8,
    palette=PALETTE
)

# Axes
ax.set_ylim(-0.25, 0.60)
ax.set_yticks(np.arange(-0.25, 0.61, 0.1))  # <--- new line

ax.axhline(0, color="k", lw=1, ls="--", alpha=0.5)
ax.set_ylabel("SHAP value (contribution to probability)")
ax.set_xlabel("Year")

# Colorful counts BELOW the x-axis (two lines, each in its class color)
counts = df.groupby(["Year", feat]).size().unstack(fill_value=0)
for i, yr in enumerate(df["Year"].cat.categories):
    n0 = int(counts.loc[yr, 0]) if 0 in counts.columns else 0
    n1 = int(counts.loc[yr, 1]) if 1 in counts.columns else 0

    # First line (0=...) a bit higher, second line (1=...) below it
    ax.text(i, -0.18, f"0={n0}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=PALETTE[0], clip_on=False)

    ax.text(i, -0.28, f"1={n1}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=PALETTE[1], clip_on=False)

# Clean horizontal legend UNDER the title (no overlap)
handles = [Patch(facecolor=PALETTE[0]), Patch(facecolor=PALETTE[1])]
labels  = [f"{feat}: 0", f"{feat}: 1"]
leg = fig.legend(handles, labels,
                 loc="upper center", bbox_to_anchor=(0.5, 0.93),
                 ncol=2, frameon=True, title=None)

# Suptitle at the very top
fig.suptitle(f"{feat}: SHAP distributions across years (0 vs 1)",
             y=0.99, fontsize=14, fontweight="bold")

plt.savefig("project_last_year_shap_split_violin_colorful_counts.png",
            dpi=300, bbox_inches="tight")
plt.show()





# ============================================================
# quartiles
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

# df: columns ["Year", "SHAP", "Project_Last_Year"]
# keep your year ordering if needed:
# df["Year"] = pd.Categorical(df["Year"], categories=years, ordered=True)

q_levels = [0.00, 0.25, 0.50, 0.75, 1.00]

quant_0 = (df[df["NGO Allocation to Country, Previous Year"] == 0]
           .groupby("Year")["SHAP"].quantile(q_levels).unstack()
           .rename(columns={0.00:"0", 0.25:"0.25", 0.50:"0.5", 0.75:"0.75", 1.00:"1"})
           .sort_index())

quant_1 = (df[df["NGO Allocation to Country, Previous Year"] == 1]
           .groupby("Year")["SHAP"].quantile(q_levels).unstack()
           .rename(columns={0.00:"0", 0.25:"0.25", 0.50:"0.5", 0.75:"0.75", 1.00:"1"})
           .sort_index())

# Asymmetric, zero-centered normalization: use full blue range for negatives and full red for positives
vmin = min(quant_0.min().min(), quant_1.min().min())  # most negative value (~ -0.04)
vmax = max(quant_0.max().max(), quant_1.max().max())  # most positive value (~ +0.23)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

cmap = "RdBu_r"
sns.set_style("white")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
plt.subplots_adjust(right=0.90, wspace=0.12)

h0 = sns.heatmap(
    quant_0, ax=axes[0],
    cmap=cmap, norm=norm,
    annot=True, fmt=".3f", linewidths=0, cbar=False,
    annot_kws={"fontsize":9}
)
axes[0].set_title("Quantiles (Project_Last_Year = 0)")
axes[0].set_xlabel("Quantile")
axes[0].set_ylabel("Year")

h1 = sns.heatmap(
    quant_1, ax=axes[1],
    cmap=cmap, norm=norm,
    annot=True, fmt=".3f", linewidths=0, cbar=False,
    annot_kws={"fontsize":9}
)
axes[1].set_title("Quantiles (Project_Last_Year = 1)")
axes[1].set_xlabel("Quantile")
axes[1].set_ylabel("")

# One shared colorbar in its own axis (won’t cover the right heatmap)
cbar_ax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
cb = fig.colorbar(h1.collections[0], cax=cbar_ax)
cb.set_label("SHAP value (zero-centered)")

for ax in axes:
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"], rotation=0)

fig.suptitle("Per-year SHAP Quantiles (split by Project_Last_Year)", y=0.98, fontsize=16)
plt.savefig("shap_quantiles_split_heatmaps_twoslopnorm.png", dpi=300, bbox_inches="tight")
plt.show()







# ============================================================
# Pearson
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# X: (N=5400, T=8, F=12)
# If your var is named `x`, just do: X = x
alloc_idx = 11                     # column 11 = NGO Allocation (prev year)
A = X[:, :, alloc_idx]             # shape (N, T) → values of that feature over years

T = A.shape[1]
corr = np.empty((T, T), dtype=float)

# Pearson correlation across years (computed over the N samples)
for i in range(T):
    xi = A[:, i]
    for j in range(T):
        xj = A[:, j]
        if np.std(xi) == 0 or np.std(xj) == 0:
            corr[i, j] = np.nan     # undefined if a column is constant
        else:
            corr[i, j] = pearsonr(xi, xj)[0]

# --- Plot heatmap ---
years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]  # adjust if needed
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, vmin=-1, vmax=1)

ax.set_xticks(range(T)); ax.set_yticks(range(T))
ax.set_xticklabels(years, rotation=45, ha="right")
ax.set_yticklabels(years)
ax.set_title("NGO Allocation (prev year): Pearson correlation across years")
plt.colorbar(im, ax=ax, label="Pearson r")

# annotate cells
for i in range(T):
    for j in range(T):
        txt = "nan" if np.isnan(corr[i, j]) else f"{corr[i, j]:.2f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.show()










# ============================================================
# Comparison with logistic model
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, average_precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================
# LOGISTIC REGRESSION BASELINE (same train/val split)
# ============================================================

# Flatten sequences into 2D features (N, T*F)
N, T, F = X.shape
X_train_flat = X_train.reshape(len(X_train), T * F)
X_val_flat   = X_val.reshape(len(X_val), T * F)

# Scale features
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_val_flat   = scaler.transform(X_val_flat)

# Train logistic regression (balanced for imbalanced data)
# Elastic Net logistic regression
log_reg = LogisticRegression(
    penalty="elasticnet",
    l1_ratio=0.5,        # mix: 0=L2, 1=L1  (tune as needed)
    C=1.0,               # inverse regularization strength (smaller = stronger)
    solver="saga",       # required for elastic net
    max_iter=2000,
    class_weight="balanced",  # helpful for your imbalanced target
    n_jobs=-1
)
log_reg.fit(X_train_flat, y_train)

# Validation predictions
y_proba_val = log_reg.predict_proba(X_val_flat)[:, 1]
y_pred_val  = (y_proba_val > 0.5).astype(int)

# ============================================================
# PERFORMANCE ANALYSIS (same style as LSTM block)
# ============================================================

cm = confusion_matrix(y_val, y_pred_val)
tn, fp, fn, tp = cm.ravel()

accuracy     = (tp + tn) / cm.sum()
precision    = tp / (tp + fp) if (tp + fp) > 0 else 0
recall       = tp / (tp + fn) if (tp + fn) > 0 else 0
f1           = f1_score(y_val, y_pred_val)
specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0
fpr, tpr, _  = roc_curve(y_val, y_proba_val)
roc_auc      = auc(fpr, tpr)
avg_prec     = average_precision_score(y_val, y_proba_val)

# 2x2 performance panel
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Confusion matrix (counts)
ax = axes[0,0]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Pred 0','Pred 1'],
            yticklabels=['Actual 0','Actual 1'],
            ax=ax, cbar_kws={'label':'Count'})
for i in range(2):
    for j in range(2):
        pct = cm[i,j] / cm.sum() * 100
        ax.text(j+0.5, i+0.72, f'({pct:.1f}%)',
                ha='center', va='center', fontsize=9, color='gray')
ax.set_title("Confusion Matrix (Validation)")

# Normalized confusion matrix
ax = axes[0,1]
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="YlOrRd",
            xticklabels=['Pred 0','Pred 1'],
            yticklabels=['Actual 0','Actual 1'],
            ax=ax, cbar_kws={'label':'Percentage'})
ax.set_title("Normalized Confusion Matrix (Validation)")

# ROC curve
ax = axes[1,0]
ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC={roc_auc:.3f})", color="darkorange")
ax.plot([0,1],[0,1],"--", color="navy")
ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (Validation)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# Metrics summary text
ax = axes[1,1]
ax.axis("off")
metrics_text = f"""
VALIDATION METRICS (Logistic @ thr=0.50)

Counts:
  • TN: {tn:,}   • FP: {fp:,}
  • FN: {fn:,}   • TP: {tp:,}

Performance:
  • Accuracy:     {accuracy:.3f}
  • Precision:    {precision:.3f}
  • Recall:       {recall:.3f}
  • F1 score:     {f1:.3f}
  • Specificity:  {specificity:.3f}
  • AUC-ROC:      {roc_auc:.3f}
  • AP (PR AUC):  {avg_prec:.3f}
"""
ax.text(0.05, 0.5, metrics_text, transform=ax.transAxes,
        fontsize=10, va="center", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.25))

plt.suptitle("Validation Performance: Logistic Regression Baseline", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Threshold sweep
print("="*60)
print("LOGISTIC REGRESSION VALIDATION PERFORMANCE BY THRESHOLD")
print("="*60)
for thr in [0.30, 0.40, 0.50, 0.60, 0.70]:
    y_pred_thr = (y_proba_val > thr).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_val, y_pred_thr).ravel()
    acc_t = (tp_t+tn_t)/cm.sum()
    prec_t = tp_t/(tp_t+fp_t) if tp_t+fp_t>0 else 0
    rec_t = tp_t/(tp_t+fn_t) if tp_t+fn_t>0 else 0
    f1_t = 2*prec_t*rec_t/(prec_t+rec_t) if prec_t+rec_t>0 else 0
    print(f"thr={thr:.2f}: Acc={acc_t:.3f} | Prec={prec_t:.3f} | Rec={rec_t:.3f} | F1={f1_t:.3f}")

# Classification report
print("\nDetailed classification report (validation):")
print(classification_report(y_val, y_pred_val, target_names=["Class 0","Class 1"]))



# Logistic coefficients (shape: (1, T*F))
coefs = log_reg.coef_.reshape(T, F).T   # reshape into (F, T)

plt.figure(figsize=(14, 9))
sns.heatmap(
    coefs,
    xticklabels=years,
    yticklabels=include_cols,
    cmap="coolwarm",
    center=0,
    annot=True, fmt=".2f",
    cbar_kws={'label': 'Logistic regression coefficient'}
)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Logistic Regression Coefficients Across Years", fontsize=14)
plt.tight_layout()
plt.show()







# ============================================================
# Print rankings
# ============================================================
year_importance = feature_time_importance.mean(axis=1)
most_important_year = years[np.argmax(year_importance)]

feature_importance = feature_time_importance.mean(axis=0)
most_important_feature = feature_names[np.argmax(feature_importance)]

print(f"Most important year overall: {most_important_year}")
print(f"Most important feature overall: {most_important_feature}")
print(f"\nYear importance ranking:")
for year, imp in sorted(zip(years, year_importance), key=lambda x: x[1], reverse=True):
    print(f"  {year}: {imp:.4f}")





# ============================================================
# isolate SHAP impact
# ============================================================


# Years (indices: 2009->0, ..., 2016->7)
year_2015_idx = 6
year_2016_idx = 7

# ---------- Project_Last_Year ----------
ply_idx = include_cols.index("Project_Last_Year")
ply_shap = shap_values_3d[:, :, ply_idx]   # (N, T)
ply_vals = X[:, :, ply_idx]                # (N, T)
min_val, max_val = ply_vals.min(), ply_vals.max()


shap_2015 = ply_shap[:, year_2015_idx]
shap_2016 = ply_shap[:, year_2016_idx]

print("="*60)
print("FEATURE: Project_Last_Year")
print("="*60)
print(f"Mean SHAP (2015): {shap_2015.mean():.4f}")
print(f"Mean SHAP (2016): {shap_2016.mean():.4f}")



shap_2015_when_0 = shap_2015[ply_vals[:, year_2015_idx] == min_val]
shap_2015_when_1 = shap_2015[ply_vals[:, year_2015_idx] == max_val]
shap_2016_when_0 = shap_2016[ply_vals[:, year_2016_idx] == min_val]
shap_2016_when_1 = shap_2016[ply_vals[:, year_2016_idx] == max_val]

print("\n2015:")
print(f"  Mean SHAP when Project_Last_Year=0: {shap_2015_when_0.mean():.4f}  (n={len(shap_2015_when_0)})")
print(f"  Mean SHAP when Project_Last_Year=1: {shap_2015_when_1.mean():.4f}  (n={len(shap_2015_when_1)})")

print("\n2016:")
print(f"  Mean SHAP when Project_Last_Year=0: {shap_2016_when_0.mean():.4f}  (n={len(shap_2016_when_0)})")
print(f"  Mean SHAP when Project_Last_Year=1: {shap_2016_when_1.mean():.4f}  (n={len(shap_2016_when_1)})")


# ---------- Delegation ----------
del_idx = include_cols.index("Delegation")
del_shap = shap_values_3d[:, :, del_idx]
del_vals = X[:, :, del_idx]
min_del, max_del = del_vals.min(), del_vals.max()


del_shap_2015 = del_shap[:, year_2015_idx]
del_shap_2016 = del_shap[:, year_2016_idx]

print("\n" + "="*60)
print("FEATURE: Delegation")
print("="*60)
print(f"Mean SHAP (2015): {del_shap_2015.mean():.4f}")
print(f"Mean SHAP (2016): {del_shap_2016.mean():.4f}")

del_2015_when_0 = del_shap_2015[del_vals[:, year_2015_idx] == min_del]
del_2015_when_1 = del_shap_2015[del_vals[:, year_2015_idx] == max_del]
del_2016_when_0 = del_shap_2016[del_vals[:, year_2016_idx] == min_del]
del_2016_when_1 = del_shap_2016[del_vals[:, year_2016_idx] == max_del]

print("\n2015:")
print(f"  Mean SHAP when Delegation=0: {del_2015_when_0.mean():.4f}  (n={len(del_2015_when_0)})")
print(f"  Mean SHAP when Delegation=1: {del_2015_when_1.mean():.4f}  (n={len(del_2015_when_1)})")

print("\n2016:")
print(f"  Mean SHAP when Delegation=0: {del_2016_when_0.mean():.4f}  (n={len(del_2016_when_0)})")
print(f"  Mean SHAP when Delegation=1: {del_2016_when_1.mean():.4f}  (n={len(del_2016_when_1)})")


# ---------- SHAP base value (expected prediction) ----------
try:
    base_val = explainer.expected_value
except AttributeError:
    # fallback: mean prediction on the SHAP background
    base_val = predict_fn(X_background_flat).mean()

print("\nSHAP base value (expected prediction):", float(np.asarray(base_val)))













# ============================================================
# VIOLIN PLOTS
# ============================================================
# Unified split-violin of SHAP for Project_Last_Year across years
# Horizontal layout (wide, short) and corrected y-axis label.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

sns.set_style("whitegrid")

feat = "Delegation"
feat_idx = include_cols.index(feat)

# Slice SHAP + feature values
shap_feat = shap_values_3d[:, :, feat_idx]   # (N, T)
vals_raw  = X[:, :, feat_idx]                # (N, T)

# Rebuild binary 0/1 per year (robust to scaling/standardization)
vals_bin = np.zeros_like(vals_raw, dtype=int)
for t in range(vals_raw.shape[1]):
    col = vals_raw[:, t]
    u = np.unique(np.round(col, 6))
    if u.size == 2:
        vals_bin[:, t] = (col == col.max()).astype(int)
    else:
        thr = 0.5 * (col.min() + col.max())
        vals_bin[:, t] = (col > thr).astype(int)

# Tidy DF
records = []
for t, y in enumerate(years):
    records.append(pd.DataFrame({
        "Year": y,
        "SHAP": shap_feat[:, t],
        feat:  vals_bin[:, t].astype(int)
    }))
df = pd.concat(records, ignore_index=True)
df["Year"] = pd.Categorical(df["Year"], categories=years, ordered=True)

# --- Colors (reuse for violins, legend, and colorful counts)
PALETTE = {0: "#4C78A8", 1: "#F58518"}

# --- Figure with extra margins for legend (top) and counts (bottom)
fig, ax = plt.subplots(figsize=(14, 3.6))
# Slightly larger bottom margin to fit two colored lines of counts
plt.subplots_adjust(top=0.78, bottom=0.34)

# Split violin
vp = sns.violinplot(
    data=df, x="Year", y="SHAP",
    hue=feat, hue_order=[0, 1],
    split=True, inner="quartile", cut=0, linewidth=0.8,
    palette=PALETTE
)

# Axes
ax.set_ylim(-0.05, 0.25)
ax.set_yticks(np.arange(-0.05, 0.26, 0.05))  # <--- new line

ax.axhline(0, color="k", lw=1, ls="--", alpha=0.5)
ax.set_ylabel("SHAP value (contribution to probability)")
ax.set_xlabel("Year")

# Colorful counts BELOW the x-axis (two lines, each in its class color)
counts = df.groupby(["Year", feat]).size().unstack(fill_value=0)
for i, yr in enumerate(df["Year"].cat.categories):
    n0 = int(counts.loc[yr, 0]) if 0 in counts.columns else 0
    n1 = int(counts.loc[yr, 1]) if 1 in counts.columns else 0

    # First line (0=...) a bit higher, second line (1=...) below it
    ax.text(i, -0.18, f"0={n0}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=PALETTE[0], clip_on=False)

    ax.text(i, -0.28, f"1={n1}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=PALETTE[1], clip_on=False)

# Clean horizontal legend UNDER the title (no overlap)
handles = [Patch(facecolor=PALETTE[0]), Patch(facecolor=PALETTE[1])]
labels  = [f"{feat}: 0", f"{feat}: 1"]
leg = fig.legend(handles, labels,
                 loc="upper center", bbox_to_anchor=(0.5, 0.93),
                 ncol=2, frameon=True, title=None)

# Suptitle at the very top
fig.suptitle(f"{feat}: SHAP distributions across years (0 vs 1)",
             y=0.99, fontsize=14, fontweight="bold")

plt.savefig("project_last_year_shap_split_violin_colorful_counts.png",
            dpi=300, bbox_inches="tight")
plt.show()





# ============================================================
# quartiles
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

# df: columns ["Year", "SHAP", "Project_Last_Year"]
# keep your year ordering if needed:
# df["Year"] = pd.Categorical(df["Year"], categories=years, ordered=True)

q_levels = [0.00, 0.25, 0.50, 0.75, 1.00]

quant_0 = (df[df["Delegation"] == 0]
           .groupby("Year")["SHAP"].quantile(q_levels).unstack()
           .rename(columns={0.00:"0", 0.25:"0.25", 0.50:"0.5", 0.75:"0.75", 1.00:"1"})
           .sort_index())

quant_1 = (df[df["Delegation"] == 1]
           .groupby("Year")["SHAP"].quantile(q_levels).unstack()
           .rename(columns={0.00:"0", 0.25:"0.25", 0.50:"0.5", 0.75:"0.75", 1.00:"1"})
           .sort_index())

# Asymmetric, zero-centered normalization: use full blue range for negatives and full red for positives
vmin = min(quant_0.min().min(), quant_1.min().min())  # most negative value (~ -0.04)
vmax = max(quant_0.max().max(), quant_1.max().max())  # most positive value (~ +0.23)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

cmap = "RdBu_r"
sns.set_style("white")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
plt.subplots_adjust(right=0.90, wspace=0.12)

h0 = sns.heatmap(
    quant_0, ax=axes[0],
    cmap=cmap, norm=norm,
    annot=True, fmt=".3f", linewidths=0, cbar=False,
    annot_kws={"fontsize":9}
)
axes[0].set_title("Quantiles (Delegation = 0)")
axes[0].set_xlabel("Quantile")
axes[0].set_ylabel("Year")

h1 = sns.heatmap(
    quant_1, ax=axes[1],
    cmap=cmap, norm=norm,
    annot=True, fmt=".3f", linewidths=0, cbar=False,
    annot_kws={"fontsize":9}
)
axes[1].set_title("Quantiles (Delegation = 1)")
axes[1].set_xlabel("Quantile")
axes[1].set_ylabel("")

# One shared colorbar in its own axis (won’t cover the right heatmap)
cbar_ax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
cb = fig.colorbar(h1.collections[0], cax=cbar_ax)
cb.set_label("SHAP value (zero-centered)")

for ax in axes:
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"], rotation=0)

fig.suptitle("Per-year SHAP Quantiles (split by Delegation)", y=0.98, fontsize=16)
plt.savefig("shap_quantiles_split_heatmaps_twoslopnorm.png", dpi=300, bbox_inches="tight")
plt.show()




































# ============================================================
# FEATURE IMPORTANCE (ONLY FOR y = 1 SAMPLES)
# ============================================================

# Mask SHAP values and inputs only for positive cases
mask_pos = (y == 1)
shap_values_pos = shap_values_3d[mask_pos]  # Shape: (N_pos, T, F)

# Load shapes
N_pos, T, F = shap_values_pos.shape

# Calculate importance for each feature at each year (restricted to y=1)
feature_time_importance_pos = np.abs(shap_values_pos).mean(axis=0)  # Shape: (T, F)

# Create feature names
if len(include_cols) != F:
    feature_names = [f"Feature_{i}" for i in range(F)]
else:
    feature_names = include_cols

years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]

# Create a heatmap (only y=1)
plt.figure(figsize=(12, 8))
sns.heatmap(feature_time_importance_pos.T,
            xticklabels=years,
            yticklabels=feature_names,
            cmap='RdBu_r',
            cbar_kws={'label': 'Mean |SHAP value| (y=1 only)'},
            fmt='.3f')

plt.xlabel('Year', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance Across Years (SHAP Analysis, y=1 only)', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance_temporal_y1.png', dpi=300, bbox_inches='tight')
plt.show()

# Also show which year is most important overall (restricted to y=1)
year_importance_pos = feature_time_importance_pos.mean(axis=1)
most_important_year_pos = years[np.argmax(year_importance_pos)]

# And which feature is most important overall (restricted to y=1)
feature_importance_pos = feature_time_importance_pos.mean(axis=0)
most_important_feature_pos = feature_names[np.argmax(feature_importance_pos)]

print(f"Most important year overall (y=1): {most_important_year_pos}")
print(f"Most important feature overall (y=1): {most_important_feature_pos}")
print(f"\nYear importance ranking (y=1):")
for year, imp in sorted(zip(years, year_importance_pos), key=lambda x: x[1], reverse=True):
    print(f"  {year}: {imp:.4f}")






# ============================================================
# FEATURE IMPORTANCE (ONLY FOR y = 0 SAMPLES)
# ============================================================

# Mask SHAP values and inputs only for positive cases
mask_pos = (y == 0)
shap_values_pos = shap_values_3d[mask_pos]  # Shape: (N_pos, T, F)

# Load shapes
N_pos, T, F = shap_values_pos.shape

# Calculate importance for each feature at each year (restricted to y=1)
feature_time_importance_pos = np.abs(shap_values_pos).mean(axis=0)  # Shape: (T, F)

# Create feature names
if len(include_cols) != F:
    feature_names = [f"Feature_{i}" for i in range(F)]
else:
    feature_names = include_cols

years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]

# Create a heatmap (only y=1)
plt.figure(figsize=(12, 8))
sns.heatmap(feature_time_importance_pos.T,
            xticklabels=years,
            yticklabels=feature_names,
            cmap='RdBu_r',
            cbar_kws={'label': 'Mean |SHAP value| (y=1 only)'},
            fmt='.3f')

plt.xlabel('Year', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance Across Years (SHAP Analysis, y=1 only)', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance_temporal_y1.png', dpi=300, bbox_inches='tight')
plt.show()

# Also show which year is most important overall (restricted to y=1)
year_importance_pos = feature_time_importance_pos.mean(axis=1)
most_important_year_pos = years[np.argmax(year_importance_pos)]

# And which feature is most important overall (restricted to y=1)
feature_importance_pos = feature_time_importance_pos.mean(axis=0)
most_important_feature_pos = feature_names[np.argmax(feature_importance_pos)]

print(f"Most important year overall (y=1): {most_important_year_pos}")
print(f"Most important feature overall (y=1): {most_important_feature_pos}")
print(f"\nYear importance ranking (y=1):")
for year, imp in sorted(zip(years, year_importance_pos), key=lambda x: x[1], reverse=True):
    print(f"  {year}: {imp:.4f}")

























# FLATTEN inputs for SHAP
T, F = background_np.shape[1], background_np.shape[2]
background_np_flat = background_np.reshape(background_np.shape[0], T*F)
to_explain_np_flat = to_explain_np.reshape(to_explain_np.shape[0], T*F)

device = torch.device("cpu")
model = model.to(device)


explainer = shap.KernelExplainer(predict_fn, background_np_flat)

shap_values = explainer.shap_values(to_explain_np_flat, nsamples=100)  # shape: [num_samples, T*F]


shap_values_matrix = shap_values[0] if isinstance(shap_values, list) else shap_values  # for binary classification
shap_values_matrix = shap_values_matrix.reshape(to_explain_np.shape[0], T, F)

########################
####importance generic 

importance_matrix = np.mean(np.abs(shap_values_matrix), axis=0)  # shape: (T, F)

feature_names = [
    'UnitedNations',
    'GDP',
    'Public_Grant',
    'Budget_Previous_Year',
    'Donor_Aid_Budget',
    'LatinAmerica',
    'Africa',
    'Ex_colony',
    'Delegation',
    'ControlofCorruption',
    'PoliticalStability_NoViolence',
    'cumulative_path_dependence',
    'NGO_YearFoundation'
]

# Year labels for rows (8 years: 2009 to 2016)
year_labels = [str(y) for y in range(2009, 2017)]  # ['2009', ..., '2016']

# Compute the mean absolute SHAP value matrix over all samples
# shap_values_matrix: [num_samples, T, F]
importance_matrix = np.mean(np.abs(shap_values_matrix), axis=0)  # shape: (T, F)

import matplotlib.pyplot as plt

# Feature and year labels as before
feature_names = [
    'UnitedNations',
    'GDP',
    'Public_Grant',
    'Budget_Previous_Year',
    'Donor_Aid_Budget',
    'LatinAmerica',
    'Africa',
    'Ex_colony',
    'Delegation',
    'ControlofCorruption',
    'PoliticalStability_NoViolence',
    'NGO_YearFoundation'
]
year_labels = [str(y) for y in range(2009, 2017)]

# Mean abs SHAP matrix: importance_matrix [T, F]
importance_matrix = np.mean(np.abs(shap_values_matrix), axis=0)  # [T, F], T=8, F=13

plt.figure(figsize=(16, 8))
im = plt.imshow(importance_matrix.T, aspect='auto', cmap='bwr')

plt.xticks(ticks=np.arange(len(year_labels)), labels=year_labels)
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
plt.xlabel('Year')
plt.ylabel('Feature')
plt.title('Mean Absolute SHAP Value (Feature × Year)')
plt.colorbar(im, label='Mean |SHAP value|')

# Annotate each cell with 3 decimals
for i in range(len(feature_names)):
    for j in range(len(year_labels)):
        plt.text(j, i, f"{importance_matrix[j, i]:.3f}",
                 ha="center", va="center", color="black", fontsize=10)

plt.tight_layout()
plt.show()








######################################

feature_idx = include_cols.index('cumulative_path_dependence')  # which column
year_idx = year_labels.index('2016')   

x_vals = to_explain_np[:, year_idx, feature_idx]   # shape: (num_samples,)

# --- Get scaler min/max for that feature ---
min_ = scaler.data_min_[feature_idx]
max_ = scaler.data_max_[feature_idx]

# --- Denormalize the feature values ---
x_vals_denorm = x_vals * (max_ - min_) + min_

# --- Get model outputs (sigmoid probabilities) ---
with torch.no_grad():
    probs = torch.sigmoid(model(torch.from_numpy(to_explain_np).float().to("cpu"))).cpu().numpy().flatten()

# --- Get true labels for these samples ---
labels = y_tensor[:len(to_explain_np)].cpu().numpy()  # adjust if you explained fewer than all samples

# --- Plot: feature value vs. model output, colored by label ---
plt.figure(figsize=(8,5))
plt.scatter(x_vals_denorm[labels==0], probs[labels==0], c='blue', alpha=0.5, label='Label=0')
plt.scatter(x_vals_denorm[labels==1], probs[labels==1], c='red',  alpha=0.5, label='Label=1')
plt.xlabel('cumulative_path_dependence (2016) [Original Scale]')
plt.ylabel('Model Output (Sigmoid Probability)')
plt.title('Relationship: cumulative_path_dependence (2016, Real Value) vs Model Output')
plt.legend()
plt.tight_layout()
plt.show()


#################################################













# ============================================================
# IMPORTANCE PROJECT LAST YEAR 1 AND 2 YEARS AGO
# ============================================================


# Find Project_Last_Year index
project_last_year_idx = include_cols.index('Project_Last_Year')

# Get SHAP values for Project_Last_Year across all samples and years
project_shap = shap_values_3d[:, :, project_last_year_idx]  # Shape: (N, T)

# Get actual Project_Last_Year values from original data
# Reshape X to get the actual feature values
project_values = X[:, :, project_last_year_idx]  # Shape: (N, T)

# Focus on 2015 and 2016 (most relevant years from heatmap)
year_2015_idx = 6  # 2015 is index 6 (0-indexed from 2009)
year_2016_idx = 7  # 2016 is index 7

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: SHAP vs Feature Value for 2015
ax = axes[0, 0]
scatter = ax.scatter(project_values[:, year_2015_idx], 
                    project_shap[:, year_2015_idx],
                    alpha=0.5, c=y, cmap='RdYlGn')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Project_Last_Year Value (2015)')
ax.set_ylabel('SHAP Value')
ax.set_title('2015: How Project_Last_Year Affects Prediction')
plt.colorbar(scatter, ax=ax, label='Actual Visit 2016')

# Plot 2: SHAP vs Feature Value for 2016
ax = axes[0, 1]
scatter = ax.scatter(project_values[:, year_2016_idx], 
                    project_shap[:, year_2016_idx],
                    alpha=0.5, c=y, cmap='RdYlGn')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Project_Last_Year Value (2016)')
ax.set_ylabel('SHAP Value')
ax.set_title('2016: How Project_Last_Year Affects Prediction')
plt.colorbar(scatter, ax=ax, label='Actual Visit 2016')

# Plot 3: Distribution comparison
ax = axes[1, 0]
shap_when_0 = project_shap[project_values == 0]
shap_when_1 = project_shap[project_values == 1]

positions = [1, 2]
bp = ax.boxplot([shap_when_0.flatten(), shap_when_1.flatten()], 
                 positions=positions, 
                 labels=['No Project Last Year (0)', 'Project Last Year (1)'],
                 patch_artist=True)
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightgreen')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.set_ylabel('SHAP Value Distribution')
ax.set_title('Impact of Project_Last_Year on Predictions')
ax.grid(True, alpha=0.3)

# Plot 4: Statistical summary
ax = axes[1, 1]
ax.axis('off')

# Calculate statistics
mean_shap_no_project = shap_when_0.mean()
mean_shap_with_project = shap_when_1.mean()
impact_difference = mean_shap_with_project - mean_shap_no_project

# Get predictions
predictions = predict_fn(X.reshape(N, -1))

# Calculate probabilities
prob_visit_no_project = predictions[project_values[:, -1] == 0].mean()
prob_visit_with_project = predictions[project_values[:, -1] == 1].mean()

summary_text = f"""
IMPACT ANALYSIS: Project_Last_Year

When Project_Last_Year = 0 (No previous project):
  • Mean SHAP value: {mean_shap_no_project:.4f}
  • Avg prediction probability: {prob_visit_no_project:.2%}

When Project_Last_Year = 1 (Had previous project):
  • Mean SHAP value: {mean_shap_with_project:.4f}
  • Avg prediction probability: {prob_visit_with_project:.2%}

IMPACT DIFFERENCE: {impact_difference:.4f}
{'✓ POSITIVE: Having a project last year INCREASES visit probability' if impact_difference > 0 else '✗ NEGATIVE: Having a project last year DECREASES visit probability'}

Probability increase: {(prob_visit_with_project - prob_visit_no_project):.1%}
"""

ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
        fontsize=11, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Project_Last_Year: Directional Impact Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('project_last_year_impact.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional analysis: Look at actual outcomes
visited_with_project = np.mean(y[project_values[:, -1] == 1])
visited_no_project = np.mean(y[project_values[:, -1] == 0])

print("="*60)
print("PROJECT_LAST_YEAR IMPACT SUMMARY")
print("="*60)
print(f"\nActual visit rates:")
print(f"  With previous project: {visited_with_project:.1%}")
print(f"  Without previous project: {visited_no_project:.1%}")
print(f"  Difference: {(visited_with_project - visited_no_project):.1%}")
print(f"\nModel's learned impact (SHAP):")
print(f"  Average SHAP when = 1: {mean_shap_with_project:.4f}")
print(f"  Average SHAP when = 0: {mean_shap_no_project:.4f}")
print(f"  Impact: {'POSITIVE' if impact_difference > 0 else 'NEGATIVE'} ({impact_difference:.4f})")





# ============================================================
# 65.7%?
# ============================================================



# Load your data
project_last_year_idx = include_cols.index('Project_Last_Year')

# Get Project_Last_Year values for 2016 (last year in sequence)
project_values_2016 = X[:, -1, project_last_year_idx]  # 2016 is last index

# Get model predictions
X_flat = X.reshape(len(X), -1)
predictions_proba = predict_fn(X_flat)

# Calculate average probabilities
prob_when_no_project = predictions_proba[project_values_2016 == 0].mean()
prob_when_has_project = predictions_proba[project_values_2016 == 1].mean()

# Different ways to express the increase
absolute_increase = prob_when_has_project - prob_when_no_project
relative_increase = (prob_when_has_project - prob_when_no_project) / prob_when_no_project * 100
odds_ratio = (prob_when_has_project / (1 - prob_when_has_project)) / (prob_when_no_project / (1 - prob_when_no_project))

print("="*60)
print("PROBABILITY INCREASE CALCULATION")
print("="*60)
print(f"\nAverage predicted probability:")
print(f"  When Project_Last_Year = 0: {prob_when_no_project:.3f} ({prob_when_no_project:.1%})")
print(f"  When Project_Last_Year = 1: {prob_when_has_project:.3f} ({prob_when_has_project:.1%})")
print(f"\n3 ways to express the increase:")
print(f"1. Absolute increase: {absolute_increase:.3f} ({absolute_increase:.1%})")
print(f"   Interpretation: +{absolute_increase:.1%} percentage points")
print(f"\n2. Relative increase: {relative_increase:.1f}%")
print(f"   Interpretation: {relative_increase:.1f}% higher than baseline")
print(f"\n3. Odds ratio: {odds_ratio:.2f}")
print(f"   Interpretation: {odds_ratio:.1f}x more likely to visit")

# Show the calculation that leads to 65.7%
if abs(relative_increase - 65.7) < 5:  # If close to 65.7%
    print(f"\nThe 65.7% figure is the RELATIVE increase:")
    print(f"  ({prob_when_has_project:.3f} - {prob_when_no_project:.3f}) / {prob_when_no_project:.3f} × 100 = {relative_increase:.1f}%")
elif abs(absolute_increase * 100 - 65.7) < 5:
    print(f"\nThe 65.7% figure is the ABSOLUTE increase:")
    print(f"  {prob_when_has_project:.3f} - {prob_when_no_project:.3f} = {absolute_increase:.3f} = {absolute_increase:.1%}")

# Create a visual explanation
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

categories = ['No Project\nLast Year', 'Project\nLast Year']
probabilities = [prob_when_no_project, prob_when_has_project]

bars = ax.bar(categories, probabilities, color=['lightcoral', 'lightgreen'], edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, prob in zip(bars, probabilities):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{prob:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add increase annotation
ax.annotate('', xy=(1, prob_when_has_project), xytext=(0, prob_when_no_project),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(0.5, (prob_when_no_project + prob_when_has_project)/2, 
        f'+{absolute_increase:.1%}\n({relative_increase:.1f}% increase)',
        ha='center', va='center', fontsize=11, color='red', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax.set_ylabel('Predicted Probability of Visit', fontsize=12)
ax.set_title('Impact of Project_Last_Year on Visit Probability', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(probabilities) * 1.2)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Also check actual data for comparison
actual_rate_no_project = y[project_values_2016 == 0].mean()
actual_rate_has_project = y[project_values_2016 == 1].mean()
actual_relative_increase = (actual_rate_has_project - actual_rate_no_project) / actual_rate_no_project * 100

print(f"\nFor comparison - ACTUAL data:")
print(f"  Visit rate without project: {actual_rate_no_project:.1%}")
print(f"  Visit rate with project: {actual_rate_has_project:.1%}")
print(f"  Relative increase: {actual_relative_increase:.1f}%")






################################################


import seaborn as sns

# --- Assumptions ---
# feature_names (or include_cols): list of your 13 features, in order of your input data
# year_labels: ['2009', ..., '2016']
# scaler: your fitted MinMaxScaler
# model: your trained PyTorch model (on CPU)
# to_explain_np: [num_samples, T, F] input matrix for the explained samples
# y_tensor: true labels (as a tensor)
# shap_values_matrix: [num_samples, T, F] SHAP value matrix from KernelExplainer

# --- Set indices for feature and year ---
feature_idx = include_cols.index('cumulative_path_dependence')
year_idx = year_labels.index('2016')

# --- Denormalize the feature values ---
x_vals = to_explain_np[:, year_idx, feature_idx]
min_ = scaler.data_min_[feature_idx]
max_ = scaler.data_max_[feature_idx]
x_vals_denorm = x_vals * (max_ - min_) + min_

# --- Model outputs (sigmoid probabilities) ---
with torch.no_grad():
    probs = torch.sigmoid(model(torch.from_numpy(to_explain_np).float().to("cpu"))).cpu().numpy().flatten()

# --- True labels ---
labels = y_tensor[:len(to_explain_np)].cpu().numpy()

# --- SHAP values for this variable ---
shap_vals = shap_values_matrix[:, year_idx, feature_idx]

# --- DataFrame for plotting ---
df = pd.DataFrame({
    'cumulative_path_dependence_2016': x_vals_denorm,
    'model_output': probs,
    'label': labels.astype(int),
    'shap_value': shap_vals
})

# --- Plot 1: Scatter of feature value vs model output, colored by label ---
plt.figure(figsize=(8,5))
plt.scatter(df.loc[df.label==0, 'cumulative_path_dependence_2016'], df.loc[df.label==0, 'model_output'],
            c='blue', alpha=0.5, label='Label=0')
plt.scatter(df.loc[df.label==1, 'cumulative_path_dependence_2016'], df.loc[df.label==1, 'model_output'],
            c='red',  alpha=0.5, label='Label=1')
plt.xlabel('cumulative_path_dependence (2016) [Original Scale]')
plt.ylabel('Model Output (Sigmoid Probability)')
plt.title('Relationship: cumulative_path_dependence (2016, Real Value) vs Model Output')
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Violin plot of SHAP values by label ---
plt.figure(figsize=(7,5))
sns.violinplot(x='label', y='shap_value', data=df, palette='Set2', inner='point')
plt.title("SHAP value for cumulative_path_dependence (2016) by Label")
plt.xlabel("Output label")
plt.ylabel("SHAP value: cumulative_path_dependence (2016)")
plt.tight_layout()
plt.show()


###############

import matplotlib.pyplot as plt

# Already have:
# x_vals_denorm : denormalized feature values (cumulative_path_dependence, 2016)
# shap_vals     : SHAP value for that feature/year
# labels        : output labels (0 or 1)

plt.figure(figsize=(8,6))
plt.scatter(
    x_vals_denorm[labels==0], shap_vals[labels==0],
    c='blue', alpha=0.6, label='Label=0'
)
plt.scatter(
    x_vals_denorm[labels==1], shap_vals[labels==1],
    c='red',  alpha=0.6, label='Label=1'
)
plt.xlabel('cumulative_path_dependence (2016) [Original Scale]')
plt.ylabel('SHAP value: cumulative_path_dependence (2016)')
plt.title('SHAP Value vs. Feature Value\ncumulative_path_dependence (2016)')
plt.legend()
plt.tight_layout()
plt.show()


#############################

# 1. Get model predictions (0/1), e.g. threshold at 0.5
with torch.no_grad():
    preds = (torch.sigmoid(model(torch.from_numpy(to_explain_np).float().to("cpu"))).cpu().numpy().flatten() > 0.5).astype(int)


# 2. Select only samples where predicted label == 1
idx_1 = np.where(preds == 0)[0]
shap_label1 = shap_values_matrix[idx_1]  # shape: (N1, T, F)
len(shap_label1)

# 3. Compute mean SHAP value (not absolute) for these samples
mean_shap = np.mean(shap_label1, axis=0)  # shape: (T, F)

# 4. Plot heatmap (years as columns, features as rows)
plt.figure(figsize=(16, 8))
im = plt.imshow(mean_shap.T, aspect='auto', cmap='bwr', vmin=-np.max(np.abs(mean_shap)), vmax=np.max(np.abs(mean_shap)))

plt.xticks(ticks=np.arange(len(year_labels)), labels=year_labels)
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
plt.xlabel('Year')
plt.ylabel('Feature')
plt.title('Mean SHAP Value (Feature × Year) for Predicted Label=1')
plt.colorbar(im, label='Mean SHAP value')

# Annotate with values
for i in range(len(feature_names)):
    for j in range(len(year_labels)):
        plt.text(j, i, f"{mean_shap[j, i]:.3f}", ha="center", va="center", color="black", fontsize=9)

plt.tight_layout()
plt.show()



##############################

# --- 1. Indices for feature/year ---
feature_name = 'cumulative_path_dependence'
year_of_interest = '2016'

feature_idx = feature_names.index(feature_name)
year_idx = year_labels.index(year_of_interest)

# --- 2. Denormalize cumulative_path_dependence (2016) ---
x_vals = to_explain_np[:, year_idx, feature_idx]
min_ = scaler.data_min_[feature_idx]
max_ = scaler.data_max_[feature_idx]
x_vals_denorm = x_vals * (max_ - min_) + min_

# --- 3. Get predicted labels (0/1), threshold at 0.5 ---
with torch.no_grad():
    probs = torch.sigmoid(model(torch.from_numpy(to_explain_np).float().to("cpu"))).cpu().numpy().flatten()
preds = (probs > 0.5).astype(int)

# --- 4. Filter: cumulative_path_dependence > 4 AND predicted label == 1 ---
idx_both = np.where((x_vals_denorm > 4) & (preds == 1))[0]

# --- 5. SHAP values for these samples ---
shap_gt4_label1 = shap_values_matrix[idx_both]  # shape: (N, T, F)

# --- 6. Compute mean SHAP value for each feature/year ---
mean_shap = np.mean(shap_gt4_label1, axis=0)  # shape: (T, F)

# --- 7. Plot heatmap (features x years) ---
plt.figure(figsize=(16, 8))
im = plt.imshow(
    mean_shap.T, aspect='auto', cmap='bwr',
    vmin=-np.max(np.abs(mean_shap)), vmax=np.max(np.abs(mean_shap))
)

plt.xticks(ticks=np.arange(len(year_labels)), labels=year_labels)
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
plt.xlabel('Year')
plt.ylabel('Feature')
plt.title('Mean SHAP Value (Feature × Year)\nfor cumulative_path_dependence (2016) > 4 and Predicted Label=1')
plt.colorbar(im, label='Mean SHAP value')

# Annotate with values
for i in range(len(feature_names)):
    for j in range(len(year_labels)):
        plt.text(j, i, f"{mean_shap[j, i]:.3f}", ha="center", va="center", color="black", fontsize=9)

plt.tight_layout()
plt.show()






