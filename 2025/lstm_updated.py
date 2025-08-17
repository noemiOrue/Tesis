# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 11:45:06 2025

@author: 72458991
"""

import pandas as pd
import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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

data['GDP']                = np.log(data['GDP']+1)
data['Public_Grant']       = np.log(data['Public_Grant']+1)
data['Budget_Previous_Year']= np.log(data['Budget_Previous_Year']+1)
data['Donor_Aid_Budget']   = np.log(data['Donor_Aid_Budget']+1)


avoid_cols = ["Country_Year", 'NGO', 'Country', 'Year', 'Visitado','RuleofLaw','RegulatoryQuality','GovernmentEffectiveness','VoiceandAccountability','generic','cumulative_path_dependence']
include_cols = [c for c in data.columns if c not in avoid_cols]

print("\nColumns included:")
for col in include_cols:
    print(f"- {col}")


scaler = MinMaxScaler(feature_range=(0, 1))
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

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of samples:", len(lookup_list))




# 1) Data → Tensors
X_tensor = torch.from_numpy(X).float()      # ensure float32
y_tensor = torch.from_numpy(y).float()      # float for BCE

dataset = TensorDataset(X_tensor, y_tensor)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)

# 2) Super–simple 1-layer LSTM, no frills
class TwoLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,        # Two LSTM layers
            batch_first=True,
            dropout=dropout      # Dropout *between* layers only
        )
        self.fc = nn.Linear(hidden_size, 1)  # output shape [B, 1]

    def forward(self, x):
        out, _ = self.lstm(x)      # out: [B, T, H]
        last = out[:, -1, :]       # [B, H] -- last time step
        return self.fc(last)       # raw logits [B, 1]

model     = TwoLayerLSTM(input_size=X_tensor.shape[2], hidden_size=32)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 3) Training loop with gradient clipping
for epoch in range(1, 21):
    model.train()
    epoch_loss = 0.0

    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)                # shape: [batch, 1]
        loss   = criterion(logits, yb.unsqueeze(1))   # match shape: [batch, 1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)


    avg = epoch_loss / len(dataset)
    print(f"Epoch {epoch:02d} — Loss: {avg:.4f}")

print("Done.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 1) Check for NaNs or Infs in X or y
print("X range:", X_tensor.min().item(), "to", X_tensor.max().item())
print("Any NaN in X?", torch.isnan(X_tensor).any().item())
print("Any Inf in X?", torch.isinf(X_tensor).any().item())

print("y unique values:", torch.unique(y_tensor))
print("Any NaN in y?", torch.isnan(y_tensor).any().item())
print("Any Inf in y?", torch.isinf(y_tensor).any().item())

# 2) If there are NaNs, locate them:
if torch.isnan(X_tensor).any():
    idx = torch.isnan(X_tensor).any(dim=(1,2)).nonzero(as_tuple=True)[0]
    print("Samples with NaN in X:", idx.tolist())
if torch.isnan(y_tensor).any():
    idx = torch.isnan(y_tensor).nonzero(as_tuple=True)[0]
    print("Samples with NaN in y:", idx.tolist())

import shap
import numpy as np
idxs = np.random.choice(len(X_tensor), size=500, replace=False)
background_np = X_tensor[idxs].cpu().numpy()
to_explain_np = X_tensor.cpu().numpy()   # shape: (1000, T, F)

def predict_fn(x_np):
    with torch.no_grad():
        # Reshape if SHAP passes 2D flattened array
        if x_np.ndim == 2 and x_np.shape[1] == background_np.shape[1] * background_np.shape[2]:
            # x_np is [batch, T*F], unflatten:
            T, F = background_np.shape[1], background_np.shape[2]
            x_np = x_np.reshape((-1, T, F))
        x_tensor = torch.from_numpy(x_np).float().to(device)
        logits   = model(x_tensor)
        probs    = torch.sigmoid(logits)
        return probs.cpu().numpy().reshape(-1, 1)  # shape: [batch, 1]

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






