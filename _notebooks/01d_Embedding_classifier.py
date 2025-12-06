# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Embedding-Based XGBoost Classifier
#
# This notebook implements a content recommendation classifier using:
# - One-hot encoding for project_id
# - XGBoost for classification

# %%
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer

# %%
# #!poetry add sentence-transformers

# %%
# Configuration
data_folder = Path("../_data/")

# %% [markdown]
# ## 1. Data Loading

# %%
# Load the preprocessed data
df = pd.read_parquet(data_folder / "data.parquet")

print(f"Total samples: {len(df)}")
print(f"\nClass distribution:")
print(df['label'].value_counts())
print(f"\nProject distribution:")
print(df['project_id'].value_counts())

# %%
# Display sample data
print("\nSample data:")
print(df.head())
print(f"\nColumns: {df.columns.tolist()}")

# %%
# Convert label to binary (1 for positive, 0 for negative)
df['label_binary'] = (df['label'] == 'positive').astype(int)

print("\nLabel distribution:")
print(df['label_binary'].value_counts())

# %% [markdown]
# ## 2. Train/Test Split
#
# Use stratified split to maintain class balance across projects

# %%
# Create stratification key combining project_id and label
strat_key = df['project_id'].astype(str) + "_" + df['label'].astype(str)

# Stratified split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, strat_key))

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"\nTrain class distribution:")
print(train_df['label_binary'].value_counts())
print(f"\nTest class distribution:")
print(test_df['label_binary'].value_counts())

# %% [markdown]
# ## 3. Embedding Generation
#
# Generate embeddings for text fields: author, title, and summary

# %%
emb_model = SentenceTransformer('all-MiniLM-L6-v2')
emb_model.save('./_emb/')

# %% [markdown]
# ### Generate Embeddings for Training Data

# %%
# Create combined input text (similar to LLM approach)
train_df['combined_text'] = (
    "Author: " + train_df['author'].fillna("Unknown") +
    "\nTitle: " + train_df['title'].fillna("") +
    "\nSummary: " + train_df['summary'].fillna("")
)

test_df['combined_text'] = (
    "Author: " + test_df['author'].fillna("Unknown") +
    "\nTitle: " + test_df['title'].fillna("") +
    "\nSummary: " + test_df['summary'].fillna("")
)

# %%

# %%
# Generate embeddings for training data
print("Generating embeddings for training data...")
train_embeddings = emb_model.encode(train_df['combined_text'].tolist())

# %%
# Generate embeddings for test data
print("Generating embeddings for test data...")
test_embeddings = emb_model.encode(test_df['combined_text'].tolist())

# %%
# Convert to numpy arrays
train_embeddings_array = np.array(train_embeddings)
test_embeddings_array = np.array(test_embeddings)

print(f"Training embeddings shape: {train_embeddings_array.shape}")
print(f"Test embeddings shape: {test_embeddings_array.shape}")

# %%
# Save embeddings for future use
np.save(data_folder / "train_embeddings.npy", train_embeddings_array)
np.save(data_folder / "test_embeddings.npy", test_embeddings_array)
print("Embeddings saved!")

# %% [markdown]
# ## 5. One-Hot Encoding for Project ID

# %%
# One-hot encode project_id
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

train_project_encoded = encoder.fit_transform(train_df[['project_id']])
test_project_encoded = encoder.transform(test_df[['project_id']])

print(f"Project one-hot encoding shape: {train_project_encoded.shape}")
print(f"Unique projects: {encoder.categories_[0]}")

# %%
# Save the encoder for future use
with open(data_folder / "project_encoder.pkl", 'wb') as f:
    pickle.dump(encoder, f)
print("Encoder saved!")

# %% [markdown]
# ## 6. Combine Features

# %%
# Combine embeddings with one-hot encoded project_id
X_train = np.hstack([train_embeddings_array, train_project_encoded])
X_test = np.hstack([test_embeddings_array, test_project_encoded])

y_train = train_df['label_binary'].values
y_test = test_df['label_binary'].values

print(f"Final training feature shape: {X_train.shape}")
print(f"Final test feature shape: {X_test.shape}")
print(f"Feature breakdown: {train_embeddings_array.shape[1]} (embeddings) + {train_project_encoded.shape[1]} (project)")

# %% [markdown]
# ## 7. XGBoost Training

# %%
# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# %%
# XGBoost parameters optimized for recall
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 600,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'tree_method': 'hist',
    'early_stopping_rounds': 20
}

# %%
# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# %%
# Train XGBoost model
print("Training XGBoost model...")
evals = [(dtrain, 'train'), (dtest, 'test')]

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=params['n_estimators'],
    evals=evals,
    early_stopping_rounds=params['early_stopping_rounds'],
    verbose_eval=10
)

print("\nTraining complete!")

# %%
# Save the model
model.save_model(data_folder / "xgboost_model.json")
print("Model saved!")

# %% [markdown]
# ## 8. Model Evaluation

# %%
# Get predictions
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba >= 0.5).astype(int)

# %%
# Classification report
print("Classification Report:")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# %%
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - XGBoost Classifier')
plt.show()

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
recall = tp / (tp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nDetailed Metrics:")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

# %% [markdown]
# ## 9. ROC and Precision-Recall Curves

# %%
# ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve - XGBoost Classifier')
plt.legend(loc='lower right')
plt.grid(True)

# Precision-Recall Curve
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

plt.subplot(1, 2, 2)
plt.plot(recall_curve, precision_curve, lw=2, label=f'PR Curve (AP = {avg_precision:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Per-Project Performance Analysis

# %%
# Analyze performance by project
test_df['prediction_proba'] = y_pred_proba
test_df['prediction'] = y_pred

project_metrics = []

for project_id in test_df['project_id'].unique():
    project_mask = test_df['project_id'] == project_id
    project_data = test_df[project_mask]

    y_true_proj = project_data['label_binary'].values
    y_pred_proj = project_data['prediction'].values
    y_proba_proj = project_data['prediction_proba'].values

    if len(np.unique(y_true_proj)) < 2:
        # Skip if only one class present
        continue

    cm_proj = confusion_matrix(y_true_proj, y_pred_proj)
    tn, fp, fn, tp = cm_proj.ravel()

    recall_proj = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_proj = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_proj = 2 * (precision_proj * recall_proj) / (precision_proj + recall_proj) if (precision_proj + recall_proj) > 0 else 0
    roc_auc_proj = roc_auc_score(y_true_proj, y_proba_proj)

    project_metrics.append({
        'project_id': project_id,
        'samples': len(project_data),
        'recall': recall_proj,
        'precision': precision_proj,
        'f1': f1_proj,
        'roc_auc': roc_auc_proj
    })

# %%
# Display per-project metrics
metrics_df = pd.DataFrame(project_metrics)
print("\nPer-Project Performance:")
print("=" * 80)
print(metrics_df.to_string(index=False))

# %%
# Visualize per-project performance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_df.plot(x='project_id', y='recall', kind='bar', ax=axes[0, 0], legend=False, color='skyblue')
axes[0, 0].set_title('Recall by Project')
axes[0, 0].set_ylabel('Recall')
axes[0, 0].set_xlabel('Project ID')
axes[0, 0].axhline(y=recall, color='r', linestyle='--', label=f'Overall: {recall:.3f}')
axes[0, 0].legend()

metrics_df.plot(x='project_id', y='precision', kind='bar', ax=axes[0, 1], legend=False, color='lightcoral')
axes[0, 1].set_title('Precision by Project')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_xlabel('Project ID')
axes[0, 1].axhline(y=precision, color='r', linestyle='--', label=f'Overall: {precision:.3f}')
axes[0, 1].legend()

metrics_df.plot(x='project_id', y='f1', kind='bar', ax=axes[1, 0], legend=False, color='lightgreen')
axes[1, 0].set_title('F1 Score by Project')
axes[1, 0].set_ylabel('F1 Score')
axes[1, 0].set_xlabel('Project ID')
axes[1, 0].axhline(y=f1, color='r', linestyle='--', label=f'Overall: {f1:.3f}')
axes[1, 0].legend()

metrics_df.plot(x='project_id', y='roc_auc', kind='bar', ax=axes[1, 1], legend=False, color='plum')
axes[1, 1].set_title('ROC AUC by Project')
axes[1, 1].set_ylabel('ROC AUC')
axes[1, 1].set_xlabel('Project ID')
axes[1, 1].axhline(y=roc_auc, color='r', linestyle='--', label=f'Overall: {roc_auc:.3f}')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Threshold Optimization for Recall
#
# Find optimal threshold to maximize recall while maintaining acceptable precision

# %%
# Calculate metrics at different thresholds
thresholds_to_test = np.arange(0.1, 0.9, 0.05)
threshold_metrics = []

for threshold in thresholds_to_test:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    cm_thresh = confusion_matrix(y_test, y_pred_thresh)
    tn, fp, fn, tp = cm_thresh.ravel()

    recall_thresh = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_thresh = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_thresh = 2 * (precision_thresh * recall_thresh) / (precision_thresh + recall_thresh) if (precision_thresh + recall_thresh) > 0 else 0

    threshold_metrics.append({
        'threshold': threshold,
        'recall': recall_thresh,
        'precision': precision_thresh,
        'f1': f1_thresh,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    })

threshold_df = pd.DataFrame(threshold_metrics)

# %%
# Plot threshold analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Recall, Precision, F1 vs Threshold
axes[0].plot(threshold_df['threshold'], threshold_df['recall'], 'b-', label='Recall', linewidth=2)
axes[0].plot(threshold_df['threshold'], threshold_df['precision'], 'r-', label='Precision', linewidth=2)
axes[0].plot(threshold_df['threshold'], threshold_df['f1'], 'g-', label='F1 Score', linewidth=2)
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Score')
axes[0].set_title('Metrics vs Classification Threshold')
axes[0].legend()
axes[0].grid(True)
axes[0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')

# False Negatives vs Threshold
axes[1].plot(threshold_df['threshold'], threshold_df['fn'], 'r-', linewidth=2, label='False Negatives')
axes[1].plot(threshold_df['threshold'], threshold_df['fp'], 'orange', linewidth=2, label='False Positives')
axes[1].set_xlabel('Threshold')
axes[1].set_ylabel('Count')
axes[1].set_title('False Negatives & False Positives vs Threshold')
axes[1].legend()
axes[1].grid(True)
axes[1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# %%
# Find threshold that maximizes recall with precision >= 0.6
min_precision = 0.6
viable_thresholds = threshold_df[threshold_df['precision'] >= min_precision]

if len(viable_thresholds) > 0:
    optimal_row = viable_thresholds.loc[viable_thresholds['recall'].idxmax()]
    print(f"\nOptimal threshold (precision >= {min_precision}):")
    print(f"Threshold: {optimal_row['threshold']:.2f}")
    print(f"Recall: {optimal_row['recall']:.4f}")
    print(f"Precision: {optimal_row['precision']:.4f}")
    print(f"F1 Score: {optimal_row['f1']:.4f}")
    print(f"False Negatives: {optimal_row['fn']:.0f}")
    print(f"False Positives: {optimal_row['fp']:.0f}")
else:
    print(f"\nNo threshold achieves precision >= {min_precision}")
    # Find threshold that maximizes F1
    optimal_row = threshold_df.loc[threshold_df['f1'].idxmax()]
    print(f"\nThreshold that maximizes F1 Score:")
    print(f"Threshold: {optimal_row['threshold']:.2f}")
    print(f"Recall: {optimal_row['recall']:.4f}")
    print(f"Precision: {optimal_row['precision']:.4f}")
    print(f"F1 Score: {optimal_row['f1']:.4f}")

# %%
