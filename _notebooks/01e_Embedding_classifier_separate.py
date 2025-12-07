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
# # Embedding-Based XGBoost Classifier - Separate Models Per Project
#
# This notebook implements a content recommendation classifier using:
# - Separate XGBoost models trained for each project_id
# - Text embeddings without project_id as a feature
# - Project-specific model evaluation and optimization

# %%
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
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
# Configuration
data_folder = Path("../_data/")
model_output_folder = Path("../src/aim/model_definitions/")
model_output_folder.mkdir(parents=True, exist_ok=True)

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

# Print per-project distribution
print(f"\nTrain samples per project:")
print(train_df['project_id'].value_counts().sort_index())
print(f"\nTest samples per project:")
print(test_df['project_id'].value_counts().sort_index())

# %% [markdown]
# ## 3. Embedding Generation
#
# Generate embeddings for text fields: author, title, and summary

# %%
# Load or create embedding model
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
# Generate embeddings for training data
print("Generating embeddings for training data...")
train_embeddings = emb_model.encode(train_df['combined_text'].tolist(), show_progress_bar=True)

# %%
# Generate embeddings for test data
print("Generating embeddings for test data...")
test_embeddings = emb_model.encode(test_df['combined_text'].tolist(), show_progress_bar=True)

# %%
# Convert to numpy arrays
train_embeddings_array = np.array(train_embeddings)
test_embeddings_array = np.array(test_embeddings)

print(f"Training embeddings shape: {train_embeddings_array.shape}")
print(f"Test embeddings shape: {test_embeddings_array.shape}")

# %%
# Save embeddings for future use
np.save(data_folder / "train_embeddings_separate.npy", train_embeddings_array)
np.save(data_folder / "test_embeddings_separate.npy", test_embeddings_array)
print("Embeddings saved!")

# %% [markdown]
# ## 4. Train Separate XGBoost Models Per Project
#
# Instead of using project_id as a feature, we train separate models for each project

# %%
# Store models and metadata
project_models = {}
project_metadata = {}

# Get unique project IDs
unique_projects = sorted(train_df['project_id'].unique())
print(f"Training models for {len(unique_projects)} projects: {unique_projects}")

# %%
# Train a model for each project
for project_id in tqdm(unique_projects, desc="Training project models"):
    print(f"\n{'='*60}")
    print(f"Training model for project {project_id}")
    print(f"{'='*60}")

    # Filter data for this project
    train_mask = train_df['project_id'] == project_id
    test_mask = test_df['project_id'] == project_id

    X_train_proj = train_embeddings_array[train_mask]
    y_train_proj = train_df.loc[train_mask, 'label_binary'].values

    X_test_proj = test_embeddings_array[test_mask]
    y_test_proj = test_df.loc[test_mask, 'label_binary'].values

    print(f"Training samples: {len(X_train_proj)}")
    print(f"Test samples: {len(X_test_proj)}")
    print(f"Training class distribution: {np.bincount(y_train_proj)}")

    # Calculate scale_pos_weight for imbalanced data
    n_negative = (y_train_proj == 0).sum()
    n_positive = (y_train_proj == 1).sum()

    if n_positive == 0:
        print(f"WARNING: No positive samples for project {project_id}, skipping...")
        continue

    scale_pos_weight = n_negative / n_positive
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'tree_method': 'hist',
        'early_stopping_rounds': 20
    }

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train_proj, label=y_train_proj)
    dtest = xgb.DMatrix(X_test_proj, label=y_test_proj)

    # Train model
    evals = [(dtrain, 'train'), (dtest, 'test')]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=params['n_estimators'],
        evals=evals,
        early_stopping_rounds=params['early_stopping_rounds'],
        verbose_eval=False
    )

    # Store model and metadata
    project_models[project_id] = model
    project_metadata[project_id] = {
        'train_samples': len(X_train_proj),
        'test_samples': len(X_test_proj),
        'train_positive': int(n_positive),
        'train_negative': int(n_negative),
        'scale_pos_weight': float(scale_pos_weight),
        'best_iteration': model.best_iteration
    }

    print(f"Model trained successfully (best iteration: {model.best_iteration})")

print(f"\n{'='*60}")
print(f"Trained {len(project_models)} models successfully")
print(f"{'='*60}")

# %% [markdown]
# ## 5. Save Models
#
# Save each model with the naming convention expected by the application

# %%
# Save models following the application's naming pattern
for project_id, model in project_models.items():
    model_filename = f"xgb_embedding_classifier_project_{project_id}.json"
    model_path = model_output_folder / model_filename
    model.save_model(str(model_path))
    print(f"Saved model for project {project_id}: {model_filename}")

# %%
# Save metadata
metadata_path = data_folder / "xgb_project_models_metadata.pkl"
with open(metadata_path, 'wb') as f:
    pickle.dump(project_metadata, f)
print(f"\nMetadata saved to: {metadata_path}")

# %%
# Display metadata
metadata_df = pd.DataFrame(project_metadata).T
print("\nProject Models Metadata:")
print("="*80)
print(metadata_df.to_string())

# %% [markdown]
# ## 6. Model Evaluation
#
# Evaluate each project-specific model on its test set

# %%
# Evaluate each model
project_results = []

for project_id in sorted(project_models.keys()):
    model = project_models[project_id]

    # Get test data for this project
    test_mask = test_df['project_id'] == project_id
    X_test_proj = test_embeddings_array[test_mask]
    y_test_proj = test_df.loc[test_mask, 'label_binary'].values

    if len(X_test_proj) == 0:
        print(f"No test samples for project {project_id}")
        continue

    # Make predictions
    dtest = xgb.DMatrix(X_test_proj)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    if len(np.unique(y_test_proj)) < 2:
        print(f"Only one class in test set for project {project_id}, skipping metrics...")
        continue

    cm = confusion_matrix(y_test_proj, y_pred)
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    roc_auc = roc_auc_score(y_test_proj, y_pred_proba)

    project_results.append({
        'project_id': project_id,
        'test_samples': len(X_test_proj),
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'roc_auc': roc_auc,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    })

# %%
# Display results
results_df = pd.DataFrame(project_results)
print("\nPer-Project Model Performance:")
print("="*100)
print(results_df.to_string(index=False))

# %%
# Calculate overall statistics
print("\nOverall Statistics:")
print("="*60)
print(f"Mean Recall: {results_df['recall'].mean():.4f} (±{results_df['recall'].std():.4f})")
print(f"Mean Precision: {results_df['precision'].mean():.4f} (±{results_df['precision'].std():.4f})")
print(f"Mean F1: {results_df['f1'].mean():.4f} (±{results_df['f1'].std():.4f})")
print(f"Mean ROC AUC: {results_df['roc_auc'].mean():.4f} (±{results_df['roc_auc'].std():.4f})")

# %% [markdown]
# ## 7. Visualize Per-Project Performance

# %%
# Visualize per-project performance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

results_df.plot(x='project_id', y='recall', kind='bar', ax=axes[0, 0], legend=False, color='skyblue')
axes[0, 0].set_title('Recall by Project')
axes[0, 0].set_ylabel('Recall')
axes[0, 0].set_xlabel('Project ID')
axes[0, 0].axhline(y=results_df['recall'].mean(), color='r', linestyle='--',
                   label=f'Mean: {results_df["recall"].mean():.3f}')
axes[0, 0].legend()
axes[0, 0].set_ylim([0, 1])

results_df.plot(x='project_id', y='precision', kind='bar', ax=axes[0, 1], legend=False, color='lightcoral')
axes[0, 1].set_title('Precision by Project')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_xlabel('Project ID')
axes[0, 1].axhline(y=results_df['precision'].mean(), color='r', linestyle='--',
                   label=f'Mean: {results_df["precision"].mean():.3f}')
axes[0, 1].legend()
axes[0, 1].set_ylim([0, 1])

results_df.plot(x='project_id', y='f1', kind='bar', ax=axes[1, 0], legend=False, color='lightgreen')
axes[1, 0].set_title('F1 Score by Project')
axes[1, 0].set_ylabel('F1 Score')
axes[1, 0].set_xlabel('Project ID')
axes[1, 0].axhline(y=results_df['f1'].mean(), color='r', linestyle='--',
                   label=f'Mean: {results_df["f1"].mean():.3f}')
axes[1, 0].legend()
axes[1, 0].set_ylim([0, 1])

results_df.plot(x='project_id', y='roc_auc', kind='bar', ax=axes[1, 1], legend=False, color='plum')
axes[1, 1].set_title('ROC AUC by Project')
axes[1, 1].set_ylabel('ROC AUC')
axes[1, 1].set_xlabel('Project ID')
axes[1, 1].axhline(y=results_df['roc_auc'].mean(), color='r', linestyle='--',
                   label=f'Mean: {results_df["roc_auc"].mean():.3f}')
axes[1, 1].legend()
axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Detailed Analysis for Each Project

# %%
# Create detailed visualizations for each project
for project_id in sorted(project_models.keys()):
    model = project_models[project_id]

    # Get test data
    test_mask = test_df['project_id'] == project_id
    X_test_proj = test_embeddings_array[test_mask]
    y_test_proj = test_df.loc[test_mask, 'label_binary'].values

    if len(X_test_proj) == 0 or len(np.unique(y_test_proj)) < 2:
        continue

    # Make predictions
    dtest = xgb.DMatrix(X_test_proj)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Create figure
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'Project {project_id} - Detailed Analysis', fontsize=14, fontweight='bold')

    # Confusion Matrix
    ax1 = plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_test_proj, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues', values_format='d', ax=ax1)
    ax1.set_title('Confusion Matrix')

    # ROC Curve
    ax2 = plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_test_proj, y_pred_proba)
    roc_auc = roc_auc_score(y_test_proj, y_pred_proba)
    ax2.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    # Precision-Recall Curve
    ax3 = plt.subplot(1, 3, 3)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test_proj, y_pred_proba)
    avg_precision = average_precision_score(y_test_proj, y_pred_proba)
    ax3.plot(recall_curve, precision_curve, lw=2, label=f'PR (AP = {avg_precision:.3f})')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve')
    ax3.legend(loc='lower left')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    # Print classification report
    print(f"\nClassification Report - Project {project_id}:")
    print("="*60)
    print(classification_report(y_test_proj, y_pred, target_names=['Negative', 'Positive']))

# %% [markdown]
# ## 9. Threshold Optimization Per Project
#
# Find optimal thresholds for each project to maximize recall while maintaining acceptable precision

# %%
# Optimize thresholds for each project
optimal_thresholds = {}

for project_id in sorted(project_models.keys()):
    model = project_models[project_id]

    # Get test data
    test_mask = test_df['project_id'] == project_id
    X_test_proj = test_embeddings_array[test_mask]
    y_test_proj = test_df.loc[test_mask, 'label_binary'].values

    if len(X_test_proj) == 0 or len(np.unique(y_test_proj)) < 2:
        continue

    # Make predictions
    dtest = xgb.DMatrix(X_test_proj)
    y_pred_proba = model.predict(dtest)

    # Test different thresholds
    thresholds_to_test = np.arange(0.1, 0.9, 0.05)
    threshold_metrics = []

    for threshold in thresholds_to_test:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        cm_thresh = confusion_matrix(y_test_proj, y_pred_thresh)
        tn, fp, fn, tp = cm_thresh.ravel()

        recall_thresh = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_thresh = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_thresh = 2 * (precision_thresh * recall_thresh) / (precision_thresh + recall_thresh) if (precision_thresh + recall_thresh) > 0 else 0

        threshold_metrics.append({
            'threshold': threshold,
            'recall': recall_thresh,
            'precision': precision_thresh,
            'f1': f1_thresh
        })

    threshold_df = pd.DataFrame(threshold_metrics)

    # Find optimal threshold (maximize recall with precision >= 0.6)
    min_precision = 0.6
    viable_thresholds = threshold_df[threshold_df['precision'] >= min_precision]

    if len(viable_thresholds) > 0:
        optimal_row = viable_thresholds.loc[viable_thresholds['recall'].idxmax()]
    else:
        # If no threshold meets precision requirement, maximize F1
        optimal_row = threshold_df.loc[threshold_df['f1'].idxmax()]

    optimal_thresholds[project_id] = {
        'threshold': optimal_row['threshold'],
        'recall': optimal_row['recall'],
        'precision': optimal_row['precision'],
        'f1': optimal_row['f1']
    }

    print(f"\nProject {project_id} - Optimal Threshold:")
    print(f"  Threshold: {optimal_row['threshold']:.2f}")
    print(f"  Recall: {optimal_row['recall']:.4f}")
    print(f"  Precision: {optimal_row['precision']:.4f}")
    print(f"  F1: {optimal_row['f1']:.4f}")

# %%
# Save optimal thresholds
thresholds_path = data_folder / "xgb_project_optimal_thresholds.pkl"
with open(thresholds_path, 'wb') as f:
    pickle.dump(optimal_thresholds, f)
print(f"\nOptimal thresholds saved to: {thresholds_path}")

# %%
# Visualize optimal thresholds
threshold_summary_df = pd.DataFrame(optimal_thresholds).T
print("\nOptimal Thresholds Summary:")
print("="*80)
print(threshold_summary_df.to_string())

# %%
fig, ax = plt.subplots(figsize=(10, 6))
threshold_summary_df.plot(kind='bar', ax=ax)
ax.set_title('Optimal Thresholds and Metrics by Project')
ax.set_ylabel('Value')
ax.set_xlabel('Project ID')
ax.legend(['Threshold', 'Recall', 'Precision', 'F1'])
ax.grid(True, alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Comparison with Single Model Approach
#
# Compare performance against the single model approach from notebook 01d

# %%
print("\n" + "="*80)
print("COMPARISON: Separate Models vs Single Model Approach")
print("="*80)
print("\nSeparate Models (this notebook):")
print(f"  Mean Recall: {results_df['recall'].mean():.4f}")
print(f"  Mean Precision: {results_df['precision'].mean():.4f}")
print(f"  Mean F1: {results_df['f1'].mean():.4f}")
print(f"  Mean ROC AUC: {results_df['roc_auc'].mean():.4f}")
print("\nNote: Compare these results with the per-project metrics from notebook 01d")
print("to evaluate if separate models provide better project-specific performance.")

# %%
