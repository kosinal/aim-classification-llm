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

# %%
import dspy
import pandas as pd
import numpy as np
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path
import os
from datetime import datetime
from matplotlib import pyplot as plt
import random
import logging

# %%

# %%
# Configuration
data_folder = Path("../_data/")
os.environ["AIM_OPENAI_KEY"] = "9bd2d13588a14f4cae62325fc68d7d64"

# Model configuration
MODEL_NAME = "gpt-5-hiring"
MINI_MODEL_NAME = "gpt-5-mini-hiring"
API_VERSION = "2025-03-01-preview"
ENDPOINT = "https://aim-australia-east.openai.azure.com/"

# %%
df = pd.read_parquet(data_folder / "data.parquet")

print(f"Total samples: {len(df)}")
print(f"\nClass distribution:")
print(df['label'].value_counts())
print(f"\nProject distribution:")
print(df['project_id'].value_counts())


# %%
# Feature Engineering: Combine Title + Summary if useful, otherwise just Summary
df['input_text'] = df['title'] + df['summary']

# %%
dataset = []
for _, row in df.iterrows():
    dataset.append(dspy.Example(
        project_id=str(row['project_id']),
        summary=row['input_text'],
        label=row['label'] # Convert bool to string for LLM
    ).with_inputs('project_id', 'summary'))

# %%
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# %%
strat_key = df['project_id'].astype(str) + "_" + df['label'].astype(str)

# %%
train_idx, dev_idx = next(splitter.split(df, strat_key))
trainset = [dataset[i] for i in train_idx]
devset = [dataset[i] for i in dev_idx]

# %%
print(f"Train size: {len(trainset)}, Dev size: {len(devset)}")


# %%
class FlagAssessor(dspy.Signature):
    """
    Analyze the text and determine if it should be recommended to user based on the project context.
    Output a recomendation score between 0.0 and 1.0, where 1.0 is highly recommendable.
    """
    project_id = dspy.InputField(desc="The ID of the project this content belongs to. Relevance context depends on this.")
    summary = dspy.InputField(desc="The summary of the content to evaluate.")
    
    # We ask for a score to allow threshold tuning for your specific FPR requirements
    reasoning = dspy.OutputField(desc="Step-by-step analysis of why this is relevant or not.")
    prediction_score = dspy.OutputField(desc="A float score between 0.0 and 1.0 indicating probability of being recommended.")
    prediction = dspy.OutputField(desc="Binary decision: 'positive' or 'negative'.")


# %%
class FlagClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought helps with the complex reasoning required per project
        self.prog = dspy.ChainOfThought(FlagAssessor)

    def forward(self, project_id, summary):
        return self.prog(project_id=project_id, summary=summary)


# %%
# def maximize_recall_metric(gold, pred, trace=None):
#     # 1. Parse strings to booleans safely
#     pred_flag = pred.prediction.strip().lower() == 'positive'
#     gold_flag = gold.label.strip().lower() == 'positive'

#     # 2. Logic to force high TPR
#     if gold_flag:
#         # CRITICAL: If it IS a flag, we strictly demand a True prediction.
#         if pred_flag:
#             return 1.0  # Nailed it (TP)
#         else:
#             return 0.0  # Missed it (FN) - Maximum Penalty
            
#     else: # gold is False
#         # If it is NOT a flag, we are lenient. 
#         if not pred_flag:
#             return 1.0  # Correctly ignored (TN)
#         else:
#             # We give a HIGH score for False Positives (e.g., 0.8 or 0.9).
#             return 0.8

# Currently, I balanced the dataset.
def maximize_recall_metric(gold, pred, trace=None):
    # 1. Parse text decisions
    pred_flag = str(pred.prediction).strip().lower() == 'positive'
    gold_flag = str(gold.label).strip().lower() == 'positive' # Added str() for safety
    # print(str(pred.prediction), str(gold.label))
    # print(pred_flag, gold_flag)
    try:
        score = float(pred.prediction_score)
    except:
        # print("Invalid score {pred.prediction_score}")
        return 0.0 # Punishment for invalid score format

    # If text says positive, score must be high (>0.5). If text says negative, score must be low.
    score_aligned = (pred_flag and score > 0.5) or (not pred_flag and score <= 0.5)
    
    if not score_aligned:
        # print("Penalization for score mismatch")
        return 0.0 # Penalize hallucinated scores that don't match the decision

    # 4. Standard Recall Logic
    if gold_flag and pred_flag:
        return 1.0
    elif gold_flag and not pred_flag:
        return 0.0 
    elif not gold_flag and pred_flag:
        return 0.5 
    return 1.0


# %%
lm = dspy.LM(
    model=f"azure/{MINI_MODEL_NAME}",
    api_base=ENDPOINT,
    api_version=API_VERSION,
    api_key=os.environ["AIM_OPENAI_KEY"],
    temperature=1.0  # Only allowed
)

dspy.configure(lm=lm)
dspy.settings.configure(lm=lm)

# %%
optimizer = BootstrapFewShotWithRandomSearch(
    metric=maximize_recall_metric,
    max_bootstrapped_demos=4, # How many examples to put in the prompt
    num_candidate_programs=6, # How many prompts to generate and test
    num_threads=8,
)

# %%
positives = [ex for ex in trainset if ex.label == 'positive']
negatives = [ex for ex in trainset if ex.label == 'negative']

val_positives = [ex for ex in devset if ex.label == 'positive']
val_negatives = [ex for ex in devset if ex.label == 'negative']

print(f"Train sizes: {len(positives)}/{len(negatives)}")
print(f"Val sizes: {len(val_positives)}/{len(val_negatives)}")

# %%
set_size = 25

# %%
optimizer_trainset = random.sample(positives, min(len(positives), set_size)) + random.sample(negatives, min(len(negatives), set_size))
optimizer_valset = random.sample(val_positives, min(len(val_positives), set_size)) + random.sample(val_negatives, min(len(val_negatives), set_size))

# %%
uncompiled_model = FlagClassifier()

# %%
logging.getLogger("dspy.utils").setLevel(logging.ERROR)
logging.getLogger("dspy.utils.parallelizer").setLevel(logging.ERROR)

# %%
print("Compiling model...")
compiled_model = optimizer.compile(uncompiled_model, trainset=optimizer_trainset, valset=optimizer_valset)

# %%
compiled_model.save(data_folder / f'flag_classifier_optimized_{datetime.now().isoformat(timespec="seconds").replace(":", "_").replace("-", "_").replace("T", "_")}.json')

print("Model saved successfully.")

# %%
model = compiled_model

# %%
y_true = []
y_scores = []

thresholds = [0.1, 0.01, 0.001]

for example in optimizer_valset:
    try:
        pred = model(project_id=example.project_id, summary=example.summary)        
        
        # FIX 1: Use prediction_score (matches Signature)
        try:
            score = float(pred.prediction_score)
        except (ValueError, TypeError):
            # Fallback based on text prediction
            score = 1.0 if pred.prediction.strip().lower() in ['positive', 'true'] else 0.0
            
        y_scores.append(score)
        
        # FIX 2: Handle label (matches Dataset)
        is_pos = str(example.label).strip().lower() in ['true', 'positive']
        y_true.append(1 if is_pos else 0)
        
    except Exception as e:
        print(f"Error on example: {e}")
        continue

# %%
fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

# %%
results = {}
for target_fpr in thresholds:
    idx = np.argmin(np.abs(fpr - target_fpr))
    actual_fpr = fpr[idx]
    actual_tpr = tpr[idx]
    cutoff = roc_thresholds[idx]
    
    results[f"FPR_{target_fpr}"] = {"TPR": actual_tpr, "Actual_FPR": actual_fpr, "Score_Cutoff": cutoff}

# %%
roc_auc = roc_auc_score(y_true, y_scores)

# %%
plt.figure(figsize=(8, 6))

# Plot the ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')

# Plot the baseline (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random Guess (AUC = 0.50)')

# Add labels, title, and legend
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
