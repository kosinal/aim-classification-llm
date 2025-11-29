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
        label=row['label']  # Convert bool to string for LLM
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
print(f"Total Train size: {len(trainset)}, Total Dev size: {len(devset)}")


# %%
class FlagAssessor(dspy.Signature):
    """
    Analyze the text and determine if it should be recommended to user based on the project context.
    Output a recomendation score between 0.0 and 1.0, where 1.0 is highly recommendable.
    """
    project_id = dspy.InputField(
        desc="The ID of the project this content belongs to. Relevance context depends on this.")
    summary = dspy.InputField(desc="The summary of the content to evaluate.")

    # We ask for a score to allow threshold tuning for your specific FPR requirements
    reasoning = dspy.OutputField(desc="Step-by-step analysis of why this is relevant or not.")
    prediction_score = dspy.OutputField(
        desc="A float score between 0.0 and 1.0 indicating probability of being recommended.")
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
# Currently, I balanced the dataset.
def maximize_recall_metric(gold, pred, trace=None):
    # 1. Parse text decisions
    pred_flag = str(pred.prediction).strip().lower() == 'positive'
    gold_flag = str(gold.label).strip().lower() == 'positive'  # Added str() for safety

    try:
        score = float(pred.prediction_score)
    except:
        return 0.0  # Punishment for invalid score format

    # If text says positive, score must be high (>0.5). If text says negative, score must be low.
    score_aligned = (pred_flag and score > 0.5) or (not pred_flag and score <= 0.5)

    if not score_aligned:
        return 0.0  # Penalize hallucinated scores that don't match the decision

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
logging.getLogger("dspy.utils").setLevel(logging.ERROR)
logging.getLogger("dspy.utils.parallelizer").setLevel(logging.ERROR)

# %% [markdown]
# ## Project-Specific Training Loop

# %%
unique_projects = df['project_id'].unique()
print(f"Found projects: {unique_projects}")

# Store compiled models and their stats
project_models = {}
project_stats = {}

set_size = 25

for pid in unique_projects:
    str_pid = str(pid)
    print(f"\n{'=' * 30}\nStarting compilation for Project ID: {str_pid}\n{'=' * 30}")

    # 1. Filter Data for this specific project
    p_trainset = [ex for ex in trainset if ex.project_id == str_pid]
    p_devset = [ex for ex in devset if ex.project_id == str_pid]

    if not p_trainset:
        print(f"Skipping {str_pid}: No training data.")
        continue

    # 2. Create balanced subsets for the optimizer (Few-Shot Candidates)
    positives = [ex for ex in p_trainset if ex.label == 'positive']
    negatives = [ex for ex in p_trainset if ex.label == 'negative']

    val_positives = [ex for ex in p_devset if ex.label == 'positive']
    val_negatives = [ex for ex in p_devset if ex.label == 'negative']

    print(f"  Train Stats -> Pos: {len(positives)}, Neg: {len(negatives)}")
    print(f"  Val Stats   -> Pos: {len(val_positives)}, Neg: {len(val_negatives)}")

    # Ensure we don't sample more than available
    optimizer_trainset = (
            random.sample(positives, min(len(positives), set_size)) +
            random.sample(negatives, min(len(negatives), set_size))
    )

    optimizer_valset = (
            random.sample(val_positives, min(len(val_positives), set_size)) +
            random.sample(val_negatives, min(len(val_negatives), set_size))
    )

    # 3. Initialize fresh optimizer and model for this project
    # We re-initialize to ensure no leakage of demos between projects
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=maximize_recall_metric,
        max_bootstrapped_demos=4,
        num_candidate_programs=6,
        num_threads=8,
    )

    uncompiled_model = FlagClassifier()

    # 4. Compile
    print(f"  Compiling model for {str_pid}...")
    try:
        compiled_model = optimizer.compile(
            uncompiled_model,
            trainset=optimizer_trainset,
            valset=optimizer_valset
        )

        # 5. Save individually
        timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "_").replace("-", "_").replace("T", "_")
        filename = f'flag_classifier_project_{str_pid}_{timestamp}.json'
        save_path = data_folder / filename
        compiled_model.save(save_path)
        print(f"  Saved to: {filename}")

        project_models[str_pid] = compiled_model

    except Exception as e:
        print(f"  FAILED to compile project {str_pid}: {e}")

# %% [markdown]
# ## Multi-Model Evaluation

# %%
# Iterate through models to evaluate them against their specific validation sets
plt.figure(figsize=(10, 8))

for pid, model in project_models.items():
    print(f"\nEvaluating Project {pid}...")

    # Get the specific dev set for this project
    p_devset = [ex for ex in devset if ex.project_id == pid]

    y_true = []
    y_scores = []

    for example in p_devset:
        try:
            pred = model(project_id=example.project_id, summary=example.summary)

            # Extract Score
            try:
                score = float(pred.prediction_score)
            except (ValueError, TypeError):
                score = 1.0 if str(pred.prediction).strip().lower() in ['positive', 'true'] else 0.0

            y_scores.append(score)

            # Extract Label
            is_pos = str(example.label).strip().lower() in ['true', 'positive']
            y_true.append(1 if is_pos else 0)

        except Exception as e:
            continue

    if len(set(y_true)) < 2:
        print(f"  Skipping ROC for {pid}: Single class present in dev set.")
        continue

    # Calculate ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    # Plot
    plt.plot(fpr, tpr, lw=2, label=f'Proj {pid} (AUC = {roc_auc:.2f})')

    print(f"  AUC: {roc_auc:.4f}")

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves by Project')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
