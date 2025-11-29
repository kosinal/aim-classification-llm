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
import os
import json
import pandas as pd
from pathlib import Path
from openai import AzureOpenAI
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# #!poetry add seaborn --group dev

# %%

# %%
data_folder = Path("../_data/")

# %%
input_data = json.loads((data_folder / "InsightRelevancyDataset.json").read_text())

# %%

# %%
print(f"Pos/Neg ratio: {len(input_data['positives'])}/{len(input_data['negatives'])}")

# %%

# %%
model_name = "gpt-5-hiring"
mini_model_name = "gpt-5-mini-hiring"
llm_client = client = AzureOpenAI(
    api_version="2025-03-01-preview",
    azure_endpoint="https://aim-australia-east.openai.azure.com/",
    api_key=os.environ["AIM_OPENAI_KEY"]
)

# %%

# %%
pos_data = pd.DataFrame(input_data['positives'])
pos_data["label"] = "positive"

# %%
neg_data = pd.DataFrame(input_data['negatives'])
neg_data["label"] = "negative"

# %%
all_data = pd.concat([pos_data, neg_data], ignore_index=True)

# %%
all_data

# %%
# Too slow for now, would take too long for make it faster for 1MD assignment
# Also it seems all sumary is in English only, so lets try leverage this.
#
# def translate_to_english(text):
#     """Translate text to English using Azure OpenAI. Returns original if already in English."""
#     if pd.isna(text) or text == "null" or not text:
#         return text
#     
#     try:
#         response = llm_client.chat.completions.create(
#             model=mini_model_name,
#             messages=[
#                 {"role": "system", "content": "You are a translator. If the text is already in English, return it unchanged. Otherwise, translate it to English. Return ONLY the translated text without any explanations."},
#                 {"role": "user", "content": f"Translate this text to English:\n\n{text}"}
#             ],
#             max_completion_tokens=5000
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"Error translating: {e}")
#         return text

# %%
# translate_to_english("Ahoj, jak se vede")

# %%
# Translate title and summary columns to English with progress tracking
# 
# translation_fields = ["title", "summary", "text"]
# 
# for translation_field in translation_fields:
#     titles_translated = []
#     for title in tqdm(all_data[translation_field], desc=f"Translating {translation_field}"):
#         titles_translated.append(translate_to_english(title))
#     all_data[f'{translation_field}_en'] = titles_translated
# 
# print("\nTranslation complete!")
# all_data.head()

# %%

# %% [markdown]
# # Flags Column Exploration

# %%
# Extract all unique keys from flags dictionaries
all_flag_keys = set()
for flags_dict in all_data['flags'].dropna():
    if isinstance(flags_dict, dict):
        all_flag_keys.update(flags_dict.keys())

print(f"Unique flag keys found: {sorted(all_flag_keys)}")
print(f"Total number of unique flags: {len(all_flag_keys)}")

# %%
notes_vals = []
insightful_vals = []
for flags_dict in all_data['flags']:
    if flags_dict is None or len(flags_dict) == 0:
        notes_vals.append("")
        insightful_vals.append("")
        continue
    notes_vals.append(flags_dict.get("note", ""))
    insightful_vals.append("\n".join(flags_dict.get("insightful", [""])))

all_data['flag_note'] = notes_vals
all_data['flag_insightful'] = insightful_vals

# %%
all_data[(all_data.flag_insightful != "") & (all_data.label == "negative")]

# %%
all_data[(all_data.flag_insightful != "") & (all_data.label != "negative")]

# %%
# Ok, flags are only for negatives, so it wont help us much

# %% [markdown]
# # Confusion Matrix: isFlagged vs Label

# %%
# Additional statistics about the relationship
print("Distribution Analysis:")
print("=" * 60)

# Count by label
print("\nCounts by label:")
print(all_data['label'].value_counts())

# Count by isFlagged
print("\nCounts by isFlagged:")
print(all_data['isFlagged'].value_counts())

# Cross-tabulation
print("\nCross-tabulation (label vs isFlagged):")
ct = pd.crosstab(all_data['label'], all_data['isFlagged'], margins=True)
print(ct)

# Percentage breakdown
print("\nPercentage breakdown:")
ct_pct = pd.crosstab(all_data['label'], all_data['isFlagged'], normalize='index') * 100
print(ct_pct.round(2))

# %%
all_data.drop(["flag_note", "flags", "flag_insightful", "id", "isFlagged"], axis=1, inplace=True)

# %%
#Ok, so it is duplicate

# %%
all_data.info()

# %%
#Fill missing author with Unknwon autor.
all_data['author'] = all_data['author'].fillna("Unknown Author")

# %%
all_data.to_parquet(data_folder / "data.parquet")
