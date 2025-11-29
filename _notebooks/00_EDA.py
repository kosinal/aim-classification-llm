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

# %%
# !poetry add pandas --group dev

# %%

# %%
data_folder = Path("../_data/")

# %%

# %%
input_data = json.loads((data_folder / "InsightRelevancyDataset.json").read())

# %%
pd.read_json()

# %%
