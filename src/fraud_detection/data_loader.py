from __future__ import annotations

import pandas as pd
from datasets import load_dataset


def load_huggingface_dataset(dataset_name: str, split: str = "train") -> pd.DataFrame:
    ds = load_dataset(dataset_name)
    return ds[split].to_pandas()


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
