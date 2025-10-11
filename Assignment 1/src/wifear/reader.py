import os
import json
from typing import cast
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "ptwiki-articles-with-redirects.arrow")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "ptwiki_clean.json")
LIMIT = 2000


def read_arrow_to_json(path: str, output_json: str, limit: int = 2000) -> DataFrame:
    """Read an Arrow dataset, clean it, keep only N records, and export to JSON."""
    print(f"Reading Arrow dataset from: {path}")

    # Load dataset
    df: DataFrame = pd.read_feather(path)
    print(f"Dataset loaded successfully with {len(df):,} documents")
    print("Available columns:", list(df.columns))

    # Cleaning
    redirect_col = cast(Series, df["redirect"])
    text_col = cast(Series, df["text"])

    mask_not_redirect = ~redirect_col.astype(bool)
    mask_not_empty = text_col.notna() & (text_col.astype(str).str.strip() != "")
    mask = mask_not_redirect & mask_not_empty

    df = df.loc[mask].copy()
    print(f"Cleaned dataset: {len(df):,} valid documents remaining")

    # Limit to first N records
    df = df.head(limit)
    print(f"Limiting dataset to first {limit:,} records")

    # Convert NumPy arrays in 'out_links' to Python lists
    if "out_links" in df.columns:
        df["out_links"] = df["out_links"].apply(
            lambda x: list(x) if isinstance(x, (np.ndarray, list)) else []
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # Convert to JSON and save
    records = df.to_dict(orient="records")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    print(f"Saved cleaned dataset to: {output_json}")

    # Preview first 2 documents
    print("\nSample documents:")
    print(json.dumps(records[:2], ensure_ascii=False, indent=4))

    return df


if __name__ == "__main__":
    _ = read_arrow_to_json(DATA_PATH, OUTPUT_JSON, LIMIT)
