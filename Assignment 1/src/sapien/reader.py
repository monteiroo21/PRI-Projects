import os
from typing import cast

import pandas as pd
from pandas import DataFrame, Series

DATA_PATH = "../../data/ptwiki-articles-with-redirects.arrow"
OUTPUT_CSV = "../../data/ptwiki_clean_subset.csv"


def read_arrow_to_csv(path: str, output_csv: str) -> DataFrame:
    """Read an Arrow dataset, clean it, and export it to CSV."""
    print(f"Reading Arrow dataset from: {path}")

    df: DataFrame = pd.read_feather(path)

    print(f"Dataset loaded successfully with {len(df):,} documents")
    columns: list[str] = list(df.columns)
    print("Available columns:", columns)

    redirect_col = cast(Series, df["redirect"])
    text_col = cast(Series, df["text"])

    mask_not_redirect = ~redirect_col.astype(bool)
    mask_not_empty = text_col.notna() & (text_col.astype(str).str.strip() != "")
    mask = mask_not_redirect & mask_not_empty

    df = df.loc[mask].copy()
    print(f"Cleaned dataset: {len(df):,} valid documents remaining")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned dataset to: {output_csv}")

    print("\nSample documents:")
    print(df.head(3))

    return df


if __name__ == "__main__":
    _ = read_arrow_to_csv(DATA_PATH, OUTPUT_CSV)
