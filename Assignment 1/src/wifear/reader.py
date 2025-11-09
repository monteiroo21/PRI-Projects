import json
import os
from typing import List
import pyarrow.dataset as ds

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "ptwiki-articles-with-redirects.arrow")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "ptwiki_clean.jsonl")
BATCH_SIZE = 50_000
LIMIT = None


def clean_record(record: dict) -> bool:
    """Filter and clean a single record from the dataset."""
    if record.get("redirect"):
        return False
    text = record.get("text", "")
    return isinstance(text, str) and text.strip() != ""


def read_arrow_to_jsonl_in_batches(path: str, output_json: str, batch_size: int = 50_000, limit: int | None = None):
    """
    Read a large Arrow dataset in small batches using pyarrow.dataset,
    clean it, and incrementally export to JSON.
    """
    dataset = ds.dataset(path, format="feather")  # Works for .arrow or .feather files
    scanner = dataset.scanner(batch_size=batch_size)
    total_written = 0
    # first_record = True

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f_out:

        for i, record_batch in enumerate(scanner.to_batches()):
            df = record_batch.to_pandas()

            # Convert 'out_links' field to list
            if "out_links" in df.columns:
                df["out_links"] = df["out_links"].apply(
                    lambda x: list(x) if isinstance(x, (list, tuple)) else []
                )

            # Clean and filter
            records = [r for r in df.to_dict(orient="records") if clean_record(r)]

            # Apply limit
            if limit and total_written + len(records) > limit:
                records = records[: limit - total_written]

            # Write each record as a line in JSONL format
            for rec in records:
                json.dump(rec, f_out, ensure_ascii=False)
                f_out.write("\n")

            total_written += len(records)
            print(f"Processed batch {i+1} → total written: {total_written:,}")

            if limit and total_written >= limit:
                print("Reached limit, stopping.")
                break

    print(f"\nSaved {total_written:,} cleaned documents to: {output_json}")


if __name__ == "__main__":
    read_arrow_to_jsonl_in_batches(DATA_PATH, OUTPUT_JSON, batch_size=BATCH_SIZE, limit=LIMIT)
