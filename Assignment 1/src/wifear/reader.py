import json
import os
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_JSONL = os.path.join(BASE_DIR, "data", "ptwiki-articles-with-redirects.jsonl")
OUTPUT_JSONL = os.path.join(BASE_DIR, "data", "ptwiki_clean.jsonl")
BATCH_SIZE = 50_000
LIMIT: Optional[int] = None


def clean_record(record: dict) -> bool:
    """Filter and clean a single record from the dataset."""
    if record.get("redirect"):
        return False
    text = record.get("text", "")
    return isinstance(text, str) and text.strip() != ""


def read_jsonl_to_jsonl_in_batches(
    input_path: str, output_path: str, batch_size: int = 50_000, limit: Optional[int] = None
):
    """Read a large JSONL file line by line"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_written = 0
    buffer: List[dict] = []

    with open(input_path, encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for i, line in enumerate(f_in, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed lines

            if not clean_record(record):
                continue

            # Normalize out_links
            if "out_links" in record and not isinstance(record["out_links"], list):
                record["out_links"] = list(record["out_links"]) if record["out_links"] else []

            buffer.append(record)

            # Write batch
            if len(buffer) >= batch_size:
                for rec in buffer:
                    json.dump(rec, f_out, ensure_ascii=False)
                    f_out.write("\n")
                total_written += len(buffer)
                print(f"Processed {i:,} lines → total written: {total_written:,}")
                buffer.clear()

                if limit and total_written >= limit:
                    print("Reached limit, stopping.")
                    break

        # Write remaining buffer
        if buffer and (not limit or total_written < limit):
            for rec in buffer[: (limit - total_written) if limit else None]:
                json.dump(rec, f_out, ensure_ascii=False)
                f_out.write("\n")
            total_written += len(buffer)

    print(f"\nSaved {total_written:,} cleaned documents to: {output_path}")


if __name__ == "__main__":
    read_jsonl_to_jsonl_in_batches(INPUT_JSONL, OUTPUT_JSONL, batch_size=BATCH_SIZE, limit=LIMIT)
