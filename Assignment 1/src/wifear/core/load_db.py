import json
import sqlite3

DB_PATH = "index.db"
JSONL_PATH = "../../../data/index_final.jsonl"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute(
    """
CREATE TABLE IF NOT EXISTS inverted_index (
    term TEXT PRIMARY KEY,
    postings TEXT
)
"""
)

batch = []
BATCH_SIZE = 1000

with open(JSONL_PATH, encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        for term, postings in data.items():
            batch.append((term, json.dumps(postings)))
        if len(batch) >= BATCH_SIZE:
            cur.executemany("INSERT OR REPLACE INTO inverted_index VALUES (?, ?)", batch)
            conn.commit()
            batch.clear()

if batch:
    cur.executemany("INSERT OR REPLACE INTO inverted_index VALUES (?, ?)", batch)
    conn.commit()

print("✅ Done importing index into SQLite")
conn.close()
