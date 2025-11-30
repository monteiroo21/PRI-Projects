import os
import json
import sqlite3

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if not os.path.exists(os.path.join(BASE_DIR, "data")):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

JSONL_PATH = os.path.join(BASE_DIR, "data", "index_final.jsonl")
DB_PATH = os.path.join(BASE_DIR, "index.db")


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

print("Done importing index into SQLite")
conn.close()
