import os
import json
import sqlite3
from collections import defaultdict

# Caminhos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if not os.path.exists(os.path.join(BASE_DIR, "data")):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

JSONL_PATH = os.path.join(BASE_DIR, "data", "index_final.jsonl")
DB_PATH = os.path.join(BASE_DIR, "index.db")

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA journal_mode = WAL")
conn.execute("PRAGMA synchronous = NORMAL")
cur = conn.cursor()

cur.execute("""
    CREATE TABLE inverted_index (
        term TEXT PRIMARY KEY,
        doc_freq INTEGER,
        postings TEXT
    )
""")

cur.execute("""
    CREATE TABLE doc_lengths (
        doc_id INTEGER PRIMARY KEY,
        length INTEGER
    )
""")

cur.execute("""
    CREATE TABLE forward_index (
        doc_id INTEGER PRIMARY KEY,
        terms_data TEXT
    )
""")

print("Processing data...")
temp_doc_lengths = defaultdict(int)
temp_forward_index = defaultdict(dict)

batch_inverted = []
BATCH_SIZE = 30000
count = 0

with open(JSONL_PATH, encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        
        for term, postings in data.items():
            doc_freq = len(postings)
            batch_inverted.append((term, doc_freq, json.dumps(postings)))
            
            for doc_id_str, positions in postings.items():
                doc_id = int(doc_id_str)
                freq = len(positions)
                
                temp_doc_lengths[doc_id] += freq
                temp_forward_index[doc_id][term] = freq
        
        count += 1
        if len(batch_inverted) >= BATCH_SIZE:
            cur.executemany("INSERT INTO inverted_index VALUES (?, ?, ?)", batch_inverted)
            conn.commit()
            batch_inverted.clear()
            print(f"Processed {count} lines...", end="\r")

if batch_inverted:
    cur.executemany("INSERT INTO inverted_index VALUES (?, ?, ?)", batch_inverted)
    conn.commit()

print("\nWriting Doc Lengths...")
dl_batch = [(doc_id, length) for doc_id, length in temp_doc_lengths.items()]
for i in range(0, len(dl_batch), BATCH_SIZE):
    cur.executemany("INSERT INTO doc_lengths VALUES (?, ?)", dl_batch[i:i+BATCH_SIZE])
conn.commit()

print("Writing Forward Index...")
fi_batch = []
for doc_id, terms_map in temp_forward_index.items():
    fi_batch.append((doc_id, json.dumps(terms_map)))
    if len(fi_batch) >= BATCH_SIZE:
        cur.executemany("INSERT INTO forward_index VALUES (?, ?)", fi_batch)
        fi_batch.clear()

if fi_batch:
    cur.executemany("INSERT INTO forward_index VALUES (?, ?)", fi_batch)
    conn.commit()

print("Creating SQL indices...")
cur.execute("CREATE INDEX idx_term ON inverted_index(term)")
cur.execute("CREATE INDEX idx_len_doc ON doc_lengths(doc_id)")
cur.execute("CREATE INDEX idx_fwd_doc ON forward_index(doc_id)")

conn.commit()
conn.close()
print("Done! Database ready for fast querying.")