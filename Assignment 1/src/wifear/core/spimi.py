import gzip
import json
import os
from collections import defaultdict
from heapq import merge
from multiprocessing import Pool, cpu_count

import ijson
import psutil

from wifear.core.tokenizer import PortugueseTokenizer


def process_chunk(chunk_id, docs, output_dir, tokenizer_config, doc_offset=0):
    """Worker process: build an inverted index from a batch of documents and write to disk."""

    tokenizer = PortugueseTokenizer(**tokenizer_config)
    index = {}
    total_tokens = 0

    for local_doc_id, doc in enumerate(docs):
        global_doc_id = doc_offset + local_doc_id
        text = doc.get("text", "")
        tokens = tokenizer.tokenize(text)
        total_tokens += len(tokens)

        for pos, term in enumerate(tokens):
            index.setdefault(term, {}).setdefault(global_doc_id, []).append(pos)

    block_path = os.path.join(output_dir, f"block_{chunk_id:03d}.json.gz")
    with gzip.open(block_path, "wt", encoding="utf-8") as f:
        # sort terms alphabetically
        sorted_index = {term: index[term] for term in sorted(index.keys())}
        json.dump(sorted_index, f, ensure_ascii=False, separators=(",", ":"))

    print(f"[Worker {chunk_id}] Block written ({len(index):,} terms) → {block_path}")
    return len(docs), total_tokens


class SPIMIIndexer:
    """Efficient Single-Pass In-Memory Indexer (SPIMI) for large-scale corpora.

    Key features:
    - Processes the corpus incrementally (no full JSON load).
    - Builds positional inverted index respecting a 2 GB memory limit.
    - Writes compressed blocks (.gz) to disk.
    - Merges partial indexes with min_df filtering.
    """

    def __init__(
        self,
        tokenizer: PortugueseTokenizer,
        output_dir: str = "index_blocks",
        memory_limit_mb: int = 2000,
    ):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.memory_limit = memory_limit_mb * 1024 * 1024
        os.makedirs(output_dir, exist_ok=True)

    def _memory_full(self) -> bool:
        """Return True if current memory usage exceeds threshold."""
        rss = psutil.Process(os.getpid()).memory_info().rss
        return rss > self.memory_limit * 0.9  # flush before hitting hard limit

    def index_documents(self, json_path: str, chunk_size: int = 5000):
        """Parallel SPIMI indexing with bounded memory usage."""
        os.makedirs(self.output_dir, exist_ok=True)

        num_docs = 0
        total_tokens = 0
        chunk_id = 0
        active_jobs = []
        max_parallel = min(cpu_count(), 2)
        global_doc_offset = 0

        with Pool(processes=max_parallel, maxtasksperchild=1) as pool:
            with open(json_path, encoding="utf-8") as f:
                chunk_docs = []
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        doc = json.loads(line)
                        chunk_docs.append(doc)
                    except json.JSONDecodeError as e:
                        print(f"[SPIMI] JSON decode error: {e}")
                        continue
                    if len(chunk_docs) >= chunk_size:
                        # submit to worker
                        job = pool.apply_async(
                            process_chunk,
                            (
                                chunk_id,
                                chunk_docs,
                                self.output_dir,
                                getattr(self.tokenizer, "config", {}),
                                global_doc_offset,
                            ),
                        )
                        global_doc_offset += len(chunk_docs)
                        active_jobs.append(job)
                        chunk_docs = []
                        chunk_id += 1

                        # limit number of concurrent jobs to keep memory safe
                        if len(active_jobs) >= max_parallel:
                            for j in active_jobs:
                                docs_processed, tokens = j.get()
                                num_docs += docs_processed
                                total_tokens += tokens
                            active_jobs.clear()

                        # monitor memory
                        if psutil.Process(os.getpid()).memory_info().rss > self.memory_limit * 0.8:
                            print("[SPIMI] Waiting for workers to free memory...")
                            for j in active_jobs:
                                docs_processed, tokens = j.get()
                                num_docs += docs_processed
                                total_tokens += tokens
                            active_jobs.clear()

                # last partial chunk
                if chunk_docs:
                    job = pool.apply_async(
                        process_chunk,
                        (
                            chunk_id,
                            chunk_docs,
                            self.output_dir,
                            getattr(self.tokenizer, "config", {}),
                            global_doc_offset,
                        ),
                    )
                    global_doc_offset += len(chunk_docs)
                    active_jobs.append(job)
                    chunk_id += 1

                # collect remaining jobs
                for j in active_jobs:
                    docs_processed, tokens = j.get()
                    num_docs += docs_processed
                    total_tokens += tokens

            pool.close()
            pool.join()

        # metadata
        avg_len = total_tokens / max(num_docs, 1)
        meta_path = os.path.join(self.output_dir, "metadata.json")
        metadata = {"num_docs": num_docs, "avg_doc_len": avg_len, "num_blocks": chunk_id}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(
            f"[SPIMI] Done — {num_docs:,} docs, {chunk_id} blocks created (avg_len={avg_len:.2f})"
        )

    def merge_blocks(self, output_path: str = "data/index_final.jsonl", min_df: int = 3):
        block_files = sorted(
            [f for f in os.listdir(self.output_dir) if f.startswith("block_") and f.endswith(".gz")]
        )
        if not block_files:
            print("[SPIMI] No blocks to merge.")
            return

        print(f"[SPIMI] Merging {len(block_files)} blocks...")

        def block_stream(path):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for term, postings in ijson.kvitems(f, ""):
                    yield term, postings

        streams = [block_stream(os.path.join(self.output_dir, f)) for f in block_files]
        merged = merge(*streams, key=lambda x: x[0])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out:
            term = None
            postings = defaultdict(list)
            buffer, buffer_size = [], 2000

            def flush_buffer():
                if not buffer:
                    return
                out.write("\n".join(buffer) + "\n")
                buffer.clear()

            for t, p in merged:     # for each term-postings pair
                if term and t != term:
                    if len(postings) >= min_df:
                        buffer.append(json.dumps({term: dict(postings)}, ensure_ascii=False))
                        if len(buffer) >= buffer_size:
                            flush_buffer()
                    postings.clear()
                term = t
                for d, pos in p.items():
                    postings[int(d)] = postings.get(int(d), []) + pos
                print(f"[SPIMI] Merging term: {term}", end="\r")

            if term and len(postings) >= min_df:
                buffer.append(json.dumps({term: dict(postings)}, ensure_ascii=False))
            flush_buffer()

        print(f"[SPIMI] Done → {output_path}")
