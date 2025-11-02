import gzip
import json
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List

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
        json.dump(index, f, ensure_ascii=False, separators=(",", ":"))

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
        return rss > self.memory_limit * 0.95  # flush before hitting hard limit

    def index_documents(self, json_path: str, chunk_size: int = 10000):
        """Parallel SPIMI indexing with bounded memory usage."""
        os.makedirs(self.output_dir, exist_ok=True)

        num_docs = 0
        total_tokens = 0
        chunk_id = 0
        active_jobs = []
        max_parallel = min(cpu_count(), 4)
        global_doc_offset = 0

        with Pool(processes=max_parallel) as pool:
            with open(json_path, encoding="utf-8") as f:
                chunk_docs = []
                for doc in ijson.items(f, "item"):
                    chunk_docs.append(doc)
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

    def merge_blocks(self, output_path: str = "data/index_final.json", min_df: int = 3):
        """Merge compressed SPIMI blocks into a single final inverted index.

        Uses streaming reads to avoid loading full blocks into memory.
        Flushes frequently to stay under the memory limit.
        """
        final_index: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        block_files = sorted(
            [f for f in os.listdir(self.output_dir) if f.startswith("block_") and f.endswith(".gz")]
        )

        print(f"[SPIMI] Merging {len(block_files)} blocks with min_df={min_df} (streaming mode)...")

        # Remove any previous final index
        if os.path.exists(output_path):
            os.remove(output_path)

        for block_id, block_file in enumerate(block_files):
            block_path = os.path.join(self.output_dir, block_file)
            print(f"[SPIMI] Streaming block {block_id}: {block_file}")

            with gzip.open(block_path, "rt", encoding="utf-8") as f:
                # Iterate key/value pairs at the root level
                for term, postings in ijson.kvitems(f, ""):
                    # Merge postings
                    for doc_id, positions in postings.items():
                        final_index[term][int(doc_id)].extend(positions)

                    # Periodic flush to prevent memory overload
                    if len(final_index) >= 50_000 or self._memory_full():
                        self._flush_partial(final_index, output_path, min_df, append=True)
                        final_index.clear()

        # Write any remaining terms
        if final_index:
            self._flush_partial(final_index, output_path, min_df, append=True)

        print(f"[SPIMI] Merge completed → {output_path}")

    def _flush_partial(
        self,
        partial_index: Dict[str, Dict[int, List[int]]],
        output_path: str,
        min_df: int,
        append: bool = False,
    ):
        """Write a partial merged index incrementally in plain JSON."""
        filtered = {t: p for t, p in partial_index.items() if len(p) >= min_df}
        mode = "a" if append else "w"

        # If appending, we ensure proper JSON array-like structure
        if append and os.path.exists(output_path):
            with open(output_path, mode, encoding="utf-8") as f:
                # Remove the last '}' to allow concatenation
                f.seek(f.tell() - 1)
                f.write(",")
                json.dump(filtered, f, ensure_ascii=False, indent=2)
                f.write("}")
        else:
            with open(output_path, mode, encoding="utf-8") as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2)

        print(f"[SPIMI] Partial merge flush ({len(filtered):,} terms) → {output_path}")
