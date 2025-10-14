import os
import gzip
import json
import psutil
import ijson
from collections import defaultdict
from typing import Dict, List

from wifear.core.tokenizer import PortugueseTokenizer


class SPIMIIndexer:
    """
    Efficient Single-Pass In-Memory Indexer (SPIMI) for large-scale corpora.

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

    # ------------------------------------------------------------
    # Memory Control
    # ------------------------------------------------------------
    def _memory_full(self) -> bool:
        """Return True if current memory usage exceeds threshold."""
        rss = psutil.Process(os.getpid()).memory_info().rss
        return rss > self.memory_limit * 0.95  # flush before hitting hard limit

    # ------------------------------------------------------------
    # Indexing Step
    # ------------------------------------------------------------
    def index_documents(self, json_path: str):
        """
        Incrementally read and index documents from a large JSON array file.
        Writes SPIMI blocks (.gz) whenever memory approaches limit.
        """
        index: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        docmap = {}
        block_id = 0
        num_docs = 0
        total_tokens = 0

        print(f"[SPIMI] Indexing from {json_path} (streaming mode)...")

        with open(json_path, "r", encoding="utf-8") as f:
            # ijson parses JSON arrays incrementally
            for doc in ijson.items(f, "item"):
                doc_id = num_docs
                num_docs += 1
                title = doc.get("title", f"doc_{doc_id}")
                text = doc.get("text", "")
                tokens = self.tokenizer.tokenize(text)
                total_tokens += len(tokens)

                docmap[doc_id] = title

                for pos, term in enumerate(tokens):
                    index[term][doc_id].append(pos)

                # Prevent memory overflow
                if len(index) > 250_000 or self._memory_full():
                    self._write_block(index, block_id)
                    index.clear()
                    block_id += 1

        # Write remaining terms
        if index:
            self._write_block(index, block_id)

        # Save document mapping and metadata
        self._save_metadata(docmap, num_docs, total_tokens)

        print(f"[SPIMI] Completed: {num_docs:,} documents processed into {block_id + 1} block(s).")

    # ------------------------------------------------------------
    # Block Writing
    # ------------------------------------------------------------
    def _write_block(self, index: Dict[str, Dict[int, List[int]]], block_id: int):
        """Write a compressed block to disk (gzip)."""
        block_path = os.path.join(self.output_dir, f"block_{block_id:03d}.json.gz")
        with gzip.open(block_path, "wt", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, separators=(",", ":"))
        print(f"[SPIMI] Block {block_id} written ({len(index):,} terms) → {block_path}")

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------
    def _save_metadata(self, docmap: Dict[int, str], num_docs: int, total_tokens: int):
        """Save document map and general metadata."""
        docmap_path = os.path.join(self.output_dir, "docmap.json.gz")
        with gzip.open(docmap_path, "wt", encoding="utf-8") as f:
            json.dump(docmap, f, ensure_ascii=False)

        metadata = {
            "num_docs": num_docs,
            "avg_doc_len": total_tokens / max(num_docs, 1),
            "tokenizer_config": getattr(self.tokenizer, "config", {}),
        }
        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(f"[SPIMI] Metadata saved ({num_docs:,} docs, avg_len={metadata['avg_doc_len']:.2f})")

    # ------------------------------------------------------------
    # Merge Step
    # ------------------------------------------------------------
    def merge_blocks(self, output_path: str = "data/index_final.json.gz", min_df: int = 3):
        """
        Merge compressed SPIMI blocks into a single final inverted index.
        Filters out terms with document frequency < min_df.
        """
        final_index: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        block_files = sorted(
            [f for f in os.listdir(self.output_dir) if f.startswith("block_") and f.endswith(".gz")]
        )

        print(f"[SPIMI] Merging {len(block_files)} blocks with min_df={min_df}...")

        for block_file in block_files:
            block_path = os.path.join(self.output_dir, block_file)
            with gzip.open(block_path, "rt", encoding="utf-8") as f:
                block = json.load(f)
            for term, postings in block.items():
                for doc_id, positions in postings.items():
                    final_index[term][int(doc_id)].extend(positions)

            # Flush periodically to keep memory safe
            if len(final_index) > 250_000 or self._memory_full():
                self._flush_partial(final_index, output_path, append=True)
                final_index.clear()

        if final_index:
            self._flush_partial(final_index, output_path, append=True)

        print(f"[SPIMI] Merge completed → {output_path}")

    def _flush_partial(self, partial_index: Dict[str, Dict[int, List[int]]], output_path: str, append=False):
        """Write a partial merged index incrementally."""
        mode = "at" if append else "wt"
        filtered = {t: p for t, p in partial_index.items() if len(p) >= 3}
        with gzip.open(output_path, mode, encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, separators=(",", ":"))
        print(f"[SPIMI] Partial merge flush ({len(filtered):,} terms) → {output_path}")
