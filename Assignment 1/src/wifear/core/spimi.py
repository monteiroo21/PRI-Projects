import json
import os
from collections import defaultdict

import psutil

from wifear.core.tokenizer import PortugueseTokenizer


class SPIMIIndexer:
    def __init__(
        self, tokenizer: PortugueseTokenizer, output_dir="index_blocks", memory_limit_mb: int = 2000
    ):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.memory_limit = memory_limit_mb * 1024 * 1024
        os.makedirs(output_dir, exist_ok=True)

    def _memory_full(self) -> bool:
        """Return True if RSS memory usage exceeds the limit."""
        return psutil.Process(os.getpid()).memory_info().rss > self.memory_limit

    def index_documents(self, json_path: str):
        """Build blocks from cleaned documents (ptwiki_clean.json)."""
        with open(json_path, encoding="utf-8") as f:
            docs = json.load(f)

        block_id = 0
        index = defaultdict(lambda: defaultdict(list))

        for doc_id, doc in enumerate(docs):
            text = doc.get("text", "")
            tokens = self.tokenizer.tokenize(text)
            for pos, term in enumerate(tokens):
                index[term][doc_id].append(pos)

            if self._memory_full():
                self._write_block(index, block_id)
                index.clear()
                block_id += 1

        if index:
            self._write_block(index, block_id)

    def _write_block(self, index, block_id: int):
        """Write one block’s dictionary to disk."""
        block_path = os.path.join(self.output_dir, f"block_{block_id:03d}.json")
        with open(block_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=4)
        print(f"[SPIMI] Block {block_id} written to {block_path}")

    def merge_blocks(self, output_path="data/index_final.json", min_df: int = 3):
        """Merge all block files into a single index."""
        final_index = defaultdict(lambda: defaultdict(list))

        for fname in sorted(os.listdir(self.output_dir)):
            with open(os.path.join(self.output_dir, fname), encoding="utf-8") as f:
                block = json.load(f)
            for term, postings in block.items():
                for doc_id, positions in postings.items():
                    final_index[term][doc_id].extend(positions)

        final_index = {t: p for t, p in final_index.items() if len(p) >= min_df}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_index, f, ensure_ascii=False, indent=4)
        print(f"[SPIMI] Final merged index saved to {output_path}")
