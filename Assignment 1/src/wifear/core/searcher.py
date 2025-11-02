import json
import math
from collections import defaultdict
from typing import List, Optional, Tuple

from wifear.core.tokenizer import PortugueseTokenizer


class BM25Ranker:
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        tokenizer_config: Optional[dict] = None,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b

        self.tokenizer = PortugueseTokenizer(**(tokenizer_config or {}))

        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.N = int(meta.get("num_docs", 0))
        self.avgdl = float(meta.get("avg_doc_len", 1.0))

        self.index = {}
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip().rstrip(",")
                if not line or line in ["{", "}", "[", "]"]:
                    continue
                try:
                    data = json.loads(line)
                    self.index.update(data)
                except json.JSONDecodeError:
                    continue

        # calcula comprimento dos documentos (dl)
        self.doc_len = defaultdict(int)
        for postings in self.index.values():
            for d, positions in postings.items():
                self.doc_len[int(d)] += len(positions)

        total_len = sum(self.doc_len.values())
        if not self.avgdl:
            self.avgdl = total_len / max(self.N, 1)

    def _idf(self, n: int) -> float:
        N = max(self.N, 1)
        n = max(1, min(n, N))
        return math.log((N - n + 0.5) / (n + 0.5) + 1.0)

    def _bm25(self, tf: int, dl: int, idf: float) -> float:
        denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
        return idf * (tf * (self.k1 + 1)) / denom if denom != 0 else 0

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        terms = self.tokenizer.tokenize(query)
        scores = defaultdict(float)

        for t in terms:
            postings = self.index.get(t)
            if not postings:
                continue
            n = len(postings)
            idf = self._idf(n)
            for d_str, positions in postings.items():
                d = int(d_str)
                tf = len(positions)
                dl = self.doc_len[d]
                scores[d] += self._bm25(tf, dl, idf)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


if __name__ == "__main__":
    index_path = "data/index_final.json"
    metadata_path = "index_blocks/metadata.json"
    docs_path = "data/ptwiki_clean.json"  # <--- ficheiro original que indexaste

    print("=== Searcher ===")
    query = input("Escreve a tua query: ").strip()

    bm25 = BM25Ranker(
        index_path=index_path,
        metadata_path=metadata_path,
        tokenizer_config={"min_len": 2},
    )

    print("[INFO] A carregar documentos originais...")
    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)

    results = bm25.search(query, top_k=5)

    print("\nTop resultados:")
    for rank, (doc_id, score) in enumerate(results, 1):
        doc = docs[doc_id]
        text = doc.get("text", "")
        title = doc.get("title", f"Documento {doc_id}")
        snippet = text[:300].replace("\n", " ") + ("..." if len(text) > 300 else "")
        print(f"\n{rank}. [{title}] (DocID={doc_id}, Score={score:.4f})")
        print(f"→ {snippet}")
