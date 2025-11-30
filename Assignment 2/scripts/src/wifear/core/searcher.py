from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from typing import Dict, List, Tuple
import os

import numpy as np
from wifear.core.tokenizer import PortugueseTokenizer


class SearchEngine:
    """
    Optimized Search Engine that loads an inverted positional index entirely
    from SQLite (index.db) into memory and performs BM25 ranking and
    relevance feedback efficiently.
    """

    def __init__(
        self,
        db_path: str,
        tokenizer: PortugueseTokenizer,
        metadata_path: str | None = None,
        docstore_path: str | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        """
        Initialize the search engine by connecting to the SQLite database,
        loading all postings into memory, and precomputing statistics.

        Args:
            db_path: Path to the SQLite database created by load_db.py.
            tokenizer: Tokenizer used for both index and queries.
            metadata_path: Optional path to metadata file (e.g., metadata.json).
            k1, b: BM25 hyperparameters.
        """
        self.tokenizer = tokenizer
        self.db_path = db_path
        self.k1 = k1
        self.b = b

        print(f"[INFO] Connecting to SQLite index at {db_path}...")
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()

        # Load metadata if available
        meta = self._load_metadata(metadata_path)

        # Load entire index into memory
        print("[INFO] Loading entire inverted index into memory...")
        self.index: Dict[str, Dict[int, List[int]]] = {}
        self.cur.execute("SELECT term, postings FROM inverted_index")
        # for term, postings_json in self.cur.fetchall():
        #     try:
        #         self.index[term] = json.loads(postings_json)
        #     except Exception:
        #         continue
        for term, postings_json in self.cur.fetchall():
            try:
                postings = json.loads(postings_json)
                # Convert string doc_ids to integers
                self.index[term] = {int(doc_id): positions for doc_id, positions in postings.items()}
            except Exception as e:
                print(f"[WARN] Failed to load term '{term}': {e}")
                continue
        print(f"[INFO] Loaded {len(self.index):,} terms into memory.")

        # Build forward (direct) index for fast document access
        print("[INFO] Building forward index (document-term map)...")
        self.forward_index: Dict[int, Dict[str, int]] = defaultdict(dict)
        for term, postings in self.index.items():
            for doc_id, positions in postings.items():
                self.forward_index[doc_id][term] = len(positions)
        print(f"[INFO] Built forward index for {len(self.forward_index):,} documents.")

        # Close DB connection (not needed anymore)
        self.conn.close()
        self.conn = None
        self.cur = None

        # Compute global stats
        self.N = meta.get("num_docs") or self._infer_num_docs()
        self.doc_len = self._compute_doc_lengths()
        self.avg_doc_len = meta.get("avg_doc_len") or (
            sum(self.doc_len.values()) / max(len(self.doc_len), 1)
        )

        # Precompute IDF
        self.idf: Dict[str, float] = {}
        self._precompute_idf()

        # Load docstore metadata
        self.docstore = {}
        if docstore_path and os.path.exists(docstore_path):
            with open(docstore_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        meta = json.loads(line)
                        self.docstore[int(meta["id"])] = meta
                    except Exception:
                        continue
            print(f"[INFO] Loaded {len(self.docstore):,} document metadata entries.")
        else:
            print("[WARN] No docstore found. Titles/descriptions will be placeholders.")

        print(f"[INFO] Documents: {self.N}, Avg. length: {self.avg_doc_len:.2f}")
        print("[INFO] Search engine ready for real-time queries.")

    # -------------------------------------------------------------------------
    # Internal utilities
    # -------------------------------------------------------------------------

    def _load_metadata(self, meta_path: str | None) -> Dict:
        """Load optional metadata (if provided)."""
        if not meta_path:
            return {}
        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _infer_num_docs(self) -> int:
        """Infer number of unique documents in memory."""
        print("[INFO] Counting total number of documents (in memory)...")
        doc_ids = set()
        for postings in self.index.values():
            doc_ids.update(postings.keys())
        print(f"[INFO] Found {len(doc_ids)} distinct documents.")
        return len(doc_ids)

    def _compute_doc_lengths(self) -> Dict[int, int]:
        """Compute document lengths from loaded postings."""
        print("[INFO] Computing document lengths...")
        dl = defaultdict(int)
        for postings in self.index.values():
            for doc_id, positions in postings.items():
                dl[int(doc_id)] += len(positions)
        print(f"[INFO] Computed lengths for {len(dl)} documents.")
        return dict(dl)

    def _precompute_idf(self):
        """Compute IDF (Inverse Document Frequency) for all terms."""
        print("[INFO] Precomputing IDF values...")
        for term, postings in self.index.items():
            df = len(postings)
            self.idf[term] = self._bm25_idf(self.N, df)
        print(f"[INFO] IDF computed for {len(self.idf)} terms.")

    # -------------------------------------------------------------------------
    # BM25 core functions
    # -------------------------------------------------------------------------

    @staticmethod
    def _bm25_idf(N: int, df: int) -> float:
        """Compute the BM25 Inverse Document Frequency with smoothing."""
        if df <= 0 or N <= 0:
            return 0.0
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25_doc_term(self, term: str, doc_id: int) -> float:
        """Compute BM25 contribution for one term in one document."""
        postings = self.index.get(term)
        if not postings:
            return 0.0
        tf = len(postings.get(doc_id, []))
        if tf == 0:
            return 0.0
        dl = self.doc_len.get(doc_id, 1)
        idf = self.idf.get(term, 0.0)
        denom = tf + self.k1 * (1 - self.b + self.b * (dl / (self.avg_doc_len or 1)))
        return idf * (tf * (self.k1 + 1)) / denom

    # -------------------------------------------------------------------------
    # Query processing (optimized)
    # -------------------------------------------------------------------------

    def _bm25(self, query_text: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Compute BM25 ranking scores for all matching documents."""
        terms = self.tokenizer.tokenize(query_text)
        if not terms:
            return []

        scores: Dict[int, float] = {}
        for t in terms:
            postings = self.index.get(t)
            if not postings:
                continue
            idf = self.idf.get(t, 0.0)
            for doc_id, positions in postings.items():
                tf = len(positions)
                dl = self.doc_len.get(doc_id, 1)
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / (self.avg_doc_len or 1)))
                score = idf * (tf * (self.k1 + 1)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        if not scores:
            return []

        doc_ids = np.fromiter(scores.keys(), dtype=int)
        doc_scores = np.fromiter(scores.values(), dtype=float)
        top_idx = np.argsort(-doc_scores)[:top_k]

        return [(int(doc_ids[i]), float(doc_scores[i])) for i in top_idx]


    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        """Return documents with metadata (title + snippet) sorted by BM25 score."""
        ranked = self._bm25(query_text, top_k)
        results = []

        for doc_id, score in ranked:
            meta = self.docstore.get(doc_id, {})
            results.append({
                "id": doc_id,
                "title": meta.get("title", f"Doc {doc_id}"),
                "description": meta.get("description", ""),
                "score": score,
            })

        return results

    # -------------------------------------------------------------------------
    # Relevance feedback (optimized)
    # -------------------------------------------------------------------------

    def like_document(
        self, doc_id: int, top_k: int = 10, expand_terms: int = 20, alpha: float = 1.0
    ) -> List[Tuple[int, float]]:
        """
        Retrieve documents similar to a given document using pseudo relevance feedback.
        Fully in-memory version.
        """
        # tf_doc: Dict[str, int] = {}
        # for term, postings in self.index.items():
        #     if doc_id in postings:
        #         tf_doc[term] = len(postings[doc_id])
        tf_doc = self.forward_index.get(doc_id, {})
        if not tf_doc:
            print(f"[WARN] Document {doc_id} not found in forward index.")
            return []

        if not tf_doc:
            print(f"[WARN] Document {doc_id} not found in index.")
            return []

        # Select top-weighted terms
        scored_terms = sorted(
            ((t, tf * self.idf.get(t, 0.0)) for t, tf in tf_doc.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:expand_terms]

        # Build pseudo-query
        pseudo_query_terms = []
        for t, w in scored_terms:
            reps = max(1, int(alpha * max(1.0, w) ** 0.5))
            pseudo_query_terms.extend([t] * reps)

        # Compute similarity scores
        scores: Dict[int, float] = {}
        for t in pseudo_query_terms:
            postings = self.index.get(t)
            if not postings:
                continue
            idf = self.idf.get(t, 0.0)
            for d, positions in postings.items():
                if d == doc_id:
                    continue
                tf = len(positions)
                dl = self.doc_len.get(d, 1)
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / (self.avg_doc_len or 1)))
                score = idf * (tf * (self.k1 + 1)) / denom
                scores[d] = scores.get(d, 0.0) + score

        if not scores:
            return []

        doc_ids = np.fromiter(scores.keys(), dtype=int)
        doc_scores = np.fromiter(scores.values(), dtype=float)
        top_idx = np.argsort(-doc_scores)[:top_k]

        return [(int(doc_ids[i]), float(doc_scores[i])) for i in top_idx]

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def close(self):
        """Compatibility method (DB already closed)."""
        print("[INFO] Search engine closed (in-memory).")


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------

if __name__ == "__main__":
    tokenizer = PortugueseTokenizer(min_len=3)
    engine = SearchEngine(
        db_path="index.db",
        tokenizer=tokenizer,
        metadata_path="index_blocks/metadata.json",
        docstore_path="data/docstore.jsonl",
        k1=1.2,
        b=0.75,
    )

    query_text = "freguesia Amares"
    results = engine.query(query_text, top_k=10)
    print("\n Search Results:")
    for doc in results:
        print(f"  {doc['id']}: {doc['title']} → {doc['score']:.4f}")

    if results:
        doc_id = results[0]["id"]
        print(f"\n[TEST] Using doc_id={doc_id} ({results[0]['title']}) for similarity search\n")

        similar_docs = engine.like_document(doc_id, top_k=10)
        if not similar_docs:
            print(f"[WARN] No similar documents found for ID {doc_id}")
        else:
            print(f"[INFO] Found {len(similar_docs)} similar documents for {doc_id}:\n")
            for d_id, score in similar_docs:
                meta = engine.docstore.get(d_id, {})
                print(f"  {d_id}: {meta.get('title', 'Untitled')} → {score:.4f}")

    else:
        print("[ERROR] No documents returned by the base query.")

    engine.close()
