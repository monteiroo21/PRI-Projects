# searcher.py
from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from typing import Dict, List, Tuple

from wifear.core.tokenizer import PortugueseTokenizer


class SearchEngine:
    """
    Search engine that loads an inverted positional index from SQLite (index.db)
    created by load_db.py, and performs BM25 ranking and relevance feedback.
    """

    def __init__(
        self,
        db_path: str,
        tokenizer: PortugueseTokenizer,
        metadata_path: str | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        """
        Initialize the search engine by connecting to the SQLite database.

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

        # Connect to SQLite database
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()

        # Load metadata if available
        meta = self._load_metadata(metadata_path)
        self.N = meta.get("num_docs") or self._infer_num_docs()
        self.doc_len = self._compute_doc_lengths()
        self.avg_doc_len = meta.get("avg_doc_len") or (
            sum(self.doc_len.values()) / max(len(self.doc_len), 1)
        )

        # Precompute IDF for all terms
        self.idf: Dict[str, float] = {}
        self._precompute_idf()

        print(f"[INFO] Search engine connected to {db_path}")
        print(f"[INFO] Documents: {self.N}, Avg. length: {self.avg_doc_len:.2f}")

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

    def _get_postings(self, term: str) -> Dict[int, List[int]]:
        """
        Retrieve the postings list (doc_id → positions) for a given term
        directly from the SQLite database.
        """
        self.cur.execute("SELECT postings FROM inverted_index WHERE term = ?", (term,))
        row = self.cur.fetchone()
        if not row:
            return {}
        try:
            return {int(doc): pos for doc, pos in json.loads(row[0]).items()}
        except Exception:
            return {}

    def _infer_num_docs(self) -> int:
        """
        Infer the total number of distinct documents in the index.
        This iterates over all postings once.
        """
        print("[INFO] Counting total number of documents...")
        doc_ids = set()
        self.cur.execute("SELECT postings FROM inverted_index")
        for (p,) in self.cur.fetchall():
            try:
                postings = json.loads(p)
                doc_ids.update(map(int, postings.keys()))
            except Exception:
                continue
        print(f"[INFO] Found {len(doc_ids)} distinct documents.")
        return len(doc_ids)

    def _compute_doc_lengths(self) -> Dict[int, int]:
        """
        Compute document lengths based on total term occurrences.
        Used for BM25 normalization.
        """
        print("[INFO] Computing document lengths...")
        dl = defaultdict(int)
        self.cur.execute("SELECT postings FROM inverted_index")
        for (p,) in self.cur.fetchall():
            try:
                postings = json.loads(p)
                for doc_id, positions in postings.items():
                    dl[int(doc_id)] += len(positions)
            except Exception:
                continue
        print(f"[INFO] Computed lengths for {len(dl)} documents.")
        return dict(dl)

    def _precompute_idf(self):
        """
        Precompute IDF (Inverse Document Frequency) values for all terms
        to speed up BM25 calculations.
        """
        print("[INFO] Precomputing IDF values...")
        self.cur.execute("SELECT term, postings FROM inverted_index")
        for term, p in self.cur.fetchall():
            try:
                postings = json.loads(p)
                df = len(postings)
                self.idf[term] = self._bm25_idf(self.N, df)
            except Exception:
                continue
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
        """Compute the BM25 contribution of a single term for one document."""
        postings = self._get_postings(term)
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
    # Query processing
    # -------------------------------------------------------------------------

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Execute a BM25-ranked search query.

        Steps:
            1. Tokenize the input query.
            2. Collect all candidate documents containing any query term.
            3. Compute BM25 scores for each document.
            4. Return top-k ranked results.
        """
        terms = self.tokenizer.tokenize(query_text)
        if not terms:
            return []

        candidate_docs = set()
        for t in terms:
            postings = self._get_postings(t)
            if postings:
                candidate_docs.update(postings.keys())

        scores: Dict[int, float] = {}
        for doc_id in candidate_docs:
            score = 0.0
            for t in terms:
                score += self._bm25_doc_term(t, doc_id)
            if score > 0:
                scores[doc_id] = score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # -------------------------------------------------------------------------
    # Relevance feedback
    # -------------------------------------------------------------------------

    def like_document(
        self, doc_id: int, top_k: int = 10, expand_terms: int = 20, alpha: float = 1.0
    ) -> List[Tuple[int, float]]:
        """
        Retrieve documents similar to a given document using pseudo relevance feedback.
        Optimized version that queries only relevant terms from SQLite.
        """
        tf_doc: Dict[str, int] = {}
        doc_str = str(doc_id)

        # Query only postings that contain this doc_id (instead of scanning all)
        self.cur.execute("SELECT term, postings FROM inverted_index")
        for term, p in self.cur.fetchall():
            if doc_str in p:  # quick substring check avoids full json.loads for most terms
                try:
                    postings = json.loads(p)
                    if doc_str in postings:
                        tf_doc[term] = len(postings[doc_str])
                except Exception:
                    continue

        if not tf_doc:
            print(f"[WARN] Document {doc_id} not found in index.")
            return []

        # Select top weighted terms
        scored_terms = sorted(
            ((t, tf * self.idf.get(t, 0.0)) for t, tf in tf_doc.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:expand_terms]

        # Build pseudo query
        pseudo_query_terms = []
        for t, w in scored_terms:
            reps = max(1, int(alpha * max(1.0, w) ** 0.5))
            pseudo_query_terms.extend([t] * reps)

        # Collect candidate documents
        candidate_docs = set()
        for t, _ in scored_terms:
            postings = self._get_postings(t)
            if postings:
                candidate_docs.update(postings.keys())
        candidate_docs.discard(doc_id)

        # Compute similarity scores
        scores: Dict[int, float] = {}
        for d in candidate_docs:
            s = 0.0
            for t in pseudo_query_terms:
                s += self._bm25_doc_term(t, d)
            if s > 0:
                scores[d] = s

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    
    def close(self):
        """Close the SQLite connection."""
        if self.conn:
            self.conn.close()
            print("[INFO] SQLite connection closed.")


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------

if __name__ == "__main__":
    tokenizer = PortugueseTokenizer(min_len=3)
    engine = SearchEngine(
        db_path="index.db",  # SQLite file generated by load_db.py
        tokenizer=tokenizer,
        metadata_path="index_blocks/metadata.json",
        k1=1.2,
        b=0.75,
    )

    query_text = "freguesia Amares"
    results = engine.query(query_text, top_k=10)
    print("\n Search Results:")
    for doc, score in results:
        print(f"  Doc {doc}: {score:.4f}")

    similar_docs = engine.like_document(doc_id=1234, top_k=5)
    print("\n Documents similar to 1234:")
    for doc, score in similar_docs:
        print(f"  Doc {doc}: {score:.4f}")

    engine.close()
