# searcher.py
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple

from wifear.core.tokenizer import PortugueseTokenizer


class SearchEngine:
    """
    Search engine that loads an inverted positional index and performs
    BM25 ranking for user queries. Also supports relevance feedback
    by retrieving documents similar to a given document.
    """

    def __init__(
        self,
        index_path: str,
        tokenizer: PortugueseTokenizer,
        metadata_path: str | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        """
        Initialize the search engine.

        Args:
            index_path: Path to the final JSON index file.
            tokenizer: Tokenizer used to preprocess both index and queries.
            metadata_path: Optional path to metadata file (e.g., metadata.json).
            k1, b: BM25 hyperparameters.
        """
        self.tokenizer = tokenizer
        self.index: Dict[str, Dict[int, List[int]]] = self._load_index(index_path)
        self.k1 = k1
        self.b = b

        meta = self._load_metadata(metadata_path)
        self.N = meta.get("num_docs") or self._infer_num_docs()
        self.doc_len: Dict[int, int] = self._compute_doc_lengths()
        self.avg_doc_len = meta.get("avg_doc_len") or (
            sum(self.doc_len.values()) / max(len(self.doc_len), 1)
        )

        # Precompute df and idf for faster BM25 scoring
        self.df: Dict[str, int] = {t: len(p) for t, p in self.index.items()}
        self.idf: Dict[str, float] = {
            t: self._bm25_idf(self.N, self.df[t]) for t in self.index.keys()
        }

    # -------------------------------------------------------------------------
    # Loading utilities
    # -------------------------------------------------------------------------

    def _load_index(self, path: str) -> Dict[str, Dict[int, List[int]]]:
        """Load the positional inverted index from disk."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Ensure doc IDs are integers
        for term, postings in data.items():
            data[term] = {int(d): pos for d, pos in postings.items()}
        return data

    def _load_metadata(self, meta_path: str | None) -> Dict:
        """Load metadata if available (num_docs, avg_doc_len, tokenizer config, etc.)."""
        if not meta_path:
            return {}
        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _infer_num_docs(self) -> int:
        """Infer total number of documents from the index itself."""
        doc_ids = set()
        for postings in self.index.values():
            doc_ids.update(postings.keys())
        return len(doc_ids)

    def _compute_doc_lengths(self) -> Dict[int, int]:
        """Compute document lengths based on total term occurrences."""
        dl = defaultdict(int)
        for postings in self.index.values():
            for doc_id, positions in postings.items():
                dl[doc_id] += len(positions)
        return dict(dl)

    # -------------------------------------------------------------------------
    # BM25 scoring
    # -------------------------------------------------------------------------

    @staticmethod
    def _bm25_idf(N: int, df: int) -> float:
        """Compute the BM25 inverse document frequency (IDF) with standard smoothing."""
        if df <= 0 or N <= 0:
            return 0.0
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25_doc_term(self, term: str, doc_id: int) -> float:
        """Compute the BM25 contribution of one term for a single document."""
        postings = self.index.get(term)
        if not postings:
            return 0.0
        tf = len(postings.get(doc_id, []))  # term frequency in document
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
        Execute a text query using BM25 ranking.

        Steps:
            1. Tokenize the query using the same tokenizer as the index.
            2. Collect all candidate documents that contain at least one query term.
            3. Compute BM25 scores for each candidate document.
            4. Return the top-k ranked results.

        Args:
            query_text: The user's input text query.
            top_k: Number of top results to return.
        """
        terms = self.tokenizer.tokenize(query_text)
        if not terms:
            return []

        # Collect all documents that contain at least one query term
        candidate_docs = set()
        for t in terms:
            postings = self.index.get(t)
            if postings:
                candidate_docs.update(postings.keys())

        if not candidate_docs:
            return []

        # Compute BM25 scores for each candidate
        scores: Dict[int, float] = {}
        for doc_id in candidate_docs:
            score = 0.0
            for t in terms:
                score += self._bm25_doc_term(t, doc_id)
            if score > 0:
                scores[doc_id] = score

        # Return top-k ranked results
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # -------------------------------------------------------------------------
    # Relevance feedback: “Find documents similar to this one”
    # -------------------------------------------------------------------------

    def like_document(
        self,
        doc_id: int,
        top_k: int = 10,
        expand_terms: int = 20,
        alpha: float = 1.0,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve documents similar to a given document.
        This implements a simple relevance feedback mechanism.

        Steps:
            1. Extract the top terms of the document based on tf-idf weight.
            2. Use them as a pseudo-query (weighted by tf-idf).
            3. Rank other documents with BM25.

        Args:
            doc_id: ID of the reference document.
            top_k: Number of results to return.
            expand_terms: Number of top terms to include in the pseudo-query.
            alpha: Scaling factor controlling term repetition in the pseudo-query.
        """
        # 1) Collect term frequencies for the given document
        tf_doc: Dict[str, int] = {}
        for term, postings in self.index.items():
            if doc_id in postings:
                tf_doc[term] = len(postings[doc_id])

        if not tf_doc:
            return []

        # 2) Rank terms by tf-idf and keep top 'expand_terms'
        scored_terms = sorted(
            ((t, tf * self.idf.get(t, 0.0)) for t, tf in tf_doc.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:expand_terms]

        # 3) Build a pseudo-query with repeated terms proportional to weight
        pseudo_query_terms: List[str] = []
        for t, w in scored_terms:
            reps = max(1, int(alpha * max(1.0, w) ** 0.5))
            pseudo_query_terms.extend([t] * reps)

        # 4) Collect candidate documents
        candidate_docs = set()
        for t, _ in scored_terms:
            postings = self.index.get(t)
            if postings:
                candidate_docs.update(postings.keys())
        candidate_docs.discard(doc_id)

        # 5) Compute BM25 scores for similar docs
        scores: Dict[int, float] = {}
        for d in candidate_docs:
            s = 0.0
            for t in pseudo_query_terms:
                s += self._bm25_doc_term(t, d)
            if s > 0:
                scores[d] = s

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------

if __name__ == "__main__":
    tokenizer = PortugueseTokenizer(min_len=3)
    engine = SearchEngine(
        index_path="data/index_final.json",
        tokenizer=tokenizer,
        metadata_path="index_blocks/metadata.json",  # Adjust this path as needed
        k1=1.2,
        b=0.75,
    )

    query_text = "freguesia Amares"
    results = engine.query(query_text, top_k=10)
    print("Search Results:", results)

    # Example of relevance feedback
    similar_docs = engine.like_document(doc_id=1234, top_k=10)
    print("Documents similar to 1234:", similar_docs)
