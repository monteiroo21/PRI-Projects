from __future__ import annotations

import json
import math
import sqlite3
import time
import os
from typing import Dict, List, Tuple
from functools import lru_cache
import torch

from sentence_transformers import CrossEncoder
from wifear.core.tokenizer import PortugueseTokenizer

class SearchEngine:
    def __init__(
        self,
        db_path: str,
        tokenizer: PortugueseTokenizer,
        docstore_path: str | None = None,
        k1: float = 1.2,
        b: float = 0.75,
        use_neural: bool = True
    ):
        t0 = time.time()
        self.tokenizer = tokenizer
        self.db_path = db_path
        self.k1 = k1
        self.b = b

        self.reranker = None
        if use_neural:
            print("[INFO] Loading Neural Reranker...")
            self.reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

        print(f"[INFO] Connecting to DB at {db_path}...")
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cur = self.conn.cursor()

        print("[INFO] Loading doc lengths...")
        self.cur.execute("SELECT doc_id, length FROM doc_lengths")
        self.doc_len: Dict[int, int] = {row[0]: row[1] for row in self.cur.fetchall()}
        
        self.N = len(self.doc_len)
        self.avg_doc_len = sum(self.doc_len.values()) / max(self.N, 1)
        print(f"[INFO] Docs: {self.N:,} | Avg Len: {self.avg_doc_len:.2f}")

        print("[INFO] Loading term stats (term, doc_freq) & computing IDF...")
        self.idf: Dict[str, float] = {}
        
        self.cur.execute("SELECT term, doc_freq FROM inverted_index")
        rows = self.cur.fetchall()
        
        for term, df in rows:
            self.idf[term] = self._bm25_idf(self.N, df)
        
        print(f"[INFO] IDF computed for {len(self.idf):,} terms.")

        self.docstore = {}
        if docstore_path and os.path.exists(docstore_path):
            print("[INFO] Loading docstore...")
            try:
                with open(docstore_path, encoding="utf-8") as f:
                    for line in f:
                        meta = json.loads(line)
                        self.docstore[int(meta["id"])] = meta
            except Exception:
                pass
        
        print(f"[INFO] Search Engine Ready! Startup: {time.time() - t0:.2f}s")

    @staticmethod
    def _bm25_idf(N: int, df: int) -> float:
        if df <= 0 or N <= 0: return 0.0
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    @lru_cache(maxsize=5000)
    def _get_postings(self, term: str) -> Dict[int, List[int]]:
        try:
            cur = self.conn.cursor() 
            cur.execute("SELECT postings FROM inverted_index WHERE term = ?", (term,))
            row = cur.fetchone()
            if row:
                postings = json.loads(row[0])
                return {int(k): v for k, v in postings.items()}
        except Exception as e:
            print(f"[ERROR] Fetching term '{term}': {e}")
        return {}

    def _bm25(self, query_text: str, top_k: int = 10) -> List[Tuple[int, float]]:
        terms = self.tokenizer.tokenize(query_text)
        if not terms: return []

        scores: Dict[int, float] = {}
        
        for t in terms:
            if t not in self.idf:
                continue
            
            postings = self._get_postings(t)
            
            idf_val = self.idf[t]
            
            for doc_id, positions in postings.items():
                tf = len(positions)
                dl = self.doc_len.get(doc_id, 1)
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_doc_len))
                score = idf_val * (tf * (self.k1 + 1)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        ranked = self._bm25(query_text, top_k)
        return [{
            "id": did,
            "title": self.docstore.get(did, {}).get("title", f"Doc {did}"),
            "description": self.docstore.get(did, {}).get("description", ""),
            "score": sc
        } for did, sc in ranked]

    def like_document(self, doc_id: int, top_k: int = 10, expand_terms: int = 20) -> List[Tuple[int, float]]:
        cur = self.conn.cursor()
        cur.execute("SELECT terms_data FROM forward_index WHERE doc_id = ?", (doc_id,))
        row = cur.fetchone()
        
        if not row:
            print(f"[WARN] Doc {doc_id} not found in forward index.")
            return []
            
        tf_doc = json.loads(row[0])
        
        scored_terms = []
        for term, freq in tf_doc.items():
            if term in self.idf:
                weight = freq * self.idf[term]
                scored_terms.append((term, weight))
        
        scored_terms.sort(key=lambda x: x[1], reverse=True)
        top_terms = scored_terms[:expand_terms]

        pseudo_query_terms = []
        for t, w in top_terms:
            reps = max(1, int(math.ceil(w)))
            pseudo_query_terms.extend([t] * min(reps, 5))
        scores: Dict[int, float] = {}
        for t in pseudo_query_terms:
            postings = self._get_postings(t)
            if not postings: continue
            
            idf = self.idf[t]
            for d, positions in postings.items():
                if d == doc_id: continue
                tf = len(positions)
                dl = self.doc_len.get(d, 1)
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_doc_len))
                scores[d] = scores.get(d, 0.0) + (idf * (tf * (self.k1 + 1)) / denom)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def neural_search(self, query_text: str, top_k: int = 10, candidates_k: int = 50) -> List[dict]:
        if not self.reranker:
            print("[WARN] Neural Reranker not loaded. Falling back to BM25.")
            return self.query(query_text, top_k)
            
        candidates = self.query(query_text, top_k=candidates_k)
        
        if not candidates:
            return []
        
        model_inputs = []
        for doc in candidates:
            title = doc.get('title', '')
            desc = doc.get('description', '')
            if title is None: title = ""
            if desc is None: desc = ""
            
            doc_text = f"{title}. {desc}".strip()
            model_inputs.append([query_text, doc_text])

        neural_scores = self.reranker.predict(model_inputs, batch_size=16, show_progress_bar=False)

        for i, doc in enumerate(candidates):
            doc['score'] = float(neural_scores[i])
            doc['rerank_score'] = float(neural_scores[i])

        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[:top_k]

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    tokenizer = PortugueseTokenizer(min_len=3)
    engine = SearchEngine(
        db_path="index.db",
        tokenizer=tokenizer,
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

    # Neural reranking
    print("\n\n Neural reranking:")
    results_neural = engine.neural_search(query_text, top_k=5, candidates_k=50)
    for i, doc in enumerate(results_neural):
        print(f"  {i+1}. {doc['title']} (Neural Score: {doc['score']:.4f})")

    engine.close()
