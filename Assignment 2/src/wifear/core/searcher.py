from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from functools import lru_cache
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from wifear.core.tokenizer import PortugueseTokenizer

try:
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"[INFO] .env carregado de: {env_path}")
except Exception:
    pass


class SearchEngine:
    def __init__(
        self,
        db_path: str,
        tokenizer: PortugueseTokenizer,
        docstore_path: str | None = None,
        k1: float = 1.2,
        b: float = 0.75,
        use_neural: bool = True,
    ):
        t0 = time.time()
        self.tokenizer = tokenizer
        self.db_path = db_path
        self.k1 = k1
        self.b = b

        self.reranker = None
        if use_neural:
            print("[INFO] Loading Neural Reranker...")
            self.reranker = CrossEncoder("unicamp-dl/mMiniLM-L6-v2-pt-v2", max_length=512)

        self.llm_model = None
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                model_name = "gemini-2.5-flash"
                print(f"[INFO] Initialized Gemini using model: '{model_name}'")
                self.llm_model = genai.GenerativeModel(model_name)
            except Exception as e:
                print(f"[ERROR] Failed to init Gemini: {e}")
        else:
            print("[WARN] GOOGLE_API_KEY not found in .env. RAG disabled.")

        print(f"[INFO] Connecting to DB at {db_path}...")
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cur = self.conn.cursor()

        print("[INFO] Loading doc lengths...")
        self.cur.execute("SELECT doc_id, length FROM doc_lengths")
        self.doc_len: dict[int, int] = {row[0]: row[1] for row in self.cur.fetchall()}

        self.N = len(self.doc_len)
        self.avg_doc_len = sum(self.doc_len.values()) / max(self.N, 1)
        print(f"[INFO] Docs: {self.N:,} | Avg Len: {self.avg_doc_len:.2f}")

        print("[INFO] Loading term stats (term, doc_freq) & computing IDF...")
        self.idf: dict[str, float] = {}

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
        if df <= 0 or N <= 0:
            return 0.0
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    @lru_cache(maxsize=5000)
    def _get_postings(self, term: str) -> dict[int, list[int]]:
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

    def _bm25(self, query_text: str, top_k: int = 10) -> list[tuple[int, float]]:
        terms = self.tokenizer.tokenize(query_text)
        if not terms:
            return []

        scores: dict[int, float] = {}

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

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        ranked = self._bm25(query_text, top_k)
        return [
            {
                "id": did,
                "title": self.docstore.get(did, {}).get("title", f"Doc {did}"),
                "description": self.docstore.get(did, {}).get("description", ""),
                "score": sc,
            }
            for did, sc in ranked
        ]

    def like_document(
        self, doc_id: int, top_k: int = 10, expand_terms: int = 20
    ) -> list[tuple[int, float]]:
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
        scores: dict[int, float] = {}
        for t in pseudo_query_terms:
            postings = self._get_postings(t)
            if not postings:
                continue

            idf = self.idf[t]
            for d, positions in postings.items():
                if d == doc_id:
                    continue
                tf = len(positions)
                dl = self.doc_len.get(d, 1)
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_doc_len))
                scores[d] = scores.get(d, 0.0) + (idf * (tf * (self.k1 + 1)) / denom)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def neural_search(self, query_text: str, top_k: int = 10, candidates_k: int = 50) -> list[dict]:
        """Performs a two-stage retrieval:

        - Retrieval: Fetches top-K candidates using BM25.
        - Reranking: Re-orders candidates using a Cross-Encoder model.
        """

        if not self.reranker:
            print("[WARN] Neural Reranker not loaded. Falling back to BM25.")
            return self.query(query_text, top_k)

        # Retrieve initial candidates using BM25
        candidates = self.query(query_text, top_k=candidates_k)

        for i in range(top_k):
            print(f"Candidate {i+1}: {candidates[i]['title']}, Score: {candidates[i]['score']:.4f}")

        if not candidates:
            return []

        # Prepare pairs to score
        pairs_to_score = []

        for doc in candidates:
            title = doc.get("title", "")
            desc = doc.get("description", "")
            # Map pair index to document index
            full_text = f"{title}. {desc}"
            pairs_to_score.append([query_text, full_text])

        if not pairs_to_score:
            return candidates[:top_k]

        # Get reranked scores
        all_scores = self.reranker.predict(pairs_to_score, batch_size=64, show_progress_bar=True)

        # Update the candidates with the best snippets
        for i, doc in enumerate(candidates):
            doc["score"] = float(all_scores[i])

        # Sort by rerank score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        for i in range(top_k):
            print(
                f"Reranked {i+1}: {candidates[i]['title']}, "
                + f"Neural Score: {candidates[i]['score']:.4f}"
            )

        return candidates[:top_k]

    def generate_answer(self, query_text: str, relevant_docs: list[dict]) -> str:
        """
        Generates a natural language answer based on the top 5 documents retrieved.
        """

        # Check if LLM model is initialized
        if not self.llm_model:
            return "Error: Gemini API is not configured or LLM model is missing."

        # Check if relevant documents are provided
        if not relevant_docs:
            return "Error: No relevant documents provided."

        # 1. Select the top 5 documents (or fewer if there are few results)
        top_k_context = 5
        docs_to_use = relevant_docs[:top_k_context]

        # 2. Build the context block by combining the texts of the documents
        context_parts = []
        for i, doc in enumerate(docs_to_use, 1):
            title = doc.get("title", "Documento Sem Título")
            content = doc.get("description", "")
            
            # Clear formatting to help the LLM distinguish between documents
            doc_str = (
                f"--- DOCUMENTO {i} ---\n"
                f"Título: {title}\n"
                f"Conteúdo: {content}\n"
            )
            context_parts.append(doc_str)

        full_context = "\n".join(context_parts)

        # 3. Build the prompt
        prompt = (
            f"És um assistente inteligente de recuperação de informação. "
            f"O utilizador fez uma pergunta e abaixo estão os 5 documentos mais relevantes encontrados no sistema.\n"
            f"A tua tarefa é sintetizar uma resposta completa e natural baseada APENAS nestes documentos.\n\n"
            f"CONTEXTO (Documentos Recuperados):\n"
            f"{full_context}\n"
            f"---------------------------------------------------\n"
            f"PERGUNTA DO UTILIZADOR: {query_text}\n\n"
            f"INSTRUÇÕES:\n"
            f"- Responde em Português de Portugal.\n"
            f"- Usa a informação de múltiplos documentos se necessário para criar uma resposta mais completa.\n"
            f"- Se a resposta não estiver nos documentos, diz 'A informação recuperada não contém a resposta'.\n"
            f"- Não inventes factos que não estejam no contexto.\n\n"
            f"RESPOSTA:"
        )

        try:
            # Adjust max_output_tokens if you want shorter or longer responses
            response = self.llm_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating response with LLM: {e}"

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
    print(f"\n--- Neural Search: '{query_text}' ---")
    results_neural = engine.neural_search(query_text, top_k=5, candidates_k=50)
    for i, doc in enumerate(results_neural):
        print(f"  {i+1}. {doc['title']} (Neural Score: {doc['score']:.4f})")

    # Answer Generation (RAG)
    if results:
        print(f"\n--- Generating Answer (Gemini): '{query_text}' ---")
        answer = engine.generate_answer(query_text, results_neural)
        print(answer)

    engine.close()
    print("\nDone!")