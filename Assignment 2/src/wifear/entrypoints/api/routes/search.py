"""Search endpoints."""

from fastapi import APIRouter, HTTPException

from wifear.core.model import Document
from wifear.core.tokenizer import PortugueseTokenizer
from wifear.entrypoints.api.model import SearchDocumentResult, SearchResponse

from ....core.searcher import SearchEngine

router = APIRouter(tags=["search engine"])

tokenizer = PortugueseTokenizer(min_len=3)
engine = SearchEngine(db_path="index.db", tokenizer=tokenizer, docstore_path="data/docstore.jsonl")


@router.get("/search", response_model=SearchResponse)
def search(query: str, num_results: int = 10) -> SearchResponse:
    """Search for documents matching the given query."""
    results = engine.neural_search(query, top_k=num_results, candidates_k=50)

    generated_answer = engine.generate_answer(query, results)

    docs = [
        SearchDocumentResult(
            id=r["id"],
            title=r.get("title", f"Doc {r['id']}"),
            content=r.get("description", "")[:1000],
            score=r["score"],
            bm25_score=r.get("initial_score"),
            snippet=r.get("snippet"),
        )
        for r in results
    ]

    print("Search query:", query)
    for result in results:
        print(
            f"  Doc ID: {result['id']}, Score: {result['score']:.4f}, "
            f"Snippet: {result.get('snippet', '')}"
        )

    return SearchResponse(results=docs, answer=generated_answer)


@router.get("/search_like", response_model=SearchResponse)
def search_like(doc_id: int, num_results: int = 10) -> SearchResponse:
    """Search for documents similar to a given document ID."""
    similar_results = engine.like_document(doc_id, top_k=num_results)

    docs = []
    for d_id, score in similar_results:
        meta = engine.docstore.get(d_id, {})
        docs.append(
            SearchDocumentResult(
                id=d_id,
                title=meta.get("title", f"Doc {d_id}"),
                content=meta.get("description", "")[:1000],
                score=score,
            )
        )

    return SearchResponse(results=docs)

@router.get("/documents/{doc_id}")
def get_document_details(doc_id: int):
    """Return document details by ID."""
    doc = engine.docstore.get(doc_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return doc
