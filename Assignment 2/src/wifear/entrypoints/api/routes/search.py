"""Search endpoints."""

from fastapi import APIRouter, Query
from wifear.core.model import Document
from wifear.entrypoints.api.model import SearchResponse
from wifear.core.tokenizer import PortugueseTokenizer
from wifear.core.searcher import SearchEngine

router = APIRouter(tags=["search engine"])

tokenizer = PortugueseTokenizer(min_len=3)
engine = SearchEngine(
    db_path="index.db",
    tokenizer=tokenizer,
    docstore_path="data/docstore.jsonl"
)

@router.get("/search", response_model=SearchResponse)
def search(query: str, num_results: int = 10) -> SearchResponse:
    """Search for documents matching the given query."""
    results = engine.query(query, top_k=num_results)

    docs = [
        Document(
            id=r["id"],
            title=r.get("title", f"Doc {r['id']}"),
            content=r.get("description", "")[:1000],
        )
        for r in results
    ]

    return SearchResponse(results=docs)

@router.get("/search_like", response_model=SearchResponse)
def search_like(doc_id: int, num_results: int = 10) -> SearchResponse:
    """Search for documents similar to a given document ID."""
    similar_results = engine.like_document(doc_id, top_k=num_results)

    docs = []
    for d_id, score in similar_results:
        meta = engine.docstore.get(d_id, {})
        docs.append(
            Document(
                id=d_id,
                title=meta.get("title", f"Doc {d_id}"),
                content=meta.get("description", "")[:1000],
            )
        )

    return SearchResponse(results=docs)
