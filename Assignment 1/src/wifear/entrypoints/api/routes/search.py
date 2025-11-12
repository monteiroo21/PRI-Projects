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
    metadata_path="index_blocks/metadata.json",
)


@router.get("/search")
def search(query: str, num_results: int = 10) -> SearchResponse:
    """Search for documents matching the given query."""
    results = engine.query(query, top_k=num_results)
    docs = [Document(id=doc_id, title=f"Doc {doc_id}", content="...") for doc_id, _ in results]
    return SearchResponse(results=docs)


@router.get("/search_like", response_model=SearchResponse)
def search_like(doc_id: int, num_results: int = 10):
    """Search for documents similar to a given document ID."""
    results = engine.like_document(doc_id, top_k=num_results)
    docs = [Document(id=doc_id, title=f"Doc {doc_id}", content="...") for doc_id, _ in results]
    return SearchResponse(results=docs)
