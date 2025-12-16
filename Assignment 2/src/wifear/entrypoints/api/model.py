from pydantic import BaseModel

from wifear.core.model import Document


class SearchRequest(BaseModel):
    query: str
    num_results: int = 10


class SearchDocumentResult(Document):
    score: float
    bm25_score: float | None = None
    snippet: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchDocumentResult]
    answer: str | None = None
