"""Query result data models."""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from src.models.chunk import Chunk


class RetrievalResult(BaseModel):
    """A single retrieved chunk with scoring metadata."""

    chunk: Chunk
    similarity_score: float
    rerank_score: float | None = None
    context_before: str | None = None
    context_after: str | None = None


class Citation(BaseModel):
    """A citation extracted from a generated answer."""

    book_title: str
    section_path: str
    source_chunk_id: str = ""
    is_valid: bool = True


class GeneratedAnswer(BaseModel):
    """An LLM-generated answer with metadata."""

    text: str
    citations: list[Citation] = Field(default_factory=list)
    model_used: str = ""
    tokens_used: int = 0
    latency_ms: int = 0


class QueryResult(BaseModel):
    """A complete query result: question, sources, and answer."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    sources: list[RetrievalResult] = Field(default_factory=list)
    answer: GeneratedAnswer | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    feedback: str | None = None  # "positive", "negative", None
