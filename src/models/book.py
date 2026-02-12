"""Book data model."""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class Book(BaseModel):
    """Represents an ingested Halachic book."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    author: str = ""
    language: str = "he"
    source_path: str
    file_format: str
    chunk_count: int = 0
    ingested_at: datetime = Field(default_factory=datetime.now)
    status: str = "active"  # "active", "ingesting", "error"
