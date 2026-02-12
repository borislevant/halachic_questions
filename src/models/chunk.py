"""Chunk data model."""

from uuid import uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single chunk of text from a Halachic book."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    book_id: str
    book_title: str
    book_author: str = ""
    section_path: str = ""
    section_type: str = ""  # "siman", "seif", "halacha", "perek", "paragraph"
    chunk_index: int = 0
    total_chunks_in_section: int = 1
    language: str = "he"
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
