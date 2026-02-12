"""Parsed book data models for the ingestion pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Section(BaseModel):
    """A structural section detected within a parsed book.

    Sections form a tree: a perek contains halachot, a siman contains
    seifim, etc. The ``subsections`` field holds child sections.
    """

    section_type: str  # "perek", "siman", "seif", "halacha", "siman_katan", "paragraph"
    title: str  # e.g. "סימן שכח" or "א" — the matched header text
    text: str  # Full text content of this section (excluding subsections' text)
    char_start: int  # Start position in the full raw_text
    char_end: int  # End position in the full raw_text
    subsections: list[Section] = Field(default_factory=list)
    level: int = 0  # Depth in hierarchy (0 = top-level)


class ParsedBook(BaseModel):
    """The result of parsing a raw book file.

    Contains the full raw text, detected structural sections,
    and metadata extracted during parsing.
    """

    title: str
    author: str = ""
    language: str = "he"  # "he", "arc", "en", "mixed"
    raw_text: str
    sections: list[Section] = Field(default_factory=list)
    source_path: str
    file_format: str  # "pdf", "txt", "docx", "html"
