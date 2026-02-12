"""Tests for data models."""

from datetime import datetime

from src.models import (
    Book,
    Chunk,
    Citation,
    GeneratedAnswer,
    QueryResult,
    RetrievalResult,
)


class TestBook:
    def test_create_book(self) -> None:
        book = Book(
            title="שולחן ערוך",
            author="רבי יוסף קארו",
            source_path="/data/books/shulchan_aruch.pdf",
            file_format="pdf",
        )
        assert book.title == "שולחן ערוך"
        assert book.author == "רבי יוסף קארו"
        assert book.status == "active"
        assert book.chunk_count == 0
        assert book.id  # UUID auto-generated

    def test_book_defaults(self) -> None:
        book = Book(
            title="Test",
            source_path="/test.txt",
            file_format="txt",
        )
        assert book.language == "he"
        assert book.author == ""
        assert book.status == "active"
        assert isinstance(book.ingested_at, datetime)

    def test_book_serialization(self) -> None:
        book = Book(
            title="Test Book",
            source_path="/test.pdf",
            file_format="pdf",
        )
        data = book.model_dump()
        assert data["title"] == "Test Book"
        restored = Book(**data)
        assert restored.title == book.title
        assert restored.id == book.id


class TestChunk:
    def test_create_chunk(self) -> None:
        chunk = Chunk(
            text="הלכה א: מותר לחמם מים",
            book_id="book-123",
            book_title="שולחן ערוך",
            section_path="אורח חיים > סימן שכח > סעיף ב",
            section_type="seif",
        )
        assert chunk.text == "הלכה א: מותר לחמם מים"
        assert chunk.section_type == "seif"
        assert chunk.id  # UUID auto-generated

    def test_chunk_defaults(self) -> None:
        chunk = Chunk(text="test", book_id="b1", book_title="t")
        assert chunk.language == "he"
        assert chunk.chunk_index == 0
        assert chunk.token_count == 0


class TestRetrievalResult:
    def test_create_retrieval_result(self) -> None:
        chunk = Chunk(text="sample", book_id="b1", book_title="t")
        result = RetrievalResult(chunk=chunk, similarity_score=0.85)
        assert result.similarity_score == 0.85
        assert result.rerank_score is None
        assert result.context_before is None


class TestCitation:
    def test_create_citation(self) -> None:
        citation = Citation(
            book_title="שולחן ערוך",
            section_path="אורח חיים, סימן שכח",
            source_chunk_id="chunk-123",
        )
        assert citation.is_valid is True

    def test_invalid_citation(self) -> None:
        citation = Citation(
            book_title="Unknown",
            section_path="???",
            is_valid=False,
        )
        assert citation.is_valid is False


class TestGeneratedAnswer:
    def test_create_answer(self) -> None:
        answer = GeneratedAnswer(
            text="לפי השולחן ערוך...",
            model_used="claude-sonnet-4-20250514",
            tokens_used=150,
            latency_ms=3200,
        )
        assert answer.citations == []
        assert answer.tokens_used == 150


class TestQueryResult:
    def test_create_query_result(self) -> None:
        result = QueryResult(question="האם מותר לחמם מים בשבת?")
        assert result.question == "האם מותר לחמם מים בשבת?"
        assert result.sources == []
        assert result.answer is None
        assert result.feedback is None
        assert result.id  # UUID auto-generated

    def test_query_result_serialization(self) -> None:
        chunk = Chunk(text="sample", book_id="b1", book_title="t")
        retrieval = RetrievalResult(chunk=chunk, similarity_score=0.9)
        answer = GeneratedAnswer(text="Answer text", model_used="test")
        result = QueryResult(
            question="Test?",
            sources=[retrieval],
            answer=answer,
        )
        data = result.model_dump()
        restored = QueryResult(**data)
        assert restored.question == "Test?"
        assert len(restored.sources) == 1
        assert restored.answer is not None
        assert restored.answer.text == "Answer text"
