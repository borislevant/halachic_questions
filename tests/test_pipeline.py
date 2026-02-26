"""Tests for the ingestion pipeline orchestration."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.config import AppConfig, ChunkingConfig, EmbeddingConfig, StorageConfig
from src.ingestion.pipeline import IngestionPipeline, IngestionReport
from src.models.book import Book
from src.models.chunk import Chunk
from src.models.parsed import ParsedBook


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    from src.storage.database import initialize_database

    db_path = tmp_path / "test.db"
    initialize_database(str(db_path))
    return str(db_path)


@pytest.fixture
def config(temp_db):
    """Create a test configuration."""
    return AppConfig(
        storage=StorageConfig(
            sqlite_path=temp_db,
            chroma_dir="./test_chroma",
            books_dir="./test_books",
            processed_dir="./test_processed",
        ),
        embedding=EmbeddingConfig(model="test-model", device="cpu"),
        chunking=ChunkingConfig(),
    )


@pytest.fixture
def mock_parser():
    """Mock BookParser."""
    parser = Mock()
    parser.parse.return_value = ParsedBook(
        title="Test Book",
        author="Test Author",
        language="he",
        raw_text="זוהי הדגמה של טקסט עברי לבדיקה.",
        source_path="test.txt",
        file_format="txt",
    )
    return parser


@pytest.fixture
def mock_chunker():
    """Mock HalachicChunker."""
    chunker = Mock()
    chunker.chunk.return_value = [
        Chunk(
            id="chunk1",
            book_id="book123",
            book_title="Test Book",
            text="זוהי פיסקה ראשונה",
            chunk_index=0,
            token_count=50,
        ),
        Chunk(
            id="chunk2",
            book_id="book123",
            book_title="Test Book",
            text="זוהי פיסקה שנייה",
            chunk_index=1,
            token_count=50,
        ),
    ]
    return chunker


@pytest.fixture
def mock_embedder():
    """Mock TextEmbedder."""
    embedder = Mock()
    embedder.embed.return_value = [[0.1] * 768, [0.2] * 768]  # 2 embeddings
    embedder.model_name = "test-model"
    embedder.device = "cpu"
    embedder.get_embedding_dimension.return_value = 768
    return embedder


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore."""
    store = Mock()
    store.add_chunks.return_value = None
    store.delete_by_book_id.return_value = 2
    store.count.return_value = 100
    store.is_initialized = True
    return store


@pytest.fixture
def pipeline(config, mock_parser, mock_chunker, mock_embedder, mock_vector_store):
    """Create a pipeline with mocked dependencies."""
    return IngestionPipeline(
        config=config,
        parser=mock_parser,
        chunker=mock_chunker,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )


# ============================================================================
# Unit Tests (Mock Dependencies)
# ============================================================================


def test_successful_ingest_book(pipeline, mock_parser, mock_chunker, mock_embedder, mock_vector_store, temp_db):
    """Test successful book ingestion writes vectors and DB metadata."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write("Test content".encode())
        file_path = f.name

    try:
        report = pipeline.ingest_book(file_path, author="Test Author", show_progress=False)

        assert report.success is True
        assert report.book_title == "Test Book"
        assert report.chunks_created == 2
        assert len(report.errors) == 0

        # Verify vector store was called
        mock_vector_store.add_chunks.assert_called_once()

        # Verify DB was updated
        from src.storage.database import get_book_by_id

        book = get_book_by_id(temp_db, report.book_id)
        assert book is not None
        assert book.title == "Test Book"
        assert book.author == "Test Author"
        assert book.chunk_count == 2
        assert book.status == "active"
    finally:
        Path(file_path).unlink()


def test_empty_parsed_text(pipeline, mock_parser, temp_db):
    """Test empty parsed text returns failure report and no vector writes."""
    mock_parser.parse.return_value = ParsedBook(
        title="Empty Book",
        author="",
        language="en",
        raw_text="",  # Empty text
        source_path="empty.txt",
        file_format="txt",
    )

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        file_path = f.name

    try:
        report = pipeline.ingest_book(file_path, show_progress=False)

        assert report.success is False
        assert "empty" in report.errors[0].lower()
        assert report.chunks_created == 0

        # Verify vector store was NOT called
        pipeline._vector_store.add_chunks.assert_not_called()
    finally:
        Path(file_path).unlink()


def test_empty_chunks(pipeline, mock_chunker, temp_db):
    """Test empty chunks returns failure report."""
    mock_chunker.chunk.return_value = []  # No chunks

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write("Some content".encode())
        file_path = f.name

    try:
        report = pipeline.ingest_book(file_path, show_progress=False)

        assert report.success is False
        assert "no chunks" in report.errors[0].lower()
        assert report.chunks_created == 0

        # Verify DB status was updated to error
        from src.storage.database import get_book_by_id

        book = get_book_by_id(temp_db, report.book_id)
        assert book is not None
        assert book.status == "error"
    finally:
        Path(file_path).unlink()


def test_embedding_count_mismatch(pipeline, mock_embedder, temp_db):
    """Test embedding count mismatch returns failure report."""
    # Return only 1 embedding for 2 chunks
    mock_embedder.embed.return_value = [[0.1] * 768]

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write("Test content".encode())
        file_path = f.name

    try:
        report = pipeline.ingest_book(file_path, show_progress=False)

        assert report.success is False
        assert "mismatch" in report.errors[0].lower()

        # Verify vector store was NOT called
        pipeline._vector_store.add_chunks.assert_not_called()
    finally:
        Path(file_path).unlink()


def test_parser_exception(pipeline, mock_parser, temp_db):
    """Test parser exception captured into report errors."""
    mock_parser.parse.side_effect = ValueError("Invalid file format")

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        file_path = f.name

    try:
        report = pipeline.ingest_book(file_path, show_progress=False)

        assert report.success is False
        assert len(report.errors) > 0
        assert "invalid file format" in report.errors[0].lower()
    finally:
        Path(file_path).unlink()


def test_vector_store_exception(pipeline, mock_vector_store, temp_db):
    """Test vector store exception captured into report errors."""
    mock_vector_store.add_chunks.side_effect = RuntimeError("Vector store error")

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write("Test content".encode())
        file_path = f.name

    try:
        report = pipeline.ingest_book(file_path, show_progress=False)

        assert report.success is False
        assert len(report.errors) > 0

        # Verify DB status was updated to error
        from src.storage.database import get_book_by_id

        book = get_book_by_id(temp_db, report.book_id)
        assert book is not None
        assert book.status == "error"
    finally:
        Path(file_path).unlink()


def test_book_id_injection(pipeline, temp_db):
    """Test book_id injection preserved in chunks and DB row."""
    custom_book_id = "custom-book-id-12345"

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write("Test content".encode())
        file_path = f.name

    try:
        report = pipeline.ingest_book(file_path, book_id=custom_book_id, show_progress=False)

        assert report.success is True
        assert report.book_id == custom_book_id

        # Verify DB has the custom book_id
        from src.storage.database import get_book_by_id

        book = get_book_by_id(temp_db, custom_book_id)
        assert book is not None
        assert book.id == custom_book_id

        # Verify chunker was called with the custom book_id
        pipeline._chunker.chunk.assert_called_once()
        call_kwargs = pipeline._chunker.chunk.call_args[1]
        assert call_kwargs["book_id"] == custom_book_id
    finally:
        Path(file_path).unlink()


def test_author_override(pipeline, mock_parser, temp_db):
    """Test author override is applied."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write("Test content".encode())
        file_path = f.name

    try:
        custom_author = "Custom Author Name"
        report = pipeline.ingest_book(file_path, author=custom_author, show_progress=False)

        assert report.success is True

        # Verify DB has the custom author
        from src.storage.database import get_book_by_id

        book = get_book_by_id(temp_db, report.book_id)
        assert book is not None
        assert book.author == custom_author
    finally:
        Path(file_path).unlink()


def test_hebrew_text_warning(pipeline, mock_parser, temp_db):
    """Test Hebrew-text warning behavior for low Hebrew char count."""
    # Return a "Hebrew" book with very little Hebrew
    mock_parser.parse.return_value = ParsedBook(
        title="Almost No Hebrew",
        author="",
        language="he",  # Marked as Hebrew
        raw_text="This is mostly English text with maybe א ב ג",
        source_path="test.txt",
        file_format="txt",
    )

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write("Test content".encode())
        file_path = f.name

    try:
        report = pipeline.ingest_book(file_path, show_progress=False)

        assert report.success is True
        assert len(report.warnings) > 0
        assert "hebrew text detected" in report.warnings[0].lower()
    finally:
        Path(file_path).unlink()


# ============================================================================
# Lifecycle Tests
# ============================================================================


def test_remove_book(pipeline, temp_db):
    """Test remove_book removes from vector store and DB."""
    # First ingest a book
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write("Test content".encode())
        file_path = f.name

    try:
        report = pipeline.ingest_book(file_path, show_progress=False)
        assert report.success is True
        book_id = report.book_id

        # Verify book exists in DB
        from src.storage.database import get_book_by_id

        book = get_book_by_id(temp_db, book_id)
        assert book is not None

        # Remove the book
        success = pipeline.remove_book(book_id)
        assert success is True

        # Verify vector store delete was called
        pipeline._vector_store.delete_by_book_id.assert_called_with(book_id)

        # Verify book is removed from DB
        book_after = get_book_by_id(temp_db, book_id)
        assert book_after is None
    finally:
        Path(file_path).unlink()


def test_reindex_book(pipeline, temp_db):
    """Test reindex_book removes old and ingests with same book_id."""
    # First ingest a book
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write("Original content".encode())
        file_path = f.name

    try:
        report = pipeline.ingest_book(file_path, show_progress=False)
        assert report.success is True
        original_book_id = report.book_id

        # Now reindex with new content
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f2:
            f2.write("New content".encode())
            new_file_path = f2.name

        try:
            reindex_report = pipeline.reindex_book(
                new_file_path, book_id=original_book_id, show_progress=False
            )

            assert reindex_report.success is True
            assert reindex_report.book_id == original_book_id

            # Verify vector store delete was called
            pipeline._vector_store.delete_by_book_id.assert_called()

            # Verify book still exists in DB with same ID
            from src.storage.database import get_book_by_id

            book = get_book_by_id(temp_db, original_book_id)
            assert book is not None
            assert book.id == original_book_id
        finally:
            Path(new_file_path).unlink()
    finally:
        Path(file_path).unlink()


def test_get_stats(pipeline):
    """Test get_stats returns expected keys and values."""
    stats = pipeline.get_stats()

    assert "total_chunks" in stats
    assert "embedding_model" in stats
    assert "embedding_device" in stats
    assert "embedding_dimension" in stats
    assert "vector_store_initialized" in stats

    assert stats["total_chunks"] == 100
    assert stats["embedding_model"] == "test-model"
    assert stats["embedding_device"] == "cpu"
    assert stats["embedding_dimension"] == 768
    assert stats["vector_store_initialized"] is True


# ============================================================================
# Directory Ingestion Tests
# ============================================================================


def test_ingest_directory_supported_files(pipeline, tmp_path, temp_db):
    """Test ingest_directory processes supported files only."""
    # Create test files
    (tmp_path / "book1.txt").write_text("Test content 1")
    (tmp_path / "book2.pdf").write_text("Test content 2")
    (tmp_path / "book3.docx").write_text("Test content 3")
    (tmp_path / "ignored.xyz").write_text("Should be ignored")

    reports = pipeline.ingest_directory(tmp_path, show_progress=False)

    # Should have processed 3 files (txt, pdf, docx), not the .xyz file
    assert len(reports) == 3
    assert all(r.success for r in reports)


def test_ingest_directory_continues_after_failure(pipeline, mock_parser, tmp_path, temp_db):
    """Test ingest_directory continues after one file fails."""
    # Create test files
    (tmp_path / "good1.txt").write_text("Test content 1")
    (tmp_path / "bad.txt").write_text("Will fail")
    (tmp_path / "good2.txt").write_text("Test content 2")

    # Make parser fail on the second file
    call_count = [0]

    def parse_side_effect(file_path):
        call_count[0] += 1
        if call_count[0] == 2:  # Second call
            raise ValueError("Simulated parsing error")
        return ParsedBook(
            title=f"Book {call_count[0]}",
            author="",
            language="en",
            raw_text="Some content",
            source_path=str(file_path),
            file_format="txt",
        )

    mock_parser.parse.side_effect = parse_side_effect

    reports = pipeline.ingest_directory(tmp_path, show_progress=False)

    # Should have 3 reports: 2 successful, 1 failed
    assert len(reports) == 3
    successful = sum(1 for r in reports if r.success)
    failed = sum(1 for r in reports if not r.success)
    assert successful == 2
    assert failed == 1


def test_ingest_directory_recursive(pipeline, tmp_path, temp_db):
    """Test recursive vs non-recursive behavior."""
    # Create nested structure
    (tmp_path / "root.txt").write_text("Root file")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("Nested file")

    # Non-recursive: should only find root.txt
    reports_non_recursive = pipeline.ingest_directory(tmp_path, recursive=False, show_progress=False)
    assert len(reports_non_recursive) == 1

    # Recursive: should find both
    reports_recursive = pipeline.ingest_directory(tmp_path, recursive=True, show_progress=False)
    assert len(reports_recursive) == 2


# ============================================================================
# Integration Test
# ============================================================================


def test_end_to_end_with_real_txt(temp_db, tmp_path):
    """End-to-end test with tiny real TXT fixture."""
    from src.config import AppConfig, ChunkingConfig, EmbeddingConfig, StorageConfig
    from src.embeddings.embedder import TextEmbedder
    from src.ingestion.chunker import HalachicChunker
    from src.ingestion.parser import BookParser
    from src.ingestion.pipeline import IngestionPipeline
    from src.retrieval.vector_store import VectorStore

    # Create a real text file
    test_file = tmp_path / "test_book.txt"
    test_file.write_text("זהו ספר לבדיקה. סימן א: הלכה ראשונה בדבר חשוב.", encoding="utf-8")

    # Create real components with fake embedder
    config = AppConfig(
        storage=StorageConfig(
            sqlite_path=temp_db,
            chroma_dir=str(tmp_path / "chroma"),
            books_dir=str(tmp_path / "books"),
            processed_dir=str(tmp_path / "processed"),
        ),
        embedding=EmbeddingConfig(model="fake-model", device="cpu"),
        chunking=ChunkingConfig(),
    )

    parser = BookParser()
    chunker = HalachicChunker(config.chunking)

    # Mock embedder that returns fake embeddings
    embedder = Mock()
    embedder.embed.return_value = [[0.1] * 768]  # 1 embedding
    embedder.model_name = "fake-model"
    embedder.device = "cpu"
    embedder.get_embedding_dimension.return_value = 768

    # Mock vector store
    vector_store = Mock()
    vector_store.add_chunks.return_value = None
    vector_store.count.return_value = 1
    vector_store.is_initialized = True

    pipeline = IngestionPipeline(
        config=config,
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    # Run ingestion
    report = pipeline.ingest_book(test_file, author="Test Author", show_progress=False)

    # Verify success
    assert report.success is True
    assert report.chunks_created >= 1
    assert "בדיקה" in report.book_title or "test_book" in report.book_title

    # Verify DB persistence
    from src.storage.database import get_book_by_id

    book = get_book_by_id(temp_db, report.book_id)
    assert book is not None
    assert book.author == "Test Author"
    assert book.status == "active"
    assert book.chunk_count >= 1
