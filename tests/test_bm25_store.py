"""Tests for BM25Store keyword-based retrieval."""

import tempfile
from pathlib import Path

import pytest

from src.models.chunk import Chunk
from src.retrieval.bm25_store import BM25Store


@pytest.fixture
def temp_bm25_dir() -> Path:
    """Create a temporary directory for BM25 index files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Sample Hebrew chunks for testing."""
    return [
        Chunk(
            id="chunk-1",
            text="שולחן ערוך אורח חיים סימן שכח דין הדלקת נרות",
            book_id="book-1",
            book_title="שולחן ערוך",
            book_author="רבי יוסף קארו",
            section_path="אורח חיים, סימן שכח",
            section_type="siman",
            chunk_index=0,
            total_chunks_in_section=1,
            language="he",
            char_start=0,
            char_end=50,
            token_count=10,
        ),
        Chunk(
            id="chunk-2",
            text="משנה ברורה על הדלקת נרות בשבת קודש",
            book_id="book-2",
            book_title="משנה ברורה",
            book_author="החפץ חיים",
            section_path="סימן רסג",
            section_type="siman",
            chunk_index=0,
            total_chunks_in_section=1,
            language="he",
            char_start=0,
            char_end=40,
            token_count=8,
        ),
        Chunk(
            id="chunk-3",
            text="חזון איש הלכות שבת ומוקצה",
            book_id="book-3",
            book_title="חזון איש",
            book_author="הרב אברהם ישעיהו קרליץ",
            section_path="שבת, סימן לז",
            section_type="siman",
            chunk_index=0,
            total_chunks_in_section=1,
            language="he",
            char_start=0,
            char_end=30,
            token_count=6,
        ),
        Chunk(
            id="chunk-4",
            text="דיני בישול בשבת לפי הרמב\"ם",
            book_id="book-4",
            book_title="משנה תורה",
            book_author="הרמב\"ם",
            section_path="הלכות שבת, פרק ט",
            section_type="perek",
            chunk_index=0,
            total_chunks_in_section=1,
            language="he",
            char_start=0,
            char_end=35,
            token_count=7,
        ),
    ]


def test_bm25_store_initialization(temp_bm25_dir: Path) -> None:
    """Test BM25Store initialization."""
    store = BM25Store(bm25_dir=temp_bm25_dir)

    assert not store.is_loaded
    assert store.chunk_count == 0
    assert temp_bm25_dir.exists()


def test_build_index(temp_bm25_dir: Path, sample_chunks: list[Chunk]) -> None:
    """Test building BM25 index from chunks."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index(sample_chunks)

    assert store.is_loaded
    assert store.chunk_count == 4


def test_build_index_empty_chunks(temp_bm25_dir: Path) -> None:
    """Test building index with empty chunk list."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index([])

    assert not store.is_loaded
    assert store.chunk_count == 0


def test_save_and_load_index(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test saving and loading BM25 index."""
    # Build and save
    store1 = BM25Store(bm25_dir=temp_bm25_dir)
    store1.build_index(sample_chunks)
    store1.save_index()

    # Check files were created
    assert (temp_bm25_dir / "bm25_index.pkl").exists()
    assert (temp_bm25_dir / "bm25_chunks.pkl").exists()

    # Load in new store
    store2 = BM25Store(bm25_dir=temp_bm25_dir)
    assert not store2.is_loaded  # Not loaded yet

    loaded = store2.load_index()
    assert loaded
    assert store2.is_loaded
    assert store2.chunk_count == 4


def test_load_index_no_files(temp_bm25_dir: Path) -> None:
    """Test loading index when files don't exist."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    loaded = store.load_index()

    assert not loaded
    assert not store.is_loaded


def test_search_exact_term_match(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test BM25 search with exact term matches."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index(sample_chunks)

    # Search for specific book name
    results = store.search(query="חזון איש", top_k=5)

    assert len(results) > 0
    # "חזון איש" chunk should be top ranked
    assert "חזון איש" in results[0]["text"]
    assert results[0]["id"] == "chunk-3"


def test_search_section_reference(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test BM25 search with section references."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index(sample_chunks)

    # Search for specific siman
    results = store.search(query="סימן שכח", top_k=5)

    assert len(results) > 0
    # Chunk with "סימן שכח" should rank high
    top_texts = [r["text"] for r in results[:2]]
    assert any("סימן שכח" in text for text in top_texts)


def test_search_multiple_terms(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test BM25 search with multiple terms."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index(sample_chunks)

    # Search for "שבת" - should match multiple chunks
    results = store.search(query="שבת", top_k=5)

    assert len(results) >= 2
    # Check that multiple relevant chunks are returned
    texts = [r["text"] for r in results]
    shabbat_mentions = sum(1 for text in texts if "שבת" in text)
    assert shabbat_mentions >= 2


def test_search_with_filter(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test BM25 search with metadata filters."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index(sample_chunks)

    # Search only in book-1
    results = store.search(
        query="הדלקת נרות",
        top_k=5,
        filter_dict={"book_id": "book-1"},
    )

    assert len(results) > 0
    # All results should be from book-1
    for result in results:
        assert result["metadata"]["book_id"] == "book-1"


def test_search_not_loaded(temp_bm25_dir: Path) -> None:
    """Test search raises error when index not loaded."""
    store = BM25Store(bm25_dir=temp_bm25_dir)

    results = store.search(query="test", top_k=5)
    assert results == []


def test_search_empty_query(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test search with empty query."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index(sample_chunks)

    results = store.search(query="", top_k=5)
    assert results == []


def test_search_top_k_limit(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test that search respects top_k parameter."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index(sample_chunks)

    results = store.search(query="שבת", top_k=2)
    assert len(results) <= 2


def test_delete_by_book_id(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test deleting chunks by book ID."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index(sample_chunks)

    initial_count = store.chunk_count
    deleted_count = store.delete_by_book_id("book-1")

    assert deleted_count == 1
    assert store.chunk_count == initial_count - 1
    assert store.is_loaded

    # Verify deleted chunk is gone
    results = store.search(query="שכח", top_k=10)
    chunk_ids = [r["id"] for r in results]
    assert "chunk-1" not in chunk_ids


def test_delete_nonexistent_book(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test deleting a book that doesn't exist."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index(sample_chunks)

    deleted_count = store.delete_by_book_id("nonexistent-book")
    assert deleted_count == 0


def test_delete_all_chunks(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test deleting all chunks from index."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index(sample_chunks)

    # Delete all books one by one
    for chunk in sample_chunks:
        store.delete_by_book_id(chunk.book_id)

    assert store.chunk_count == 0
    assert not store.is_loaded


def test_clear_index(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test clearing the in-memory index."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index(sample_chunks)
    store.save_index()

    store.clear()

    assert not store.is_loaded
    assert store.chunk_count == 0
    # Files should still exist
    assert (temp_bm25_dir / "bm25_index.pkl").exists()


def test_rebuild_from_all_chunks(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test rebuilding index from scratch."""
    store = BM25Store(bm25_dir=temp_bm25_dir)

    # Initial build
    store.build_index(sample_chunks[:2])
    assert store.chunk_count == 2

    # Rebuild with all chunks
    store.rebuild_from_all_chunks(sample_chunks)
    assert store.chunk_count == 4
    assert store.is_loaded

    # Verify saved
    assert (temp_bm25_dir / "bm25_index.pkl").exists()


def test_bm25_scoring_relevance(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test that BM25 scores reflect term relevance."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index(sample_chunks)

    # Query for a specific unique term
    results = store.search(query="משנה ברורה", top_k=5)

    assert len(results) > 0
    # Top result should be the chunk containing "משנה ברורה"
    assert results[0]["id"] == "chunk-2"
    # Scores should be in descending order
    if len(results) > 1:
        assert results[0]["score"] >= results[1]["score"]


def test_hebrew_tokenization(
    temp_bm25_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    """Test that Hebrew text is tokenized correctly."""
    store = BM25Store(bm25_dir=temp_bm25_dir)
    store.build_index(sample_chunks)

    # Search with Hebrew words
    results1 = store.search(query="הדלקת", top_k=5)
    results2 = store.search(query="נרות", top_k=5)

    # Both should return results
    assert len(results1) > 0
    assert len(results2) > 0

    # Should find chunks containing these terms
    assert any("הדלקת" in r["text"] for r in results1)
    assert any("נרות" in r["text"] for r in results2)
