"""Tests for the structure-aware chunker."""

from pathlib import Path

import pytest

from src.config import ChunkingConfig
from src.ingestion.chunker import HalachicChunker, estimate_tokens
from src.ingestion.parser import BookParser
from src.models.parsed import ParsedBook

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_texts"
REAL_PDF_PATH = Path(__file__).parent.parent / "data" / "מנחת איש מסכת ברכות.pdf"


@pytest.fixture
def config() -> ChunkingConfig:
    return ChunkingConfig()


@pytest.fixture
def chunker(config: ChunkingConfig) -> HalachicChunker:
    return HalachicChunker(config=config)


def _make_parsed_book(text: str, title: str = "Test Book") -> ParsedBook:
    return ParsedBook(
        title=title,
        author="Test Author",
        language="he",
        raw_text=text,
        source_path="/test/book.txt",
        file_format="txt",
    )


# ── Token estimation ─────────────────────────────────────────────────────────


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_single_word(self) -> None:
        assert estimate_tokens("שלום") == 1

    def test_hebrew_sentence(self) -> None:
        result = estimate_tokens("כל אדם חייב לברך ברכת הנהנין")
        assert result == 6

    def test_mixed_language(self) -> None:
        result = estimate_tokens("Hello שלום World עולם")
        assert result == 4


# ── Structural chunking ─────────────────────────────────────────────────────


class TestHalachicChunkerStructural:
    def test_detects_siman_boundaries(self, chunker: HalachicChunker) -> None:
        text = (
            "סימן א\nסעיף א\nתוכן ראשון של סעיף א " * 10 + "\n\n"
            "סימן ב\nסעיף א\nתוכן שני של סעיף א " * 10
        )
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        paths = [c.section_path for c in chunks]
        assert any("סימן א" in p for p in paths)
        assert any("סימן ב" in p for p in paths)

    def test_detects_seif_within_siman(self, chunker: HalachicChunker) -> None:
        text = (
            "סימן א\n"
            "סעיף א\nתוכן של סעיף א עם מספיק מילים כדי ליצור חלק " * 5 + "\n"
            "סעיף ב\nתוכן של סעיף ב עם מספיק מילים כדי ליצור חלק " * 5
        )
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        paths = [c.section_path for c in chunks]
        assert any("סעיף א" in p for p in paths)
        assert any("סעיף ב" in p for p in paths)

    def test_section_path_hierarchy(self, chunker: HalachicChunker) -> None:
        text = (
            "סימן א\n"
            "סעיף א\nתוכן הלכתי חשוב מאד בנושא ברכות הנהנין " * 5
        )
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        assert any("סימן א > סעיף א" in c.section_path for c in chunks)

    def test_siman_katan_detection(self, chunker: HalachicChunker) -> None:
        text = (
            'סימן א\nס"ק א\nביאור ראשון של ההלכה בענייני ברכות ' * 5 + "\n"
            'ס"ק ב\nביאור שני של ההלכה בענייני ברכות הנהנין ' * 5
        )
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        paths = [c.section_path for c in chunks]
        assert any('ס"ק' in p or "ק א" in p for p in paths)

    def test_perek_as_top_level(self, chunker: HalachicChunker) -> None:
        text = (
            "פרק א\n"
            "תוכן של פרק ראשון עם מספיק מילים לבדוק את החלוקה " * 10
            + "\n"
            "פרק ב\n"
            "תוכן של פרק שני עם מספיק מילים לבדוק את החלוקה " * 10
        )
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        paths = [c.section_path for c in chunks]
        assert any("פרק א" in p for p in paths)
        assert any("פרק ב" in p for p in paths)

    def test_chunk_respects_max_tokens(self, chunker: HalachicChunker) -> None:
        # Create a very long section
        long_text = "סימן א\n" + "מילה " * 2000
        book = _make_parsed_book(long_text)
        chunks = chunker.chunk(book)
        for chunk in chunks:
            assert chunk.token_count <= chunker._config.max_tokens

    def test_section_type_populated(self, chunker: HalachicChunker) -> None:
        text = "סימן א\nסעיף א\nתוכן של סעיף " * 10
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        types = {c.section_type for c in chunks}
        assert len(types) > 0
        assert all(t != "" for t in types)

    def test_book_metadata_propagated(self, chunker: HalachicChunker) -> None:
        text = "סימן א\nתוכן הלכתי של הסימן הראשון " * 10
        book = _make_parsed_book(text, title="שולחן ערוך")
        chunks = chunker.chunk(book)
        for chunk in chunks:
            assert chunk.book_title == "שולחן ערוך"
            assert chunk.book_author == "Test Author"
            assert chunk.book_id  # UUID present

    def test_token_count_populated(self, chunker: HalachicChunker) -> None:
        text = "סימן א\nתוכן הלכתי חשוב בנושא ברכות " * 10
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_fixture_shulchan_aruch(self, chunker: HalachicChunker) -> None:
        text = (FIXTURES_DIR / "sample_shulchan_aruch.txt").read_text(encoding="utf-8")
        book = _make_parsed_book(text, title="שולחן ערוך")
        chunks = chunker.chunk(book)
        assert len(chunks) > 0
        paths = [c.section_path for c in chunks]
        assert any("סימן א" in p for p in paths)
        assert any("סימן ב" in p for p in paths)

    def test_fixture_mishna_berura(self, chunker: HalachicChunker) -> None:
        text = (FIXTURES_DIR / "sample_mishna_berura.txt").read_text(encoding="utf-8")
        book = _make_parsed_book(text, title="משנה ברורה")
        chunks = chunker.chunk(book)
        assert len(chunks) > 0


# ── Paragraph chunking ───────────────────────────────────────────────────────


class TestHalachicChunkerParagraph:
    def test_splits_on_double_newline(self, chunker: HalachicChunker) -> None:
        text = (
            "פסקה ראשונה עם מספיק מילים " * 10
            + "\n\n"
            + "פסקה שנייה עם מספיק מילים " * 10
        )
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        assert len(chunks) >= 2

    def test_merges_small_paragraphs(self) -> None:
        config = ChunkingConfig(min_tokens=50, max_tokens=800, target_tokens=450)
        chunker = HalachicChunker(config=config)
        # Each paragraph is ~4 tokens (well under min_tokens=50)
        text = "מילה אחת פה\n\nמילה שנייה פה\n\nמילה שלישית פה"
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        # Small paragraphs should be merged
        assert len(chunks) <= 2

    def test_splits_large_paragraphs(self) -> None:
        config = ChunkingConfig(max_tokens=100, target_tokens=50, overlap_tokens=10)
        chunker = HalachicChunker(config=config)
        # One huge paragraph
        text = "מילה " * 500
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_count <= config.max_tokens

    def test_fixture_unstructured(self, chunker: HalachicChunker) -> None:
        text = (FIXTURES_DIR / "sample_unstructured.txt").read_text(encoding="utf-8")
        book = _make_parsed_book(text, title="טקסט חופשי")
        chunks = chunker.chunk(book)
        assert len(chunks) > 0
        assert all(c.section_type == "paragraph" for c in chunks)


# ── Sliding window ───────────────────────────────────────────────────────────


class TestHalachicChunkerSlidingWindow:
    def test_single_block_of_text(self) -> None:
        config = ChunkingConfig(max_tokens=100, target_tokens=50, overlap_tokens=10)
        chunker = HalachicChunker(config=config)
        text = "מילה " * 300  # No paragraph breaks, no structure
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        assert len(chunks) > 1

    def test_window_size_respects_target(self) -> None:
        config = ChunkingConfig(max_tokens=100, target_tokens=50, overlap_tokens=10)
        chunker = HalachicChunker(config=config)
        text = "מילה " * 300
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        for chunk in chunks:
            assert chunk.token_count <= config.max_tokens

    def test_overlap_between_windows(self) -> None:
        config = ChunkingConfig(max_tokens=100, target_tokens=50, overlap_tokens=10)
        chunker = HalachicChunker(config=config)
        text = " ".join(f"מילה{i}" for i in range(200))
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        if len(chunks) >= 2:
            # Check that consecutive chunks share some words
            words_0 = set(chunks[0].text.split())
            words_1 = set(chunks[1].text.split())
            overlap = words_0 & words_1
            assert len(overlap) > 0


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestHalachicChunkerEdgeCases:
    def test_empty_text(self, chunker: HalachicChunker) -> None:
        book = _make_parsed_book("")
        chunks = chunker.chunk(book)
        assert chunks == []

    def test_whitespace_only(self, chunker: HalachicChunker) -> None:
        book = _make_parsed_book("   \n\n   \t  ")
        chunks = chunker.chunk(book)
        assert chunks == []

    def test_single_short_section(self, chunker: HalachicChunker) -> None:
        book = _make_parsed_book("סימן א\nתוכן קצר")
        chunks = chunker.chunk(book)
        assert len(chunks) >= 1

    def test_all_chunks_have_uuid(self, chunker: HalachicChunker) -> None:
        text = "סימן א\nתוכן " * 20 + "\nסימן ב\nתוכן " * 20
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))  # All unique

    def test_config_injection(self) -> None:
        custom = ChunkingConfig(target_tokens=100, max_tokens=200, min_tokens=10)
        chunker = HalachicChunker(config=custom)
        assert chunker._config.target_tokens == 100
        assert chunker._config.max_tokens == 200

    def test_language_propagated(self, chunker: HalachicChunker) -> None:
        book = ParsedBook(
            title="Test",
            language="mixed",
            raw_text="סימן א\nContent here " * 10,
            source_path="/test.txt",
            file_format="txt",
        )
        chunks = chunker.chunk(book)
        for chunk in chunks:
            assert chunk.language == "mixed"

    def test_chunk_index_and_total(self, chunker: HalachicChunker) -> None:
        text = "סימן א\n" + "מילה " * 2000
        book = _make_parsed_book(text)
        chunks = chunker.chunk(book)
        if len(chunks) > 1:
            # All chunks with same section_path should have sequential indices
            for chunk in chunks:
                assert chunk.chunk_index >= 0
                assert chunk.total_chunks_in_section > 0
                assert chunk.chunk_index < chunk.total_chunks_in_section

    def test_book_id_can_be_injected(self, chunker: HalachicChunker) -> None:
        book = _make_parsed_book("סימן א\nתוכן " * 10)
        chunks = chunker.chunk(book, book_id="custom-id-123")
        for chunk in chunks:
            assert chunk.book_id == "custom-id-123"


# ── Integration with real PDF ────────────────────────────────────────────────


class TestChunkerRealPdf:
    @pytest.mark.skipif(
        not REAL_PDF_PATH.exists(),
        reason="Real PDF not available",
    )
    def test_chunk_real_pdf(self, chunker: HalachicChunker) -> None:
        parser = BookParser()
        book = parser.parse(REAL_PDF_PATH)
        chunks = chunker.chunk(book)

        assert len(chunks) > 100  # 880-page book should produce many chunks
        assert any("פרק" in c.section_path for c in chunks)
        # No chunk should exceed max_tokens
        for chunk in chunks:
            assert chunk.token_count <= chunker._config.max_tokens
            assert chunk.book_title
            assert chunk.language == "he"
