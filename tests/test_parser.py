"""Tests for the book parser."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.parser import BookParser, SUPPORTED_FORMATS

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_texts"
REAL_PDF_PATH = Path(__file__).parent.parent / "data" / "מנחת איש מסכת ברכות.pdf"


@pytest.fixture
def parser() -> BookParser:
    return BookParser()


class TestBookParserTxt:
    """Tests for plain text file parsing."""

    def test_parse_utf8_file(self, parser: BookParser, tmp_path: Path) -> None:
        content = "סימן א\nסעיף א\nהלכה ראשונה בעניין ברכות"
        f = tmp_path / "test.txt"
        f.write_text(content, encoding="utf-8")

        result = parser.parse(f)
        assert "סימן א" in result.raw_text
        assert result.file_format == "txt"

    def test_parse_windows_1255_file(self, parser: BookParser, tmp_path: Path) -> None:
        content = "שלום עולם"
        f = tmp_path / "test.txt"
        f.write_bytes(content.encode("windows-1255"))

        result = parser.parse(f)
        assert "שלום" in result.raw_text

    def test_parse_utf16_file(self, parser: BookParser, tmp_path: Path) -> None:
        content = "הלכות ברכות הנהנין"
        f = tmp_path / "test.txt"
        f.write_bytes(content.encode("utf-16"))

        result = parser.parse(f)
        assert "ברכות" in result.raw_text

    def test_parse_empty_file(self, parser: BookParser, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")

        result = parser.parse(f)
        assert result.raw_text == ""
        assert result.file_format == "txt"

    def test_parse_fixture_shulchan_aruch(self, parser: BookParser) -> None:
        result = parser.parse(FIXTURES_DIR / "sample_shulchan_aruch.txt")
        assert "סימן א" in result.raw_text
        assert "סעיף א" in result.raw_text
        assert result.language == "he"


class TestBookParserPdf:
    """Tests for PDF parsing (mocked fitz)."""

    def test_parse_pdf_extracts_text(self, parser: BookParser, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        mock_page = MagicMock()
        mock_page.get_text.return_value = "סימן א\nסעיף א\nהלכה ראשונה"

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.__enter__ = lambda self: self
        mock_doc.__exit__ = MagicMock(return_value=False)

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = parser.parse(pdf_path)

        assert "סימן א" in result.raw_text
        assert result.file_format == "pdf"

    def test_parse_pdf_corrupt_logs_error(
        self, parser: BookParser, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pdf_path = tmp_path / "corrupt.pdf"
        pdf_path.touch()

        mock_fitz = MagicMock()
        mock_fitz.open.side_effect = RuntimeError("Corrupt PDF")

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = parser.parse(pdf_path)

        assert result.raw_text == ""
        assert "Failed to parse PDF" in caplog.text


class TestBookParserDocx:
    """Tests for DOCX parsing (mocked python-docx)."""

    def test_parse_docx_extracts_paragraphs(
        self, parser: BookParser, tmp_path: Path
    ) -> None:
        docx_path = tmp_path / "test.docx"
        docx_path.touch()

        mock_para1 = MagicMock()
        mock_para1.text = "פרק ראשון"
        mock_para2 = MagicMock()
        mock_para2.text = "הלכה א: דין ברכות"
        mock_para_empty = MagicMock()
        mock_para_empty.text = "  "

        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para1, mock_para_empty, mock_para2]

        mock_docx_mod = MagicMock()
        mock_docx_mod.Document.return_value = mock_doc

        with patch.dict("sys.modules", {"docx": mock_docx_mod}):
            result = parser.parse(docx_path)

        assert "פרק ראשון" in result.raw_text
        assert "הלכה א" in result.raw_text
        # Empty paragraphs should be filtered
        assert result.raw_text.count("\n\n") == 1
        assert result.file_format == "docx"


class TestBookParserHtml:
    """Tests for HTML parsing."""

    def test_parse_html_strips_tags(self, parser: BookParser, tmp_path: Path) -> None:
        html = "<html><body><p>הלכות ברכות</p><p>סימן א</p></body></html>"
        f = tmp_path / "test.html"
        f.write_text(html, encoding="utf-8")

        result = parser.parse(f)
        assert "הלכות ברכות" in result.raw_text
        assert "<p>" not in result.raw_text
        assert result.file_format == "html"

    def test_parse_html_removes_scripts(
        self, parser: BookParser, tmp_path: Path
    ) -> None:
        html = (
            "<html><head><script>alert('x')</script></head>"
            "<body><style>.x{}</style><p>תוכן חשוב</p></body></html>"
        )
        f = tmp_path / "test.html"
        f.write_text(html, encoding="utf-8")

        result = parser.parse(f)
        assert "תוכן חשוב" in result.raw_text
        assert "alert" not in result.raw_text
        assert ".x{}" not in result.raw_text


class TestDetectFormat:
    """Tests for format detection from file extension."""

    def test_supported_extensions(self, parser: BookParser) -> None:
        for ext, fmt in SUPPORTED_FORMATS.items():
            result = parser._detect_format(Path(f"book{ext}"))
            assert result == fmt

    def test_unsupported_extension_raises(self, parser: BookParser) -> None:
        with pytest.raises(ValueError, match="Unsupported file format"):
            parser._detect_format(Path("book.xyz"))

    def test_case_insensitive(self, parser: BookParser) -> None:
        assert parser._detect_format(Path("book.PDF")) == "pdf"
        assert parser._detect_format(Path("book.Txt")) == "txt"
        assert parser._detect_format(Path("book.HTML")) == "html"


class TestDetectLanguage:
    """Tests for language detection."""

    def test_hebrew_text(self, parser: BookParser) -> None:
        assert parser._detect_language("שלום עולם ברוך הבא") == "he"

    def test_english_text(self, parser: BookParser) -> None:
        assert parser._detect_language("Hello world this is English text") == "en"

    def test_mixed_text(self, parser: BookParser) -> None:
        result = parser._detect_language("שלום Hello עולם World")
        assert result == "mixed"

    def test_empty_text(self, parser: BookParser) -> None:
        assert parser._detect_language("") == "he"
        assert parser._detect_language("   ") == "he"


class TestExtractTitle:
    """Tests for title extraction."""

    def test_title_from_first_line(self, parser: BookParser) -> None:
        text = "מנחת איש\nהלכות ברכות"
        title = parser._extract_title_from_text(text, Path("book.pdf"))
        assert title == "מנחת איש"

    def test_title_fallback_to_filename(self, parser: BookParser) -> None:
        text = ""
        title = parser._extract_title_from_text(text, Path("my_book.pdf"))
        assert title == "my_book"

    def test_file_format_populated(self, parser: BookParser, tmp_path: Path) -> None:
        f = tmp_path / "book.txt"
        f.write_text("תוכן", encoding="utf-8")
        result = parser.parse(f)
        assert result.file_format == "txt"


class TestParserErrors:
    """Tests for error handling."""

    def test_nonexistent_file_raises(self, parser: BookParser) -> None:
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.txt")

    def test_unsupported_format_raises(
        self, parser: BookParser, tmp_path: Path
    ) -> None:
        f = tmp_path / "book.xyz"
        f.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse(f)


class TestParserRealPdf:
    """Integration tests with the real מנחת איש PDF."""

    @pytest.mark.skipif(
        not REAL_PDF_PATH.exists(),
        reason="Real PDF not available",
    )
    def test_parse_real_pdf(self, parser: BookParser) -> None:
        result = parser.parse(REAL_PDF_PATH)
        assert result.language == "he"
        assert result.file_format == "pdf"
        assert len(result.raw_text) > 100_000
        assert "פרק" in result.raw_text
        assert "ברכ" in result.raw_text
