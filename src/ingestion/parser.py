"""Book file parser supporting PDF, TXT, DOCX, and HTML formats."""

import logging
import re
from pathlib import Path

import chardet

from src.models.parsed import ParsedBook

logger = logging.getLogger(__name__)

# Supported file extensions mapped to format identifiers
SUPPORTED_FORMATS: dict[str, str] = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".md": "txt",
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
}


class BookParser:
    """Parses book files into a structured ParsedBook representation.

    Supports PDF, TXT/Markdown, DOCX, and HTML formats.
    Handles Hebrew, Aramaic, English, and mixed-language texts.
    """

    def parse(self, file_path: str | Path) -> ParsedBook:
        """Parse a book file into a ParsedBook structure.

        Args:
            file_path: Path to the book file.

        Returns:
            A ParsedBook containing raw text and metadata.

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If the file format is not supported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_format = self._detect_format(path)

        dispatch = {
            "pdf": self._parse_pdf,
            "txt": self._parse_txt,
            "docx": self._parse_docx,
            "html": self._parse_html,
        }
        raw_text = dispatch[file_format](path)

        language = self._detect_language(raw_text)
        title = self._extract_title_from_text(raw_text, path)

        return ParsedBook(
            title=title,
            author="",
            language=language,
            raw_text=raw_text,
            sections=[],
            source_path=str(path),
            file_format=file_format,
        )

    def _detect_format(self, file_path: Path) -> str:
        """Determine file format from extension.

        Args:
            file_path: Path to the file.

        Returns:
            Format string ("pdf", "txt", "docx", "html").

        Raises:
            ValueError: If extension is not supported.
        """
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: '{ext}'. "
                f"Supported: {', '.join(SUPPORTED_FORMATS.keys())}"
            )
        return SUPPORTED_FORMATS[ext]

    def _parse_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file using pymupdf (fitz).

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted raw text with pages separated by newlines.
        """
        import fitz  # type: ignore[import-untyped]

        try:
            with fitz.open(str(file_path)) as doc:
                pages = []
                for page in doc:
                    text = page.get_text("text")
                    if text.strip():
                        pages.append(text)
                return "\n".join(pages)
        except Exception:
            logger.exception("Failed to parse PDF: %s", file_path)
            return ""

    def _parse_txt(self, file_path: Path) -> str:
        """Read a plain text or Markdown file with encoding detection.

        Tries UTF-8 first, then uses chardet for fallback detection.
        Handles UTF-8, UTF-16, and Windows-1255 encodings.

        Args:
            file_path: Path to the text file.

        Returns:
            The file content as a string.
        """
        # Try UTF-8 first
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            pass

        # Fallback to encoding detection
        raw_bytes = file_path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"
        confidence = detected.get("confidence", 0)

        if confidence < 0.7:
            logger.warning(
                "Low confidence encoding detection for %s: %s (%.0f%%)",
                file_path,
                encoding,
                confidence * 100,
            )

        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # Last resort: try windows-1255 (common Hebrew encoding)
            try:
                return raw_bytes.decode("windows-1255")
            except UnicodeDecodeError:
                logger.error("Failed to decode file: %s", file_path)
                return raw_bytes.decode("utf-8", errors="replace")

    def _parse_docx(self, file_path: Path) -> str:
        """Extract text from a DOCX file using python-docx.

        Preserves paragraph boundaries as double newlines.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            Extracted text with paragraph structure preserved.
        """
        import docx

        try:
            doc = docx.Document(str(file_path))
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception:
            logger.exception("Failed to parse DOCX: %s", file_path)
            return ""

    def _parse_html(self, file_path: Path) -> str:
        """Extract text from an HTML file using BeautifulSoup.

        Strips all HTML tags, scripts, and styles, preserving text content.

        Args:
            file_path: Path to the HTML file.

        Returns:
            Clean extracted text.
        """
        from bs4 import BeautifulSoup

        try:
            raw = self._parse_txt(file_path)
            soup = BeautifulSoup(raw, "lxml")

            # Remove script and style elements
            for tag in soup(["script", "style"]):
                tag.decompose()

            return soup.get_text(separator="\n")
        except Exception:
            logger.exception("Failed to parse HTML: %s", file_path)
            return ""

    def _detect_language(self, text: str) -> str:
        """Detect the primary language of the text.

        Analyzes character distribution to classify as Hebrew, English,
        or mixed.

        Args:
            text: The text to analyze.

        Returns:
            Language code: "he", "en", or "mixed".
        """
        if not text.strip():
            return "he"

        hebrew_count = len(re.findall(r"[\u0590-\u05FF]", text))
        latin_count = len(re.findall(r"[a-zA-Z]", text))
        total = hebrew_count + latin_count

        if total == 0:
            return "he"

        hebrew_ratio = hebrew_count / total
        if hebrew_ratio > 0.6:
            return "he"
        if hebrew_ratio < 0.4:
            return "en"
        return "mixed"

    def _extract_title_from_text(self, text: str, file_path: Path) -> str:
        """Attempt to extract a title from the first lines of text.

        Falls back to the filename (without extension) if no suitable
        title line is detected.

        Args:
            text: The raw extracted text.
            file_path: Path to the source file (for fallback).

        Returns:
            Best-guess title string.
        """
        if not text.strip():
            return file_path.stem

        lines = text.strip().split("\n")
        for line in lines[:5]:
            stripped = line.strip()
            # A good title candidate: short, non-empty, mostly letters
            if stripped and len(stripped) <= 100:
                alpha_chars = len(re.findall(r"[\u0590-\u05FFa-zA-Z]", stripped))
                if alpha_chars > 0 and alpha_chars / max(len(stripped), 1) > 0.5:
                    return stripped

        return file_path.stem
