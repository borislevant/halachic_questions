"""Structure-aware text chunker for Halachic texts."""

import logging
import re
from uuid import uuid4

from src.config import ChunkingConfig
from src.models.chunk import Chunk
from src.models.parsed import ParsedBook, Section

logger = logging.getLogger(__name__)

# Hebrew structure patterns ordered by hierarchy level (highest to lowest).
# Compiled with MULTILINE so ^ matches start of each line.
STRUCTURE_PATTERNS: dict[str, re.Pattern[str]] = {
    "perek": re.compile(r"^\s*פרק\s+[א-ת]{1,4}\s*$", re.MULTILINE),
    "siman": re.compile(r"^\s*סימן\s+[א-ת]{1,4}", re.MULTILINE),
    "seif": re.compile(r"^\s*סעיף\s+[א-ת]{1,4}", re.MULTILINE),
    "halacha": re.compile(r"^\s*\.?([א-ת]{1,3})\.\s*$", re.MULTILINE),
    "siman_katan": re.compile(r'ס["\u05F4\u05F3]{1,2}ק\s+[א-ת]{1,4}', re.MULTILINE),
}

# Hierarchy levels: lower number = higher (more general) level
HIERARCHY_LEVELS: dict[str, int] = {
    "perek": 0,
    "siman": 1,
    "halacha": 2,
    "seif": 3,
    "siman_katan": 4,
    "paragraph": 5,
}


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Uses word-splitting as a proxy. For Hebrew text, this provides
    a reasonable approximation (roughly 1 token per word).

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    return len(text.split())


class HalachicChunker:
    """Splits parsed books into semantically meaningful chunks.

    Chunking strategy (priority order):
    1. Structural: Split on detected section boundaries
       (perek, siman, seif, halacha, siman_katan)
    2. Paragraph: If no structure detected, split on paragraph boundaries
    3. Sliding window: For long unstructured text, use token-based
       sliding window with overlap

    Args:
        config: ChunkingConfig with target_tokens, max_tokens,
                min_tokens, and overlap_tokens settings.
    """

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config

    def chunk(self, parsed_book: ParsedBook, book_id: str | None = None) -> list[Chunk]:
        """Split a parsed book into chunks.

        Args:
            parsed_book: The parsed book to chunk.
            book_id: Optional pre-generated book ID. If None, a UUID is generated.

        Returns:
            List of Chunk objects with metadata populated.
        """
        effective_book_id = book_id or str(uuid4())
        text = parsed_book.raw_text

        if not text.strip():
            return []

        # Try structural chunking first
        sections = self._detect_sections(text)
        if sections:
            chunks = self._chunk_sections(
                sections=sections,
                book_id=effective_book_id,
                book_title=parsed_book.title,
                book_author=parsed_book.author,
                language=parsed_book.language,
            )
        else:
            # Fall back to paragraph chunking
            chunks = self._chunk_by_paragraphs(
                text=text,
                book_id=effective_book_id,
                book_title=parsed_book.title,
                book_author=parsed_book.author,
                language=parsed_book.language,
            )

        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)

        # Set chunk_index and total_chunks_in_section
        self._assign_indices(chunks)

        return chunks

    def _detect_sections(self, text: str) -> list[Section]:
        """Detect structural sections in the raw text.

        Scans for all structure patterns, sorts matches by position,
        and builds a nested section tree based on hierarchy levels.

        Args:
            text: The raw text to analyze.

        Returns:
            List of top-level Section objects (with subsections nested).
        """
        markers: list[dict[str, str | int]] = []

        for section_type, pattern in STRUCTURE_PATTERNS.items():
            for match in pattern.finditer(text):
                markers.append({
                    "type": section_type,
                    "title": match.group().strip().rstrip(".").lstrip(".").strip(),
                    "position": match.start(),
                    "level": HIERARCHY_LEVELS[section_type],
                })

        if not markers:
            return []

        # Sort by position in text
        markers.sort(key=lambda m: m["position"])  # type: ignore[arg-type]

        return self._build_section_tree(markers, text)

    def _build_section_tree(
        self, markers: list[dict[str, str | int]], text: str
    ) -> list[Section]:
        """Build a nested Section tree from a flat list of markers.

        Uses a stack-based approach: when encountering a marker at
        level N, pop all stack entries at level >= N, then push.

        Args:
            markers: Sorted list of detected section markers.
            text: Full raw text for extracting section content.

        Returns:
            List of root-level Section objects.
        """
        roots: list[Section] = []
        stack: list[Section] = []

        for i, marker in enumerate(markers):
            # Determine text range: from this marker to the next marker (or end)
            start = int(marker["position"])
            if i + 1 < len(markers):
                end = int(markers[i + 1]["position"])
            else:
                end = len(text)

            section_text = text[start:end].strip()
            section_type = str(marker["type"])
            level = int(marker["level"])

            section = Section(
                section_type=section_type,
                title=str(marker["title"]),
                text=section_text,
                char_start=start,
                char_end=end,
                level=level,
            )

            # Pop stack entries at same or lower level
            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                stack[-1].subsections.append(section)
            else:
                roots.append(section)

            stack.append(section)

        return roots

    def _chunk_sections(
        self,
        sections: list[Section],
        book_id: str,
        book_title: str,
        book_author: str,
        language: str,
        parent_path: str = "",
    ) -> list[Chunk]:
        """Recursively chunk a list of sections.

        For each section:
        - If the section has subsections, recurse into them
        - If the section text fits in one chunk, create one Chunk
        - If the section text exceeds max_tokens, split with sliding window

        Args:
            sections: List of Section objects to chunk.
            book_id: The book's UUID.
            book_title: Title for chunk metadata.
            book_author: Author for chunk metadata.
            language: Language code.
            parent_path: Section path prefix from parent sections.

        Returns:
            Flat list of Chunk objects.
        """
        chunks: list[Chunk] = []

        for section in sections:
            section_path = self._build_section_path(parent_path, section)

            if section.subsections:
                # Recurse into subsections
                chunks.extend(
                    self._chunk_sections(
                        sections=section.subsections,
                        book_id=book_id,
                        book_title=book_title,
                        book_author=book_author,
                        language=language,
                        parent_path=section_path,
                    )
                )
            else:
                # Leaf section — chunk its text
                section_text = section.text
                token_count = estimate_tokens(section_text)

                if token_count <= self._config.max_tokens:
                    if section_text.strip():
                        chunks.append(
                            Chunk(
                                text=section_text,
                                book_id=book_id,
                                book_title=book_title,
                                book_author=book_author,
                                section_path=section_path,
                                section_type=section.section_type,
                                language=language,
                                char_start=section.char_start,
                                char_end=section.char_end,
                                token_count=token_count,
                            )
                        )
                else:
                    # Section too large — use sliding window
                    chunks.extend(
                        self._sliding_window_chunks(
                            text=section_text,
                            book_id=book_id,
                            book_title=book_title,
                            book_author=book_author,
                            language=language,
                            section_path=section_path,
                            section_type=section.section_type,
                            char_offset=section.char_start,
                        )
                    )

        return chunks

    def _chunk_by_paragraphs(
        self,
        text: str,
        book_id: str,
        book_title: str,
        book_author: str,
        language: str,
        section_path: str = "",
        char_offset: int = 0,
    ) -> list[Chunk]:
        """Split text by paragraph boundaries (double newlines).

        Merges small consecutive paragraphs that are under min_tokens.
        Splits paragraphs that exceed max_tokens using sliding window.

        Args:
            text: The text to split.
            book_id: Book UUID.
            book_title: Title for metadata.
            book_author: Author for metadata.
            language: Language code.
            section_path: Section path for metadata.
            char_offset: Offset in original text for char_start/char_end.

        Returns:
            List of Chunk objects.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[Chunk] = []
        current_pos = 0

        for para in paragraphs:
            para_stripped = para.strip()
            if not para_stripped:
                current_pos += len(para) + 1
                continue

            # Find the actual position of this paragraph in the text
            para_start = text.find(para, current_pos)
            if para_start == -1:
                para_start = current_pos
            para_end = para_start + len(para)
            current_pos = para_end

            token_count = estimate_tokens(para_stripped)

            if token_count > self._config.max_tokens:
                # Paragraph too large — use sliding window
                chunks.extend(
                    self._sliding_window_chunks(
                        text=para_stripped,
                        book_id=book_id,
                        book_title=book_title,
                        book_author=book_author,
                        language=language,
                        section_path=section_path,
                        section_type="paragraph",
                        char_offset=char_offset + para_start,
                    )
                )
            else:
                chunks.append(
                    Chunk(
                        text=para_stripped,
                        book_id=book_id,
                        book_title=book_title,
                        book_author=book_author,
                        section_path=section_path,
                        section_type="paragraph",
                        language=language,
                        char_start=char_offset + para_start,
                        char_end=char_offset + para_end,
                        token_count=token_count,
                    )
                )

        return chunks

    def _sliding_window_chunks(
        self,
        text: str,
        book_id: str,
        book_title: str,
        book_author: str,
        language: str,
        section_path: str = "",
        section_type: str = "paragraph",
        char_offset: int = 0,
    ) -> list[Chunk]:
        """Split text using a token-based sliding window.

        Creates chunks of target_tokens size with overlap_tokens overlap.
        Breaks at word boundaries.

        Args:
            text: The text to split.
            book_id: Book UUID.
            book_title: Title for metadata.
            book_author: Author for metadata.
            language: Language code.
            section_path: Section path for metadata.
            section_type: Type label for metadata.
            char_offset: Offset in original text.

        Returns:
            List of Chunk objects.
        """
        words = text.split()
        if not words:
            return []

        if len(words) <= self._config.max_tokens:
            return [
                Chunk(
                    text=text.strip(),
                    book_id=book_id,
                    book_title=book_title,
                    book_author=book_author,
                    section_path=section_path,
                    section_type=section_type,
                    language=language,
                    char_start=char_offset,
                    char_end=char_offset + len(text),
                    token_count=len(words),
                )
            ]

        chunks: list[Chunk] = []
        target = self._config.target_tokens
        overlap = self._config.overlap_tokens
        step = max(target - overlap, 1)
        pos = 0

        while pos < len(words):
            end = min(pos + target, len(words))
            chunk_words = words[pos:end]
            chunk_text = " ".join(chunk_words)

            # Approximate character positions
            char_start = char_offset + text.find(chunk_words[0], 0 if pos == 0 else 0)
            # Simpler: use proportional estimation
            char_start = char_offset + int(pos / len(words) * len(text))
            char_end = char_offset + int(end / len(words) * len(text))

            chunks.append(
                Chunk(
                    text=chunk_text,
                    book_id=book_id,
                    book_title=book_title,
                    book_author=book_author,
                    section_path=section_path,
                    section_type=section_type,
                    language=language,
                    char_start=char_start,
                    char_end=char_end,
                    token_count=len(chunk_words),
                )
            )

            if end >= len(words):
                break
            pos += step

        return chunks

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge consecutive chunks that are below min_tokens.

        Adjacent chunks with the same section_path and combined token count
        under max_tokens are merged into a single chunk.

        Args:
            chunks: List of chunks, some potentially too small.

        Returns:
            List of chunks with small ones merged.
        """
        if not chunks:
            return []

        merged: list[Chunk] = [chunks[0]]

        for chunk in chunks[1:]:
            prev = merged[-1]
            combined_tokens = prev.token_count + chunk.token_count

            if (
                prev.token_count < self._config.min_tokens
                and combined_tokens <= self._config.max_tokens
                and prev.section_path == chunk.section_path
            ):
                # Merge into previous
                merged[-1] = Chunk(
                    text=prev.text + "\n\n" + chunk.text,
                    book_id=prev.book_id,
                    book_title=prev.book_title,
                    book_author=prev.book_author,
                    section_path=prev.section_path,
                    section_type=prev.section_type,
                    language=prev.language,
                    char_start=prev.char_start,
                    char_end=chunk.char_end,
                    token_count=combined_tokens,
                )
            else:
                merged.append(chunk)

        return merged

    def _assign_indices(self, chunks: list[Chunk]) -> None:
        """Assign chunk_index and total_chunks_in_section.

        Groups chunks by section_path and sets sequential indices.

        Args:
            chunks: List of chunks to update in place.
        """
        # Group by section_path
        groups: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            groups.setdefault(chunk.section_path, []).append(chunk)

        for path_chunks in groups.values():
            total = len(path_chunks)
            for i, chunk in enumerate(path_chunks):
                chunk.chunk_index = i
                chunk.total_chunks_in_section = total

    def _build_section_path(self, parent_path: str, section: Section) -> str:
        """Construct a hierarchical section path string.

        Args:
            parent_path: Path from parent sections.
            section: Current section.

        Returns:
            Combined path string (e.g. "פרק א > סעיף ב").
        """
        if parent_path:
            return f"{parent_path} > {section.title}"
        return section.title
