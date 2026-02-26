"""Citation extraction and validation for LLM-generated answers."""

import re
from typing import ClassVar

from src.models.query_result import Citation, RetrievalResult


class CitationParser:
    """Parses and validates inline [N] citations from LLM-generated text.

    The LLM is instructed to cite sources using [N] notation where N is the
    1-based index into the provided sources list. This class extracts those
    markers, deduplicates them, and maps each to a Citation object.
    Citations referencing indices outside the source list are marked invalid.
    """

    _CITATION_RE: ClassVar[re.Pattern[str]] = re.compile(r"\[(\d+)\]")

    def parse(
        self,
        answer_text: str,
        sources: list[RetrievalResult],
    ) -> list[Citation]:
        """Extract and validate citations from LLM answer text.

        Args:
            answer_text: Raw text returned by the LLM.
            sources: The ordered list of RetrievalResult objects that were
                provided to the LLM, used for index-to-chunk mapping.

        Returns:
            Deduplicated list of Citation objects in the order they first
            appear in the answer text. Invalid indices produce a Citation
            with is_valid=False.
        """
        indices = self._extract_indices(answer_text)
        seen: set[int] = set()
        citations: list[Citation] = []

        for index in indices:
            if index in seen:
                continue
            seen.add(index)
            citations.append(self._build_citation(index, sources))

        return citations

    def _extract_indices(self, text: str) -> list[int]:
        """Return citation indices in the order they appear in text.

        Args:
            text: The answer text to scan.

        Returns:
            List of 1-based integer indices (may contain duplicates).
        """
        return [int(m) for m in self._CITATION_RE.findall(text)]

    def _build_citation(
        self,
        index: int,
        sources: list[RetrievalResult],
    ) -> Citation:
        """Build a Citation for a 1-based source index.

        Args:
            index: 1-based citation index as used by the LLM.
            sources: The sources list passed to the LLM.

        Returns:
            Citation with is_valid=True when the index is in range,
            or is_valid=False when the LLM hallucinated an index.
        """
        if 1 <= index <= len(sources):
            chunk = sources[index - 1].chunk
            return Citation(
                book_title=chunk.book_title,
                section_path=chunk.section_path,
                source_chunk_id=chunk.id,
                is_valid=True,
            )

        return Citation(
            book_title="Unknown",
            section_path="",
            source_chunk_id="",
            is_valid=False,
        )
