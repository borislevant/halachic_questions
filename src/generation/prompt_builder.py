"""Prompt builder for Halachic Q&A generation."""

from src.models.query_result import RetrievalResult

_SYSTEM_PROMPT = """\
You are a Halachic research assistant. Answer the user's question using ONLY the \
provided sources. For every claim you make, cite the source number inline using [N] notation.

Rules:
- Use ONLY the information found in the provided sources.
- If the sources do not contain sufficient information to answer the question, say so explicitly.
- Preserve the original Hebrew or Aramaic text when quoting directly.
- Do NOT invent rulings, authorities, page numbers, or any information not present in the sources.
- Every factual statement must be followed by at least one citation [N].
- This answer is for research purposes only and does not replace a ruling from a qualified Rabbi.\
"""


class PromptBuilder:
    """Builds LLM prompts grounded in retrieved Halachic sources.

    Formats a numbered source list and a user question into a
    (system_prompt, user_prompt) pair ready for submission to an LLM.
    """

    def build(
        self,
        question: str,
        sources: list[RetrievalResult],
    ) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for the given question and sources.

        Args:
            question: The user's Halachic question.
            sources: Ranked retrieval results to ground the answer in.

        Returns:
            A tuple of (system_prompt, user_prompt) strings.
        """
        user_prompt = self._build_user_prompt(question, sources)
        return _SYSTEM_PROMPT, user_prompt

    def _build_user_prompt(
        self,
        question: str,
        sources: list[RetrievalResult],
    ) -> str:
        """Compose the user-turn prompt with sources and the question.

        Args:
            question: The user's question.
            sources: Retrieved sources to include.

        Returns:
            Formatted user prompt string.
        """
        parts: list[str] = []

        if sources:
            parts.append("Sources:\n")
            for i, result in enumerate(sources, start=1):
                parts.append(self._format_single_source(i, result))
        else:
            parts.append(
                "Note: No relevant sources were found in the library for this question.\n"
            )

        parts.append("---")
        parts.append(f"Question: {question}")

        return "\n".join(parts)

    def _format_single_source(
        self,
        index: int,
        result: RetrievalResult,
    ) -> str:
        """Format a single source entry.

        Includes book title, section path, optional surrounding context,
        and the chunk's main text.

        Args:
            index: 1-based source number used in [N] citations.
            result: RetrievalResult containing the chunk and context.

        Returns:
            Formatted source block string.
        """
        chunk = result.chunk
        lines: list[str] = []

        header_parts = [f"[{index}] {chunk.book_title}"]
        if chunk.section_path:
            header_parts.append(chunk.section_path)
        lines.append(" — ".join(header_parts))

        if result.context_before:
            lines.append(f"(context: {result.context_before.strip()})")

        lines.append(chunk.text.strip())

        if result.context_after:
            lines.append(f"(context: {result.context_after.strip()})")

        lines.append("")  # blank line between sources
        return "\n".join(lines)
