"""Component: generated answer with inline citation highlighting."""

import re

import streamlit as st

from src.models.query_result import Citation, GeneratedAnswer, QueryResult

# Matches [1], [2], … [99] as used by the LLM
_CITATION_RE = re.compile(r"\[(\d+)\]")


def render_answer(query_result: QueryResult) -> None:
    """Render the answer section for a completed query.

    If an answer was generated, displays the answer text (with [N] markers
    styled as superscripts), a citation list, and a collapsed metadata
    expander. If generation failed (answer is None), shows a warning and
    an explanation so the user knows sources are still available below.

    Args:
        query_result: The completed QueryResult from the search pipeline.
    """
    st.markdown("---")
    st.markdown("#### 💬 Answer")

    if query_result.answer is None:
        _render_llm_unavailable_notice()
        return

    answer = query_result.answer
    _render_answer_text(answer)

    if answer.citations:
        _render_citations(answer.citations)

    _render_answer_metadata(answer)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _render_answer_text(answer: GeneratedAnswer) -> None:
    """Display the answer text with [N] markers as styled superscripts.

    Args:
        answer: The GeneratedAnswer whose text to render.
    """
    html_text = _highlight_citations(answer.text)
    st.markdown(
        f'<div class="answer-text">{html_text}</div>',
        unsafe_allow_html=True,
    )


def _render_citations(citations: list[Citation]) -> None:
    """Render the citation list beneath the answer text.

    Valid citations are shown as plain reference lines. Invalid citations
    (hallucinated indices) are highlighted with a warning colour.

    Args:
        citations: List of Citation objects parsed from the answer.
    """
    st.markdown("**📖 Citations**")

    for i, citation in enumerate(citations, start=1):
        if citation.is_valid:
            section = f" — {citation.section_path}" if citation.section_path else ""
            st.markdown(
                f'<div class="citation-item">'
                f"[{i}] {citation.book_title}{section}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="citation-item-invalid">'
                f"⚠️ [{i}] Source not found in provided results"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_llm_unavailable_notice() -> None:
    """Show a non-fatal warning when LLM generation failed."""
    st.warning(
        "⚠️ **LLM generation failed** — retrieved sources are shown below.\n\n"
        "Check that your API key is set in `.env` or try again later."
    )


def _render_answer_metadata(answer: GeneratedAnswer) -> None:
    """Show model name, token count, and latency inside a collapsed expander.

    Args:
        answer: The GeneratedAnswer whose metadata to display.
    """
    with st.expander("ℹ️ Generation metadata", expanded=False):
        cols = st.columns(3)
        cols[0].metric("Model", answer.model_used or "—")
        cols[1].metric("Tokens used", answer.tokens_used if answer.tokens_used else "—")
        latency_label = f"{answer.latency_ms} ms" if answer.latency_ms else "—"
        cols[2].metric("Latency", latency_label)


def _highlight_citations(text: str) -> str:
    """Replace ``[N]`` markers with styled HTML superscripts.

    Converts plain ``[1]`` citation markers into
    ``<sup class="citation">[1]</sup>`` so they render as small blue
    superscripts in the browser.

    Args:
        text: Raw answer text from the LLM.

    Returns:
        HTML string safe to pass to ``st.markdown(unsafe_allow_html=True)``.
    """
    # Escape any HTML that might be in the LLM response before injecting our tags
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    # Restore newlines as <br> for HTML rendering
    safe = safe.replace("\n", "<br>")
    return _CITATION_RE.sub(r'<sup class="citation">[\1]</sup>', safe)
