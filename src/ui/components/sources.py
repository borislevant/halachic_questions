"""Component: expandable source cards for retrieved chunks."""

import streamlit as st

from src.models.query_result import RetrievalResult

# Hebrew Unicode block: U+0590–U+05FF
_HEB_START = 0x0590
_HEB_END = 0x05FF


def render_sources(sources: list[RetrievalResult]) -> None:
    """Render all retrieved sources as expandable cards.

    Each card shows the book title, section path, and similarity score
    in the header. Expanding reveals the full chunk text, and
    context-before / context-after if present.

    Args:
        sources: Ordered list of RetrievalResult objects to display.
    """
    if not sources:
        return

    st.markdown("---")
    st.markdown(f"#### 🔍 Sources ({len(sources)})")

    for i, result in enumerate(sources, start=1):
        _render_source_card(i, result)


def _render_source_card(index: int, result: RetrievalResult) -> None:
    """Render a single expandable source card.

    Args:
        index: 1-based display index matching the citation number used
            in the LLM answer.
        result: The retrieval result to display.
    """
    chunk = result.chunk

    # Build the header line shown in the collapsed expander
    rr = result.rerank_score
    score = rr if rr is not None else result.similarity_score
    score_html = f'<span class="score-badge">{score:.2f}</span>'
    section = f" — {chunk.section_path}" if chunk.section_path else ""
    header = f"[{index}] {chunk.book_title}{section}"

    with st.expander(header, expanded=False):
        # Score badge
        st.markdown(
            f'Relevance score: {score_html}',
            unsafe_allow_html=True,
        )

        # Context before
        if result.context_before:
            st.markdown(
                f'<div class="context-line">↑ {result.context_before.strip()}</div>',
                unsafe_allow_html=True,
            )

        # Main chunk text — RTL if Hebrew
        text = chunk.text.strip()
        if _is_hebrew(text):
            st.markdown(
                f'<div class="rtl-text">{text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(text)

        # Context after
        if result.context_after:
            st.markdown(
                f'<div class="context-line">↓ {result.context_after.strip()}</div>',
                unsafe_allow_html=True,
            )


def _is_hebrew(text: str) -> bool:
    """Return True when the majority of alphabetic chars are Hebrew.

    Args:
        text: The text to inspect.

    Returns:
        True if more than half of the non-whitespace characters fall in
        the Hebrew Unicode block (U+0590–U+05FF).
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    heb_count = sum(1 for c in chars if _HEB_START <= ord(c) <= _HEB_END)
    return heb_count / len(chars) > 0.5
