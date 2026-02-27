"""Main Streamlit application for Halachic Q&A."""

from pathlib import Path

import streamlit as st

from src.generation.summarizer import Summarizer
from src.models.query_result import QueryResult
from src.retrieval.retriever import Retriever
from src.ui.components.answer import render_answer
from src.ui.components.book_list import render_book_list
from src.ui.components.ingestion import render_ingestion
from src.ui.components.sources import render_sources
from src.ui.services import load_services

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Halachic Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_css() -> None:
    """Inject the project stylesheet into the Streamlit page."""
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def _init_session_state() -> None:
    """Initialise session state keys on first run."""
    defaults: dict = {
        "query_result": None,   # QueryResult | None
        "is_searching": False,  # bool — guard against double-submit
        "search_error": None,   # str | None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _handle_search(
    question: str,
    retriever: Retriever,
    summarizer: Summarizer,
) -> None:
    """Run retrieval + generation and store the result in session state.

    Clears any previous result and error before running. On success,
    ``st.session_state.query_result`` is set to a populated QueryResult.
    On failure, ``st.session_state.search_error`` is set instead.

    Args:
        question: The user's question text.
        retriever: Configured Retriever instance.
        summarizer: Configured Summarizer instance.
    """
    st.session_state.query_result = None
    st.session_state.search_error = None

    try:
        with st.spinner("🔍 Retrieving sources…"):
            sources = retriever.search(question)

        if not sources:
            st.session_state.search_error = (
                "No relevant sources found for that question. "
                "Try rephrasing, or ingest more books."
            )
            return

        with st.spinner("✍️ Generating answer…"):
            answer = summarizer.generate(question, sources)
            # answer may be None if all LLM providers fail — that's OK,
            # the UI still shows retrieved sources.

        st.session_state.query_result = QueryResult(
            question=question,
            sources=sources,
            answer=answer,
        )

    except RuntimeError as exc:
        # Raised by Retriever when vector store is not initialised
        st.session_state.search_error = str(exc)
    except Exception as exc:
        st.session_state.search_error = f"Unexpected error: {exc}"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def _render_header() -> None:
    """Render the app title and the mandatory research disclaimer."""
    st.title("📚 Halachic Q&A")
    disclaimer = (
        '<div class="disclaimer">'
        "⚠️ <strong>Research tool only.</strong> "
        "This application provides AI-generated answers based on retrieved source texts"
        " for study purposes. It is <em>not</em> a substitute for a ruling from a "
        "qualified Rabbi."
        "</div>"
    )
    st.markdown(disclaimer, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Question form
# ---------------------------------------------------------------------------


def _render_question_form(retriever: Retriever, summarizer: Summarizer) -> None:
    """Render the question input and Search button.

    Submitting the form calls ``_handle_search``.  Uses ``st.form`` so
    pressing Enter in the textarea also triggers submission.

    Args:
        retriever: Configured Retriever instance.
        summarizer: Configured Summarizer instance.
    """
    with st.form(key="question_form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            placeholder=(
                "e.g. מה דין נטילת ידיים לסעודה? "
                "/ What is the law of handwashing before a meal?"
            ),
            height=100,
            label_visibility="collapsed",
            key="question_input",
        )
        submitted = st.form_submit_button(
            "🔍 Search",
            use_container_width=True,
            disabled=st.session_state.is_searching,
        )

    if submitted:
        question = question.strip()
        if not question:
            st.warning("Please enter a question before searching.")
        else:
            st.session_state.is_searching = True
            _handle_search(question, retriever, summarizer)
            st.session_state.is_searching = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point.

    Sets up the page, loads cached services, renders the sidebar and
    the main Q&A interface.
    """
    _load_css()
    _init_session_state()

    services = load_services()
    config = services["config"]
    retriever: Retriever = services["retriever"]
    summarizer: Summarizer = services["summarizer"]
    pipeline = services["pipeline"]

    # ---- Sidebar ----
    with st.sidebar:
        render_book_list(config, pipeline)
        st.divider()
        render_ingestion(pipeline, config)

    # ---- Main area ----
    _render_header()
    _render_question_form(retriever, summarizer)

    # Error banner (shown when retrieval or generation raises)
    if st.session_state.search_error:
        st.error(st.session_state.search_error)

    # Results
    result: QueryResult | None = st.session_state.query_result
    if result is not None:
        render_answer(result)
        render_sources(result.sources)


if __name__ == "__main__":
    main()
