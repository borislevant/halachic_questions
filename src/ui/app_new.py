"""Main Streamlit application for Halachic Q&A."""

from pathlib import Path

import streamlit as st

from src.auth.auth_service import AuthService
from src.generation.summarizer import Summarizer
from src.models.query_result import QueryResult
from src.models.user import User
from src.retrieval.retriever import Retriever
from src.storage.database import save_query
from src.ui.components.answer import render_answer
from src.ui.components.book_list import render_book_list
from src.ui.components.history_sidebar import render_history_sidebar
from src.ui.components.ingestion import render_ingestion
from src.ui.components.sources import render_sources
from src.ui.pages.auth_page import render_auth_page
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
        "auth_token": None,     # str | None — JWT token
        "user": None,           # User | None — authenticated user
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _handle_search(
    question: str,
    retriever: Retriever,
    summarizer: Summarizer,
    user: User,
    db_path: str,
) -> None:
    """Run retrieval + generation and store the result in session state.

    Clears any previous result and error before running. On success,
    ``st.session_state.query_result`` is set to a populated QueryResult.
    On failure, ``st.session_state.search_error`` is set instead.

    Args:
        question: The user's question text.
        retriever: Configured Retriever instance.
        summarizer: Configured Summarizer instance.
        user: The authenticated user.
        db_path: Path to SQLite database for saving query history.
    """
    st.session_state.query_result = None
    st.session_state.search_error = None

    try:
        with st.spinner("🔍 מחפש מקורות..."):
            sources = retriever.search(question)

        if not sources:
            st.session_state.search_error = (
                "לא נמצאו מקורות רלוונטיים לשאלה זו. "
                "נסה לנסח מחדש או להוסיף ספרים נוספים."
            )
            return

        with st.spinner("✍️ מייצר תשובה..."):
            answer = summarizer.generate(question, sources)
            # answer may be None if all LLM providers fail — that's OK,
            # the UI still shows retrieved sources.

        query_result = QueryResult(
            question=question,
            sources=sources,
            answer=answer,
            user_id=user.id,
        )
        
        st.session_state.query_result = query_result
        
        # Save to query history
        save_query(db_path, query_result, user.id)

    except RuntimeError as exc:
        # Raised by Retriever when vector store is not initialised
        st.session_state.search_error = str(exc)
    except Exception as exc:
        st.session_state.search_error = f"שגיאה בלתי צפויה: {exc}"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def _render_header() -> None:
    """Render the app title and the mandatory research disclaimer."""
    st.title("📚 Halachic Q&A")
    disclaimer = (
        '<div class="disclaimer">'
        "⚠️ <strong>כלי מחקר בלבד.</strong> "
        "יישום זה מספק תשובות המבוססות על AI למטרות לימוד. "
        "הוא <em>אינו</em> תחליף לפסק הלכה מרב מוסמך."
        "</div>"
    )
    st.markdown(disclaimer, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Question form
# ---------------------------------------------------------------------------


def _render_question_form(
    retriever: Retriever,
    summarizer: Summarizer,
    user: User,
    db_path: str,
) -> None:
    """Render the question input and Search button.

    Submitting the form calls ``_handle_search``.  Uses ``st.form`` so
    pressing Enter in the textarea also triggers submission.

    Args:
        retriever: Configured Retriever instance.
        summarizer: Configured Summarizer instance.
        user: The authenticated user.
        db_path: Path to SQLite database.
    """
    with st.form(key="question_form", clear_on_submit=False):
        question = st.text_area(
            "השאלה שלך",
            placeholder=(
                "לדוגמה: מה דין נטילת ידיים לסעודה?"
            ),
            height=100,
            label_visibility="collapsed",
            key="question_input",
        )
        submitted = st.form_submit_button(
            "🔍 חפש",
            use_container_width=True,
            disabled=st.session_state.is_searching,
        )

    if submitted:
        question = question.strip()
        if not question:
            st.warning("נא להזין שאלה לפני החיפוש.")
        else:
            st.session_state.is_searching = True
            _handle_search(question, retriever, summarizer, user, db_path)
            st.session_state.is_searching = False


# ---------------------------------------------------------------------------
# Authentication check and page routing
# ---------------------------------------------------------------------------


def _check_authentication(auth_service: AuthService) -> User | None:
    """Check if user is authenticated via JWT token in session state.
    
    Args:
        auth_service: Configured AuthService instance.
        
    Returns:
        User object if authenticated, None otherwise.
    """
    token = st.session_state.get("auth_token")
    if not token:
        return None
    
    user, error = auth_service.verify_token(token)
    if error:
        # Token expired or invalid - clear session
        st.session_state.auth_token = None
        st.session_state.user = None
        return None
    
    # Update user in session state if it changed
    st.session_state.user = user
    return user


def _render_qa_page(user: User) -> None:
    """Render the main Q&A page for authenticated users.
    
    Args:
        user: The authenticated user.
    """
    services = load_services()
    config = services["config"]
    retriever: Retriever = services["retriever"]
    summarizer: Summarizer = services["summarizer"]
    pipeline = services["pipeline"]
    db_path = config["storage"]["sqlite_path"]

    # ---- Sidebar ----
    with st.sidebar:
        # User info and logout
        st.markdown(f"**👤 {user.username}**")
        if user.role == "admin":
            st.caption("🔑 מנהל מערכת")
        
        if st.button("🚪 התנתק", use_container_width=True):
            st.session_state.auth_token = None
            st.session_state.user = None
            st.session_state.query_result = None
            st.rerun()
        
        st.divider()
        
        # Book management
        render_book_list(config, pipeline, user)
        st.divider()
        render_ingestion(pipeline, config, user)
        
        # Query history
        render_history_sidebar(user.id, db_path)

    # ---- Main area ----
    _render_header()
    _render_question_form(retriever, summarizer, user, db_path)

    # Error banner (shown when retrieval or generation raises)
    if st.session_state.search_error:
        st.error(st.session_state.search_error)

    # Results
    result: QueryResult | None = st.session_state.query_result
    if result is not None:
        render_answer(result)
        render_sources(result.sources)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point.

    Checks authentication status and routes to either auth page or Q&A page.
    """
    _load_css()
    _init_session_state()

    # Load minimal config for auth service
    services = load_services()
    config = services["config"]
    
    # Initialize auth service
    auth_service = AuthService(
        db_path=config["storage"]["sqlite_path"],
        jwt_secret=config["auth"]["jwt_secret_key"],
        jwt_algorithm=config["auth"]["jwt_algorithm"],
        jwt_expiry_hours=config["auth"]["jwt_expiry_hours"],
    )
    
    # Check authentication
    user = _check_authentication(auth_service)
    
    if user is None:
        # Not authenticated - show login/register page
        render_auth_page(auth_service)
    else:
        # Authenticated - show Q&A page
        _render_qa_page(user)


if __name__ == "__main__":
    main()
