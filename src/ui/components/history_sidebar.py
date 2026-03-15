"""History sidebar component for displaying past queries."""

from datetime import datetime, timedelta

import streamlit as st

from src.storage.database import (
    delete_query,
    get_query_by_id,
    get_query_history,
    search_query_history,
)


def _group_by_date(queries: list[dict]) -> dict[str, list[dict]]:
    """Group queries by date categories.
    
    Args:
        queries: List of query dictionaries with 'created_at' field.
        
    Returns:
        Dictionary with keys: 'Today', 'Yesterday', 'This Week', 'Older'.
    """
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    week_start = today_start - timedelta(days=7)
    
    grouped = {
        "Today": [],
        "Yesterday": [],
        "This Week": [],
        "Older": [],
    }
    
    for query in queries:
        created_at = datetime.fromisoformat(query["created_at"])
        if created_at >= today_start:
            grouped["Today"].append(query)
        elif created_at >= yesterday_start:
            grouped["Yesterday"].append(query)
        elif created_at >= week_start:
            grouped["This Week"].append(query)
        else:
            grouped["Older"].append(query)
    
    return grouped


def render_history_sidebar(user_id: str, db_path: str) -> None:
    """Render the query history sidebar.
    
    Shows past queries grouped by date, with search and delete functionality.
    Clicking a query loads it into the main view.
    
    Args:
        user_id: The current user's ID.
        db_path: Path to SQLite database.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("📜 Query History")
    
    # Search box
    search_term = st.sidebar.text_input(
        "Search history",
        placeholder="Type keywords...",
        key="history_search",
    )
    
    # Fetch queries
    if search_term:
        queries = search_query_history(db_path, user_id, search_term, limit=50)
    else:
        queries = get_query_history(db_path, user_id, limit=50)
    
    if not queries:
        st.sidebar.info("No queries in history" if not search_term else "No results found")
        return
    
    # Group by date
    grouped = _group_by_date(queries)
    
    # Display grouped queries
    for category, items in grouped.items():
        if not items:
            continue
        
        st.sidebar.markdown(f"**{category}**")
        
        for query in items:
            # Truncate question for display
            question_preview = query["question"][:50]
            if len(query["question"]) > 50:
                question_preview += "..."
            
            col1, col2 = st.sidebar.columns([4, 1])
            
            with col1:
                # Button to load query
                if st.button(
                    question_preview,
                    key=f"load_{query['id']}",
                    use_container_width=True,
                    help=query["question"],
                ):
                    _load_query_into_view(db_path, query["id"], user_id)
            
            with col2:
                # Delete button
                if st.button(
                    "🗑️",
                    key=f"delete_{query['id']}",
                    help="Delete query",
                ):
                    if delete_query(db_path, query["id"], user_id):
                        st.success("Deleted!")
                        st.rerun()
                    else:
                        st.error("Error deleting")
        
        st.sidebar.markdown("")  # Spacing


def _load_query_into_view(db_path: str, query_id: str, user_id: str) -> None:
    """Load a historical query into the main view.
    
    Args:
        db_path: Path to SQLite database.
        query_id: The query's UUID.
        user_id: The user's UUID (for ownership check).
    """
    from src.models.chunk import Chunk
    from src.models.query_result import (
        Citation,
        GeneratedAnswer,
        QueryResult,
        RetrievalResult,
    )
    
    query_data = get_query_by_id(db_path, query_id, user_id)
    if not query_data:
        st.sidebar.error("Query not found")
        return
    
    # Reconstruct QueryResult from database data
    sources = []
    for src_data in query_data.get("sources_json", []):
        chunk = Chunk(
            id=src_data["chunk_id"],
            book_title=src_data["book_title"],
            section_path=src_data["section_path"],
            text=src_data["text"],
            book_id="",  # Not stored in history
        )
        sources.append(
            RetrievalResult(
                chunk=chunk,
                similarity_score=src_data["similarity_score"],
                rerank_score=src_data.get("rerank_score"),
            )
        )
    
    answer = None
    if query_data["answer_text"]:
        answer = GeneratedAnswer(
            text=query_data["answer_text"],
            citations=[],  # Citations not stored separately in DB
            model_used=query_data["model_used"] or "",
            tokens_used=query_data["tokens_used"] or 0,
            latency_ms=query_data["latency_ms"] or 0,
        )
    
    query_result = QueryResult(
        id=query_data["id"],
        question=query_data["question"],
        sources=sources,
        answer=answer,
        timestamp=datetime.fromisoformat(query_data["created_at"]),
        feedback=query_data["feedback"],
        user_id=user_id,
    )
    
    # Store in session state to display in main view
    st.session_state.query_result = query_result
    st.rerun()
