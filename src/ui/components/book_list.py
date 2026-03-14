"""Sidebar component: library of ingested books with delete actions."""

import streamlit as st

from src.auth.permissions import can_delete_book
from src.config import AppConfig
from src.ingestion.pipeline import IngestionPipeline
from src.models.book import Book
from src.models.user import User
from src.storage import database


def render_book_list(config: AppConfig, pipeline: IngestionPipeline, user: User) -> None:
    """Render the library panel in the sidebar.

    Lists books visible to the user: shared books + user's private books.
    Shows ownership badges and delete buttons with permission checks.

    Args:
        config: Application configuration (provides sqlite_path).
        pipeline: IngestionPipeline used to remove books from all stores.
        user: The authenticated user.
    """
    books: list[Book] = database.list_books(config.storage.sqlite_path, user.id)
    active_books = [b for b in books if b.status == "active"]

    count_label = f"({len(active_books)})" if active_books else ""
    st.markdown(f"### 📚 Library {count_label}")

    if not active_books:
        st.caption("No books yet. Upload a file below to get started.")
        return
    
    # Filter controls
    filter_option = st.radio(
        "Filter",
        ["All", "Shared", "My Books"],
        horizontal=True,
        key="book_filter",
    )
    
    # Apply filter
    if filter_option == "Shared":
        active_books = [b for b in active_books if b.user_id is None]
    elif filter_option == "My Books":
        active_books = [b for b in active_books if b.user_id == user.id]

    for book in active_books:
        _render_book_row(book, pipeline, user)


def _render_book_row(book: Book, pipeline: IngestionPipeline, user: User) -> None:
    """Render a single book entry with title, metadata, and a delete button.

    Args:
        book: The Book model to display.
        pipeline: Pipeline used to remove the book on delete.
        user: The authenticated user.
    """
    col_text, col_btn = st.columns([5, 1])

    with col_text:
        # Ownership badge
        if book.user_id is None:
            badge = '<span style="background-color: #4CAF50; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7em; margin-left: 5px;">Shared</span>'
        else:
            badge = '<span style="background-color: #2196F3; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7em; margin-left: 5px;">Private</span>'
        
        st.markdown(
            f'<div class="book-row-title">{book.title} {badge}</div>',
            unsafe_allow_html=True,
        )
        author_str = book.author if book.author else "Unknown author"
        meta_html = (
            f'<div class="book-row-meta">'
            f"{author_str} &nbsp;·&nbsp; {book.chunk_count} chunks"
            f"</div>"
        )
        st.markdown(meta_html, unsafe_allow_html=True)

    with col_btn:
        # Show delete button only if user has permission
        if can_delete_book(user, book):
            if st.button("🗑", key=f"delete_{book.id}", help=f"Remove '{book.title}'"):
                with st.spinner(f"Removing '{book.title}'..."):
                    success = pipeline.remove_book(book.id)
                if success:
                    st.success(f"'{book.title}' removed.")
                else:
                    st.error(f"Failed to remove '{book.title}'.")
                st.rerun()

    st.divider()
