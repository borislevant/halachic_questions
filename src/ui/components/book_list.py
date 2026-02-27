"""Sidebar component: library of ingested books with delete actions."""

import streamlit as st

from src.config import AppConfig
from src.ingestion.pipeline import IngestionPipeline
from src.models.book import Book
from src.storage import database


def render_book_list(config: AppConfig, pipeline: IngestionPipeline) -> None:
    """Render the library panel in the sidebar.

    Lists all active books from SQLite with their metadata and a delete
    button for each. Deleting a book removes it from both the vector store
    and the SQLite record via the pipeline's ``remove_book`` method.

    Args:
        config: Application configuration (provides sqlite_path).
        pipeline: IngestionPipeline used to remove books from all stores.
    """
    books: list[Book] = database.list_books(config.storage.sqlite_path)
    active_books = [b for b in books if b.status == "active"]

    count_label = f"({len(active_books)})" if active_books else ""
    st.markdown(f"### 📚 Library {count_label}")

    if not active_books:
        st.caption("No books ingested yet. Upload a file below to get started.")
        return

    for book in active_books:
        _render_book_row(book, pipeline)


def _render_book_row(book: Book, pipeline: IngestionPipeline) -> None:
    """Render a single book entry with title, metadata, and a delete button.

    Args:
        book: The Book model to display.
        pipeline: Pipeline used to remove the book on delete.
    """
    col_text, col_btn = st.columns([5, 1])

    with col_text:
        st.markdown(
            f'<div class="book-row-title">{book.title}</div>',
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
        # Unique key prevents widget ID collisions when multiple books shown
        if st.button("🗑", key=f"delete_{book.id}", help=f"Remove '{book.title}'"):
            with st.spinner(f"Removing '{book.title}'…"):
                success = pipeline.remove_book(book.id)
            if success:
                st.success(f"'{book.title}' removed.")
            else:
                st.error(f"Failed to remove '{book.title}'.")
            st.rerun()

    st.divider()
