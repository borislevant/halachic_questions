"""Sidebar component: book upload and ingestion widget."""

import tempfile
from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.config import AppConfig
from src.ingestion.pipeline import IngestionPipeline

_SUPPORTED_TYPES = ["pdf", "txt", "docx", "html", "htm"]


def render_ingestion(pipeline: IngestionPipeline, config: AppConfig) -> None:
    """Render the 'Upload a Book' form in the sidebar.

    Accepts a file upload, optional metadata overrides, and triggers
    ``IngestionPipeline.ingest_book()``. Displays progress feedback and
    calls ``st.rerun()`` on success so the book list refreshes.

    Args:
        pipeline: The ingestion pipeline to call.
        config: Application configuration (provides books_dir path).
    """
    st.markdown("### Upload a Book")

    uploaded_file: UploadedFile | None = st.file_uploader(
        "Choose a file",
        type=_SUPPORTED_TYPES,
        help="Supported formats: PDF, TXT, DOCX, HTML",
        label_visibility="collapsed",
    )

    author_override = st.text_input(
        "Author (optional)",
        placeholder="e.g. Yosef Karo",
        key="ingest_author",
    )

    ingest_btn = st.button(
        "⬆️ Ingest",
        disabled=uploaded_file is None,
        use_container_width=True,
        key="ingest_button",
    )

    if ingest_btn and uploaded_file is not None:
        _run_ingestion(uploaded_file, author_override, pipeline, config)


def _run_ingestion(
    uploaded_file: UploadedFile,
    author: str,
    pipeline: IngestionPipeline,
    config: AppConfig,
) -> None:
    """Save uploaded bytes to disk and run the ingestion pipeline.

    Writes the uploaded file to a temporary location (not inside the
    books_dir, to avoid permanent storage of uploaded content on every
    failed attempt), runs ``pipeline.ingest_book()``, and reports the
    outcome to the user.

    Args:
        uploaded_file: The Streamlit UploadedFile object.
        author: Optional author name override.
        pipeline: The ingestion pipeline.
        config: Application configuration.
    """
    suffix = Path(uploaded_file.name).suffix

    # Write bytes to a named temp file so the parser gets a real path
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        with st.spinner(f"Ingesting '{uploaded_file.name}'…"):
            report = pipeline.ingest_book(
                file_path=tmp_path,
                author=author.strip(),
                show_progress=False,
            )
    finally:
        # Always clean up the temp file
        tmp_path.unlink(missing_ok=True)

    if report.success:
        msg = (
            f"✅ **{report.book_title}** ingested successfully "
            f"({report.chunks_created} chunks, "
            f"{report.processing_time_seconds:.1f}s)"
        )
        if report.warnings:
            warn_lines = "\n".join(f"- {w}" for w in report.warnings)
            st.warning(msg + "\n\n⚠️ Warnings:\n" + warn_lines)
        else:
            st.success(msg)
        st.rerun()
    else:
        error_detail = report.errors[0] if report.errors else "Unknown error"
        st.error(f"❌ Ingestion failed: {error_detail}")
