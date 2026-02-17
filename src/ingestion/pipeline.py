"""End-to-end book ingestion pipeline."""

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config import AppConfig
from src.embeddings.embedder import TextEmbedder
from src.ingestion.chunker import HalachicChunker
from src.ingestion.parser import BookParser
from src.models.book import Book
from src.models.chunk import Chunk
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionReport:
    """Report of an ingestion operation."""

    def __init__(self) -> None:
        self.success: bool = False
        self.book_id: str = ""
        self.book_title: str = ""
        self.source_path: str = ""
        self.chunks_created: int = 0
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.processing_time_seconds: float = 0.0

    def __str__(self) -> str:
        """Format report as a readable string."""
        lines = [
            f"Ingestion Report: {self.book_title}",
            f"  Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"  Source: {self.source_path}",
            f"  Chunks Created: {self.chunks_created}",
            f"  Time: {self.processing_time_seconds:.2f}s",
        ]

        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"    - {warning}")

        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"    - {error}")

        return "\n".join(lines)


class IngestionPipeline:
    """Orchestrates the full book ingestion process.

    Pipeline steps:
    1. Parse book file (PDF, TXT, DOCX, HTML)
    2. Detect structure and chunk text
    3. Generate embeddings for each chunk
    4. Store chunks and embeddings in vector database
    5. Save book metadata to SQLite

    Args:
        config: Application configuration.
        parser: Book file parser.
        chunker: Text chunker.
        embedder: Embedding generator.
        vector_store: Vector database.
    """

    def __init__(
        self,
        config: AppConfig,
        parser: BookParser,
        chunker: HalachicChunker,
        embedder: TextEmbedder,
        vector_store: VectorStore,
    ) -> None:
        self._config = config
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    def ingest_book(
        self,
        file_path: str | Path,
        author: str = "",
        show_progress: bool = True,
    ) -> IngestionReport:
        """Ingest a book file into the system.

        Args:
            file_path: Path to the book file.
            author: Optional author name (if not detected from file).
            show_progress: Whether to show progress bars.

        Returns:
            IngestionReport with results and any warnings/errors.
        """
        import time

        report = IngestionReport()
        start_time = time.time()

        file_path = Path(file_path)
        report.source_path = str(file_path)

        try:
            # Step 1: Parse the book
            logger.info("Parsing book: %s", file_path)
            parsed_book = self._parser.parse(file_path)

            if not parsed_book.raw_text.strip():
                report.errors.append("Book file is empty or contains no readable text")
                report.success = False
                report.processing_time_seconds = time.time() - start_time
                return report

            # Override author if provided
            if author:
                parsed_book.author = author

            report.book_title = parsed_book.title
            book_id = str(uuid4())
            report.book_id = book_id

            # Validate Hebrew content for Hebrew books
            if parsed_book.language == "he":
                hebrew_chars = sum(1 for c in parsed_book.raw_text if "\u0590" <= c <= "\u05FF")
                if hebrew_chars < 10:
                    report.warnings.append(
                        "Very little Hebrew text detected. File may be corrupted or misencoded."
                    )

            # Step 2: Chunk the text
            logger.info("Chunking text for book: %s", parsed_book.title)
            chunks = self._chunker.chunk(parsed_book, book_id=book_id)

            if not chunks:
                report.errors.append("Chunking produced no chunks (text too short or empty)")
                report.success = False
                report.processing_time_seconds = time.time() - start_time
                return report

            report.chunks_created = len(chunks)

            # Check for very small chunks
            small_chunks = sum(1 for c in chunks if c.token_count < 20)
            if small_chunks > len(chunks) * 0.3:
                report.warnings.append(
                    f"{small_chunks}/{len(chunks)} chunks are very small (<20 tokens). "
                    "Consider adjusting chunking parameters."
                )

            # Step 3: Generate embeddings
            logger.info("Generating embeddings for %d chunks", len(chunks))
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self._embedder.embed(chunk_texts, show_progress=show_progress)

            if len(embeddings) != len(chunks):
                report.errors.append(
                    f"Embedding count mismatch: {len(embeddings)} embeddings "
                    f"for {len(chunks)} chunks"
                )
                report.success = False
                report.processing_time_seconds = time.time() - start_time
                return report

            # Step 4: Store in vector database
            logger.info("Storing chunks in vector database")
            self._vector_store.add_chunks(
                chunks=chunks,
                embeddings=embeddings,
                show_progress=show_progress,
            )

            # Step 5: Save book metadata
            # TODO: This will be implemented when we add storage/history.py
            # For now, we just log success
            logger.info(
                "Successfully ingested book '%s' (%s) with %d chunks",
                parsed_book.title,
                book_id,
                len(chunks),
            )

            report.success = True
            report.processing_time_seconds = time.time() - start_time

        except Exception as e:
            logger.exception("Failed to ingest book: %s", file_path)
            report.errors.append(f"Ingestion failed: {str(e)}")
            report.success = False
            report.processing_time_seconds = time.time() - start_time

        return report

    def remove_book(self, book_id: str) -> bool:
        """Remove a book and all its chunks from the system.

        Args:
            book_id: The book's UUID.

        Returns:
            True if successful, False otherwise.
        """
        try:
            logger.info("Removing book with ID: %s", book_id)

            # Delete from vector store
            deleted_count = self._vector_store.delete_by_book_id(book_id)

            # TODO: Delete from SQLite metadata when storage/history.py is implemented

            logger.info("Removed book %s (%d chunks deleted)", book_id, deleted_count)
            return True

        except Exception:
            logger.exception("Failed to remove book: %s", book_id)
            return False

    def reindex_book(
        self,
        file_path: str | Path,
        book_id: str,
        author: str = "",
        show_progress: bool = True,
    ) -> IngestionReport:
        """Re-index an existing book.

        Removes the old version and ingests the new one.

        Args:
            file_path: Path to the book file.
            book_id: The existing book's UUID.
            author: Optional author name.
            show_progress: Whether to show progress bars.

        Returns:
            IngestionReport with results.
        """
        logger.info("Re-indexing book: %s", book_id)

        # Remove old version
        self.remove_book(book_id)

        # Ingest new version (but use the same book_id)
        # Note: We'll need to modify ingest_book to accept an optional book_id
        # For now, this creates a new book_id
        return self.ingest_book(file_path, author=author, show_progress=show_progress)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the ingestion system.

        Returns:
            Dict with system statistics.
        """
        return {
            "total_chunks": self._vector_store.count(),
            "embedding_model": self._embedder.model_name,
            "embedding_device": self._embedder.device,
            "embedding_dimension": (
                self._embedder.get_embedding_dimension()
                if self._embedder.device != "not loaded"
                else None
            ),
            "vector_store_initialized": self._vector_store.is_initialized,
        }

    @property
    def vector_store(self) -> VectorStore:
        """Access to the vector store for cleanup or advanced operations."""
        return self._vector_store


def create_ingestion_pipeline(config: AppConfig) -> IngestionPipeline:
    """Factory function to create a fully configured IngestionPipeline.

    This is the recommended way to instantiate the pipeline with all
    dependencies properly injected.

    Args:
        config: Application configuration.

    Returns:
        Ready-to-use IngestionPipeline instance.
    """
    # Create components
    parser = BookParser()
    chunker = HalachicChunker(config.chunking)
    embedder = TextEmbedder(config.embedding)

    vector_store = VectorStore(
        persist_directory=config.storage.chroma_dir,
        collection_name="halachic_texts",
    )
    vector_store.initialize()

    # Assemble pipeline
    pipeline = IngestionPipeline(
        config=config,
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    return pipeline
