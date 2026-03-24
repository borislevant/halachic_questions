"""End-to-end book ingestion pipeline."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config import AppConfig
from src.embeddings.embedder import TextEmbedder
from src.ingestion.chunker import HalachicChunker
from src.ingestion.parser import BookParser
from src.models.book import Book
from src.models.chunk import Chunk
from src.retrieval.bm25_store import BM25Store
from src.retrieval.vector_store import VectorStore
from src.storage import database

logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    """Report of an ingestion operation."""

    success: bool = False
    book_id: str = ""
    book_title: str = ""
    source_path: str = ""
    chunks_created: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0

    def format_report(self) -> str:
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
    5. Build BM25 index for keyword search
    6. Save book metadata to SQLite

    Args:
        config: Application configuration.
        parser: Book file parser.
        chunker: Text chunker.
        embedder: Embedding generator.
        vector_store: Vector database.
        bm25_store: Optional BM25 keyword search index.
    """

    def __init__(
        self,
        config: AppConfig,
        parser: BookParser,
        chunker: HalachicChunker,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        bm25_store: BM25Store | None = None,
    ) -> None:
        self._config = config
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_store = bm25_store

    def ingest_book(
        self,
        file_path: str | Path,
        author: str = "",
        book_id: str | None = None,
        user_id: str | None = None,
        show_progress: bool = True,
    ) -> IngestionReport:
        """Ingest a book file into the system.

        Args:
            file_path: Path to the book file.
            author: Optional author name (if not detected from file).
            book_id: Optional book ID (for re-indexing). If None, generates a new UUID.
            user_id: Optional user ID (book owner). If None, book is shared (public).
            show_progress: Whether to show progress bars.

        Returns:
            IngestionReport with results and any warnings/errors.
        """
        import time

        report = IngestionReport()
        start_time = time.time()

        file_path = Path(file_path)
        report.source_path = str(file_path)

        # Generate or use provided book_id
        if book_id is None:
            book_id = str(uuid4())
        report.book_id = book_id

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

            # Create Book model for persistence
            book = Book(
                id=book_id,
                title=parsed_book.title,
                author=parsed_book.author,
                language=parsed_book.language,
                source_path=str(file_path.resolve()),
                file_format=file_path.suffix.lstrip(".").lower(),
                chunk_count=0,
                ingested_at=datetime.now(),
                status="ingesting",
                user_id=user_id,
            )

            # Write to DB with status='ingesting'
            database.upsert_book(self._config.storage.sqlite_path, book)

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
                # Update status to error
                database.update_book_status(self._config.storage.sqlite_path, book_id, "error")
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

            # Step 5: Build BM25 index (if enabled)
            if self._bm25_store is not None:
                logger.info("Building BM25 index")
                # Get all chunks from vector store (including this book's new chunks)
                all_chunks = self._get_all_chunks_from_vector_store()
                self._bm25_store.build_index(all_chunks, show_progress=show_progress)
                self._bm25_store.save_index()
                logger.info("BM25 index built and saved")

            # Step 6: Save book metadata with final status
            book.chunk_count = len(chunks)
            book.status = "active"
            database.upsert_book(self._config.storage.sqlite_path, book)

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
            # Try to update status to error
            try:
                database.update_book_status(self._config.storage.sqlite_path, book_id, "error")
            except Exception:
                logger.exception("Failed to update book status to error")

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

            # Delete from BM25 store and rebuild index
            if self._bm25_store is not None:
                bm25_deleted = self._bm25_store.delete_by_book_id(book_id)
                logger.info("Removed %d chunks from BM25 index", bm25_deleted)

            # Delete from SQLite metadata
            db_deleted = database.delete_book(self._config.storage.sqlite_path, book_id)

            logger.info("Removed book %s (%d chunks deleted, DB row deleted: %s)", 
                       book_id, deleted_count, db_deleted)
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

        Removes the old version and ingests the new one with the same book_id.

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

        # Ingest new version with the same book_id
        return self.ingest_book(file_path, author=author, book_id=book_id, show_progress=show_progress)

    def ingest_directory(
        self,
        dir_path: str | Path,
        author: str = "",
        recursive: bool = False,
        show_progress: bool = True,
    ) -> list[IngestionReport]:
        """Ingest all supported book files from a directory.

        Args:
            dir_path: Path to the directory containing book files.
            author: Optional author name for all books.
            recursive: Whether to recursively scan subdirectories.
            show_progress: Whether to show progress bars.

        Returns:
            List of IngestionReport for each file processed.
        """
        from src.ingestion.parser import SUPPORTED_FORMATS

        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            logger.error("Directory not found: %s", dir_path)
            return []

        # Collect all supported files
        files = []
        if recursive:
            for ext in SUPPORTED_FORMATS.keys():
                files.extend(dir_path.rglob(f"*{ext}"))
        else:
            for ext in SUPPORTED_FORMATS.keys():
                files.extend(dir_path.glob(f"*{ext}"))

        # Sort for deterministic order
        files = sorted(files)

        logger.info("Found %d supported files in %s (recursive=%s)", 
                   len(files), dir_path, recursive)

        reports = []
        for file_path in files:
            logger.info("Processing file: %s", file_path)
            try:
                report = self.ingest_book(
                    file_path, author=author, show_progress=show_progress
                )
                reports.append(report)
                if report.success:
                    logger.info("Successfully ingested: %s", file_path.name)
                else:
                    logger.warning("Failed to ingest: %s", file_path.name)
            except Exception as e:
                logger.exception("Error processing file %s: %s", file_path, e)
                # Create a failure report
                failed_report = IngestionReport(
                    success=False,
                    book_title=file_path.name,
                    source_path=str(file_path),
                    errors=[f"Unexpected error: {str(e)}"],
                )
                reports.append(failed_report)

        successful = sum(1 for r in reports if r.success)
        logger.info(
            "Directory ingestion complete: %d/%d files succeeded",
            successful,
            len(files),
        )

        return reports

    def _get_all_chunks_from_vector_store(self) -> list[Chunk]:
        """Retrieve all chunks from the vector store to rebuild BM25 index.

        Returns:
            List of all Chunk objects currently in the vector store.
        """
        # Get all books from database
        books = database.get_all_books(self._config.storage.sqlite_path)
        
        all_chunks: list[Chunk] = []
        
        for book in books:
            # Query vector store for all chunks of this book
            # We use a dummy embedding since we're filtering by book_id
            # and want ALL chunks regardless of similarity
            try:
                # Get chunk count
                if book.chunk_count == 0:
                    continue
                
                # Create a dummy query and get many results
                # This is inefficient but ChromaDB doesn't have a "get all by filter" method
                # For production, consider maintaining a separate chunks table in SQLite
                dummy_embedding = [0.0] * self._embedder.get_embedding_dimension()
                results = self._vector_store.search(
                    query_embedding=dummy_embedding,
                    top_k=book.chunk_count,
                    filter_dict={"book_id": book.id},
                )
                
                # Convert results to Chunk objects
                for result in results:
                    metadata = result["metadata"]
                    chunk = Chunk(
                        id=result["id"],
                        text=result["text"],
                        book_id=metadata.get("book_id", ""),
                        book_title=metadata.get("book_title", ""),
                        book_author=metadata.get("book_author", ""),
                        section_path=metadata.get("section_path", ""),
                        section_type=metadata.get("section_type", ""),
                        chunk_index=metadata.get("chunk_index", 0),
                        total_chunks_in_section=metadata.get("total_chunks_in_section", 1),
                        language=metadata.get("language", "he"),
                        char_start=metadata.get("char_start", 0),
                        char_end=metadata.get("char_end", 0),
                        token_count=metadata.get("token_count", 0),
                    )
                    all_chunks.append(chunk)
                    
            except Exception:
                logger.exception("Failed to retrieve chunks for book %s", book.id)
                continue
        
        logger.info("Retrieved %d total chunks from vector store", len(all_chunks))
        return all_chunks

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

    # Create BM25 store if hybrid retrieval is enabled
    bm25_store = None
    if config.retrieval.use_hybrid:
        bm25_store = BM25Store(bm25_dir=config.storage.bm25_dir)
        # Try to load existing index
        bm25_store.load_index()

    # Assemble pipeline
    pipeline = IngestionPipeline(
        config=config,
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    return pipeline
