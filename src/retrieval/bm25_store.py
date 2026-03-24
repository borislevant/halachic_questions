"""BM25 lexical search for keyword-based retrieval."""

import logging
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.models.chunk import Chunk

logger = logging.getLogger(__name__)


class BM25Store:
    """BM25 keyword-based search index.

    Provides lexical retrieval to complement semantic search. Particularly
    effective for:
    - Exact term matches (e.g., "סימן שכח", "משנה ברורה")
    - Book names and author names
    - Specific Halachic terminology
    - Numbers and section references

    The index is built from chunk texts and persisted to disk. It's
    rebuilt during ingestion and loaded on retrieval.

    Args:
        bm25_dir: Directory to store BM25 index files.
    """

    def __init__(self, bm25_dir: str | Path) -> None:
        self._bm25_dir = Path(bm25_dir)
        self._bm25_dir.mkdir(parents=True, exist_ok=True)

        # BM25 index (loaded lazily)
        self._bm25: BM25Okapi | None = None
        # Chunk objects corresponding to BM25 corpus (same order)
        self._chunks: list[Chunk] = []
        # Flag to track if index is loaded
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if BM25 index is loaded."""
        return self._is_loaded

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the index."""
        return len(self._chunks)

    def build_index(
        self,
        chunks: list[Chunk],
        show_progress: bool = False,
    ) -> None:
        """Build BM25 index from a list of chunks.

        Args:
            chunks: List of Chunk objects to index.
            show_progress: Whether to show progress (for large corpora).
        """
        if not chunks:
            logger.warning("Cannot build BM25 index from empty chunk list")
            self._bm25 = None
            self._chunks = []
            self._is_loaded = False
            return

        logger.info("Building BM25 index from %d chunks", len(chunks))

        # Tokenize chunk texts (split by whitespace)
        # For Hebrew, this is reasonable. For more advanced tokenization,
        # consider using a Hebrew tokenizer like YAP or spaCy
        tokenized_corpus = [chunk.text.split() for chunk in chunks]

        # Build BM25 index
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._chunks = chunks
        self._is_loaded = True

        logger.info("BM25 index built successfully")

    def save_index(self) -> None:
        """Persist BM25 index and chunks to disk."""
        if not self._is_loaded:
            logger.warning("Cannot save BM25 index: not built yet")
            return

        bm25_path = self._bm25_dir / "bm25_index.pkl"
        chunks_path = self._bm25_dir / "bm25_chunks.pkl"

        logger.info("Saving BM25 index to %s", bm25_path)

        with open(bm25_path, "wb") as f:
            pickle.dump(self._bm25, f)

        with open(chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)

        logger.info("BM25 index saved successfully")

    def load_index(self) -> bool:
        """Load BM25 index and chunks from disk.

        Returns:
            True if loaded successfully, False if files don't exist.
        """
        bm25_path = self._bm25_dir / "bm25_index.pkl"
        chunks_path = self._bm25_dir / "bm25_chunks.pkl"

        if not bm25_path.exists() or not chunks_path.exists():
            logger.warning(
                "BM25 index files not found in %s. Run ingestion to build index.",
                self._bm25_dir,
            )
            return False

        logger.info("Loading BM25 index from %s", bm25_path)

        try:
            with open(bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)

            with open(chunks_path, "rb") as f:
                self._chunks = pickle.load(f)

            self._is_loaded = True
            logger.info("BM25 index loaded successfully (%d chunks)", len(self._chunks))
            return True

        except Exception:
            logger.exception("Failed to load BM25 index")
            self._bm25 = None
            self._chunks = []
            self._is_loaded = False
            return False

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search using BM25 keyword matching.

        Args:
            query: The search query text.
            top_k: Number of results to return.
            filter_dict: Optional metadata filters (e.g., {"book_id": "..."}).
                         Note: BM25 filtering is post-hoc (not as efficient as vector DB).

        Returns:
            List of dicts with keys: 'id', 'text', 'metadata', 'score'.
            Results are sorted by BM25 score (highest first).
        """
        if not self._is_loaded or self._bm25 is None:
            logger.warning("BM25 index not loaded. Call load_index() first.")
            return []

        if not query.strip():
            logger.warning("Empty query provided to BM25 search")
            return []

        # Tokenize query
        tokenized_query = query.split()

        # Get BM25 scores for all documents
        scores = self._bm25.get_scores(tokenized_query)

        # Create (score, index) pairs and sort
        scored_indices = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Apply filters if provided
        results: list[dict[str, Any]] = []

        for idx, score in scored_indices:
            if len(results) >= top_k:
                break

            # Validate index is within bounds
            if idx >= len(self._chunks):
                logger.warning(
                    "BM25 index out of range: %d >= %d. Skipping.",
                    idx,
                    len(self._chunks),
                )
                continue

            chunk = self._chunks[idx]

            # Apply metadata filters
            if filter_dict:
                match = all(
                    getattr(chunk, key, None) == value
                    for key, value in filter_dict.items()
                )
                if not match:
                    continue

            # Build result dict (same format as vector_store.search)
            result = {
                "id": chunk.id,
                "text": chunk.text,
                "score": float(score),  # BM25 score (not normalized)
                "metadata": {
                    "book_id": chunk.book_id,
                    "book_title": chunk.book_title,
                    "book_author": chunk.book_author,
                    "section_path": chunk.section_path,
                    "section_type": chunk.section_type,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks_in_section": chunk.total_chunks_in_section,
                    "language": chunk.language,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "token_count": chunk.token_count,
                },
            }
            results.append(result)

        logger.debug("BM25 search returned %d results", len(results))
        return results

    def delete_by_book_id(self, book_id: str) -> int:
        """Remove all chunks belonging to a specific book from the index.

        This requires rebuilding the entire index without the deleted chunks.

        Args:
            book_id: The book's UUID.

        Returns:
            Number of chunks deleted.
        """
        if not self._is_loaded:
            logger.warning("BM25 index not loaded. Cannot delete.")
            return 0

        original_count = len(self._chunks)

        # Filter out chunks from the target book
        remaining_chunks = [
            chunk for chunk in self._chunks if chunk.book_id != book_id
        ]
        deleted_count = original_count - len(remaining_chunks)

        if deleted_count == 0:
            logger.info("No chunks found for book_id=%s in BM25 index", book_id)
            return 0

        logger.info(
            "Removing %d chunks from BM25 index (book_id=%s)",
            deleted_count,
            book_id,
        )

        # Rebuild index without deleted chunks
        if remaining_chunks:
            self.build_index(remaining_chunks)
            self.save_index()
        else:
            # No chunks left
            self._bm25 = None
            self._chunks = []
            self._is_loaded = False
            # Delete index files
            bm25_path = self._bm25_dir / "bm25_index.pkl"
            chunks_path = self._bm25_dir / "bm25_chunks.pkl"
            bm25_path.unlink(missing_ok=True)
            chunks_path.unlink(missing_ok=True)
            logger.info("BM25 index is now empty")

        return deleted_count

    def clear(self) -> None:
        """Clear the in-memory index (does not delete files)."""
        self._bm25 = None
        self._chunks = []
        self._is_loaded = False
        logger.info("BM25 index cleared from memory")

    def rebuild_from_all_chunks(self, all_chunks: list[Chunk]) -> None:
        """Rebuild the entire BM25 index from a fresh list of chunks.

        Used when re-syncing BM25 with the vector store.

        Args:
            all_chunks: Complete list of all chunks in the system.
        """
        logger.info("Rebuilding BM25 index from %d chunks", len(all_chunks))
        self.build_index(all_chunks)
        self.save_index()
        logger.info("BM25 index rebuilt successfully")
