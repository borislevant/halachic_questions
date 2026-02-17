"""Vector database management using ChromaDB."""

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.config import Settings

from src.models.chunk import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB.

    Handles:
    - Creating and persisting collections
    - Storing chunk embeddings with metadata
    - Semantic similarity search
    - Filtering by book/section

    Args:
        persist_directory: Path to ChromaDB storage directory.
        collection_name: Name of the ChromaDB collection (default: "halachic_texts").
        use_persistent: If False, uses EphemeralClient (in-memory only, for testing).
    """

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "halachic_texts",
        use_persistent: bool = True,
    ) -> None:
        self._persist_dir = Path(persist_directory)
        self._collection_name = collection_name
        self._use_persistent = use_persistent
        self._client: chromadb.Client | None = None
        self._collection: chromadb.Collection | None = None

    def initialize(self) -> None:
        """Initialize the ChromaDB client and collection.

        Creates the persist directory if it doesn't exist and loads
        or creates the collection.
        """
        if self._client is not None:
            return

        try:
            if self._use_persistent:
                self._persist_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Initializing ChromaDB at '%s'", self._persist_dir)
                self._client = chromadb.PersistentClient(
                    path=str(self._persist_dir),
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    ),
                )
            else:
                logger.info("Initializing ChromaDB in-memory (ephemeral)")
                self._client = chromadb.EphemeralClient(
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    ),
                )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},  # Cosine similarity for normalized vectors
            )

            logger.info(
                "Collection '%s' ready with %d items",
                self._collection_name,
                self._collection.count(),
            )

        except Exception:
            logger.exception("Failed to initialize ChromaDB")
            raise

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        show_progress: bool = False,
    ) -> None:
        """Add chunks with their embeddings to the vector store.

        Args:
            chunks: List of Chunk objects to store.
            embeddings: Corresponding embeddings (same length as chunks).
            show_progress: Whether to log progress (for large batches).

        Raises:
            ValueError: If chunks and embeddings have different lengths.
            RuntimeError: If the store is not initialized.
        """
        if self._collection is None:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        if not chunks:
            logger.warning("add_chunks called with empty list")
            return

        # Prepare data for ChromaDB
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk, embedding in zip(chunks, embeddings):
            # Use chunk's ID (auto-generated if not provided)
            chunk_id = chunk.id

            ids.append(chunk_id)
            documents.append(chunk.text)

            # Store all chunk metadata
            metadatas.append({
                "book_id": chunk.book_id,
                "book_title": chunk.book_title,
                "book_author": chunk.book_author or "",
                "section_path": chunk.section_path,
                "section_type": chunk.section_type,
                "language": chunk.language,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "token_count": chunk.token_count,
                "chunk_index": chunk.chunk_index,
                "total_chunks_in_section": chunk.total_chunks_in_section,
            })

        try:
            # Add to ChromaDB in batches
            batch_size = 5000
            for i in range(0, len(ids), batch_size):
                batch_end = min(i + batch_size, len(ids))

                self._collection.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings[i:batch_end],  # type: ignore[arg-type]
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                )

                if show_progress:
                    logger.info("Added chunks %d-%d/%d", i, batch_end, len(ids))

            logger.info("Successfully added %d chunks to vector store", len(chunks))

        except Exception:
            logger.exception("Failed to add chunks to vector store")
            raise

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks using semantic similarity.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.
            filter_dict: Optional metadata filters (e.g., {"book_id": "..."}).

        Returns:
            List of search results, each containing:
            - id: chunk_id
            - score: similarity score (0-1, higher is better)
            - text: chunk text
            - metadata: all chunk metadata

        Raises:
            RuntimeError: If the store is not initialized.
        """
        if self._collection is None:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict,
                include=["documents", "metadatas", "distances"],
            )

            # Convert ChromaDB results to a cleaner format
            search_results: list[dict[str, Any]] = []

            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    # ChromaDB returns distances (lower is better for cosine)
                    # Convert to similarity score (higher is better)
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    similarity = 1.0 - distance

                    search_results.append({
                        "id": results["ids"][0][i],
                        "score": similarity,
                        "text": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    })

            return search_results

        except Exception:
            logger.exception("Failed to search vector store")
            raise

    def delete_by_book_id(self, book_id: str) -> int:
        """Delete all chunks belonging to a specific book.

        Args:
            book_id: The book's UUID.

        Returns:
            Number of chunks deleted.

        Raises:
            RuntimeError: If the store is not initialized.
        """
        if self._collection is None:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        try:
            # Query to get all chunk IDs for this book
            results = self._collection.get(
                where={"book_id": book_id},
                include=[],
            )

            if not results["ids"]:
                logger.info("No chunks found for book_id '%s'", book_id)
                return 0

            # Delete all matching IDs
            self._collection.delete(ids=results["ids"])

            count = len(results["ids"])
            logger.info("Deleted %d chunks for book_id '%s'", count, book_id)
            return count

        except Exception:
            logger.exception("Failed to delete chunks for book_id '%s'", book_id)
            raise

    def get_chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        """Retrieve a specific chunk by its ID.

        Args:
            chunk_id: The chunk's UUID.

        Returns:
            Dict with chunk data, or None if not found.

        Raises:
            RuntimeError: If the store is not initialized.
        """
        if self._collection is None:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        try:
            results = self._collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )

            if not results["ids"]:
                return None

            return {
                "id": results["ids"][0],
                "text": results["documents"][0] if results["documents"] else "",
                "metadata": results["metadatas"][0] if results["metadatas"] else {},
            }

        except Exception:
            logger.exception("Failed to get chunk by ID '%s'", chunk_id)
            raise

    def get_neighboring_chunks(
        self,
        chunk_id: str,
        context_size: int = 1,
    ) -> dict[str, list[dict[str, Any]]]:
        """Get chunks before and after a specific chunk.

        Useful for expanding context around a retrieved source.

        Args:
            chunk_id: The chunk's UUID.
            context_size: Number of chunks to retrieve before and after.

        Returns:
            Dict with "previous" and "next" lists of chunks.

        Raises:
            RuntimeError: If the store is not initialized.
        """
        if self._collection is None:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        # Get the target chunk
        chunk = self.get_chunk_by_id(chunk_id)
        if not chunk:
            return {"previous": [], "next": []}

        metadata = chunk["metadata"]
        book_id = metadata.get("book_id")
        section_path = metadata.get("section_path")
        chunk_index = metadata.get("chunk_index")

        if book_id is None or chunk_index is None:
            return {"previous": [], "next": []}

        try:
            # Get all chunks in the same section
            # ChromaDB requires $and operator for multiple conditions
            where_filter = {
                "$and": [
                    {"book_id": book_id},
                    {"section_path": section_path},
                ]
            }
            results = self._collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )

            if not results["ids"]:
                return {"previous": [], "next": []}

            # Sort by chunk_index
            chunks_with_index = []
            for i, chunk_id_result in enumerate(results["ids"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                idx = meta.get("chunk_index", 0)
                chunks_with_index.append({
                    "id": chunk_id_result,
                    "text": results["documents"][i] if results["documents"] else "",
                    "metadata": meta,
                    "chunk_index": idx,
                })

            chunks_with_index.sort(key=lambda x: x["chunk_index"])

            # Find target chunk position
            target_pos = next(
                (i for i, c in enumerate(chunks_with_index) if c["id"] == chunk_id),
                None,
            )

            if target_pos is None:
                return {"previous": [], "next": []}

            # Get previous and next chunks
            prev_start = max(0, target_pos - context_size)
            next_end = min(len(chunks_with_index), target_pos + context_size + 1)

            previous = chunks_with_index[prev_start:target_pos]
            next_chunks = chunks_with_index[target_pos + 1:next_end]

            # Remove chunk_index from results (internal detail)
            for chunk_list in [previous, next_chunks]:
                for c in chunk_list:
                    c.pop("chunk_index", None)

            return {
                "previous": previous,
                "next": next_chunks,
            }

        except Exception:
            logger.exception("Failed to get neighboring chunks for '%s'", chunk_id)
            raise

    def count(self) -> int:
        """Get the total number of chunks in the vector store.

        Returns:
            Total chunk count.

        Raises:
            RuntimeError: If the store is not initialized.
        """
        if self._collection is None:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        return self._collection.count()

    def reset(self) -> None:
        """Delete all data in the vector store.

        WARNING: This is destructive and cannot be undone.

        Raises:
            RuntimeError: If the store is not initialized.
        """
        if self._collection is None:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        logger.warning("Resetting vector store - all data will be deleted!")

        try:
            self._client.delete_collection(name=self._collection_name)  # type: ignore[union-attr]
            self._collection = self._client.get_or_create_collection(  # type: ignore[union-attr]
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Vector store reset complete")

        except Exception:
            logger.exception("Failed to reset vector store")
            raise

    @property
    def collection_name(self) -> str:
        """Get the name of the current collection."""
        return self._collection_name

    @property
    def is_initialized(self) -> bool:
        """Check if the vector store has been initialized."""
        return self._collection is not None

    def close(self) -> None:
        """Close the ChromaDB client and release resources.

        This should be called when done with the vector store,
        especially important on Windows to release file locks.
        """
        if self._client is not None:
            logger.debug("Closing ChromaDB client")
            # Clear references to allow garbage collection
            self._collection = None
            self._client = None

    def __enter__(self) -> "VectorStore":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
