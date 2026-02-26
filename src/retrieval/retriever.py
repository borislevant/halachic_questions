"""High-level retrieval service for question answering."""

import logging
from typing import Any

from src.config import RetrievalConfig
from src.embeddings.embedder import TextEmbedder
from src.models.chunk import Chunk
from src.models.query_result import RetrievalResult
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """High-level retrieval service for question answering.

    Orchestrates:
    - Query embedding
    - Vector similarity search
    - Optional reranking
    - Score filtering
    - Context enrichment

    Args:
        embedder: TextEmbedder instance.
        vector_store: VectorStore instance.
        config: RetrievalConfig with top_k, min_similarity, etc.
        reranker: Optional Reranker for precision improvement.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        config: RetrievalConfig,
        reranker: Any | None = None,  # Use Any to avoid circular import
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._config = config
        self._reranker = reranker

    def search(
        self,
        question: str,
        top_k: int | None = None,
        min_similarity: float | None = None,
        filter_dict: dict[str, Any] | None = None,
        include_context: bool = True,
    ) -> list[RetrievalResult]:
        """Search for relevant chunks given a question.

        Args:
            question: User's question text.
            top_k: Number of results to return (overrides config).
            min_similarity: Minimum similarity score (overrides config).
            filter_dict: Metadata filters (e.g., {"book_id": "..."}).
            include_context: Whether to fetch neighboring chunks.

        Returns:
            List of RetrievalResult objects, sorted by score (highest first).
        """
        # Handle empty query
        if not question or not question.strip():
            logger.warning("Empty query provided to retriever")
            return []

        # Override config with parameters if provided
        final_top_k = top_k if top_k is not None else self._config.top_k
        final_min_similarity = (
            min_similarity if min_similarity is not None else self._config.min_similarity
        )
        initial_candidates = self._config.initial_candidates

        # Check vector store is initialized
        if not self._vector_store.is_initialized:
            raise RuntimeError(
                "VectorStore not initialized. Call vector_store.initialize() first."
            )

        try:
            # Step 1: Embed the query with correct prefix
            logger.debug("Embedding query: '%s'", question[:50])
            query_embedding = self._embedder.embed_query(question)

            # Step 2: Vector similarity search
            # Retrieve more candidates if we're going to rerank
            search_top_k = initial_candidates if self._reranker else final_top_k
            logger.debug(
                "Searching vector store (top_k=%d, filter=%s)",
                search_top_k,
                filter_dict,
            )

            raw_results = self._vector_store.search(
                query_embedding=query_embedding,
                top_k=search_top_k,
                filter_dict=filter_dict,
            )

            if not raw_results:
                logger.info("No results found for query")
                return []

            # Step 3: Optional reranking
            if self._reranker:
                logger.debug("Reranking %d candidates", len(raw_results))
                raw_results = self._reranker.rerank(
                    query=question,
                    candidates=raw_results,
                    top_k=final_top_k,
                )
            else:
                # Limit to top_k if no reranking
                raw_results = raw_results[:final_top_k]

            # Step 4: Filter by minimum similarity
            filtered_results = [
                result for result in raw_results
                if result["score"] >= final_min_similarity
            ]

            if not filtered_results:
                logger.info(
                    "No results above threshold (min_similarity=%.2f)",
                    final_min_similarity,
                )
                return []

            logger.info(
                "Found %d results (filtered from %d)",
                len(filtered_results),
                len(raw_results),
            )

            # Step 5: Convert to Chunk objects
            chunks: list[Chunk] = []
            chunk_ids: list[str] = []

            for result in filtered_results:
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
                chunks.append(chunk)
                chunk_ids.append(result["id"])

            # Step 6: Enrich with context if requested
            context_before_list: list[str | None] = [None] * len(chunks)
            context_after_list: list[str | None] = [None] * len(chunks)

            if include_context:
                logger.debug("Enriching with neighboring context")
                context_before_list, context_after_list = self._enrich_with_context(
                    chunks=chunks,
                    chunk_ids=chunk_ids,
                )

            # Step 7: Build RetrievalResult objects
            retrieval_results: list[RetrievalResult] = []

            for i, (chunk, result) in enumerate(zip(chunks, filtered_results)):
                retrieval_result = RetrievalResult(
                    chunk=chunk,
                    similarity_score=result["score"],
                    rerank_score=result.get("rerank_score"),
                    context_before=context_before_list[i],
                    context_after=context_after_list[i],
                )
                retrieval_results.append(retrieval_result)

            return retrieval_results

        except Exception:
            logger.exception("Error during retrieval")
            raise

    def _enrich_with_context(
        self,
        chunks: list[Chunk],
        chunk_ids: list[str],
    ) -> tuple[list[str | None], list[str | None]]:
        """Fetch previous and next chunks for context.

        Args:
            chunks: List of Chunk objects.
            chunk_ids: Corresponding chunk IDs.

        Returns:
            Tuple of (context_before_list, context_after_list).
            Each list has the same length as chunks, with None for missing context.
        """
        context_before_list: list[str | None] = []
        context_after_list: list[str | None] = []

        for chunk_id in chunk_ids:
            try:
                neighbors = self._vector_store.get_neighboring_chunks(
                    chunk_id=chunk_id,
                    context_size=1,  # Get 1 chunk before and after
                )

                # Extract text from previous chunk
                previous_chunks = neighbors.get("previous", [])
                context_before = (
                    previous_chunks[-1]["text"] if previous_chunks else None
                )

                # Extract text from next chunk
                next_chunks = neighbors.get("next", [])
                context_after = next_chunks[0]["text"] if next_chunks else None

                context_before_list.append(context_before)
                context_after_list.append(context_after)

            except Exception:
                logger.warning(
                    "Failed to get context for chunk '%s', using None",
                    chunk_id,
                )
                context_before_list.append(None)
                context_after_list.append(None)

        return context_before_list, context_after_list
