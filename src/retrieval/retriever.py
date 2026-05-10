"""High-level retrieval service for question answering."""

import logging
from typing import Any

from src.config import RetrievalConfig
from src.embeddings.embedder import TextEmbedder
from src.models.chunk import Chunk
from src.models.query_result import RetrievalResult
from src.retrieval.bm25_store import BM25Store
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """High-level retrieval service for question answering.

    Orchestrates:
    - Query embedding
    - Vector similarity search
    - Optional BM25 keyword search
    - Hybrid search with fusion (RRF)
    - Optional reranking
    - Score filtering
    - Context enrichment

    Args:
        embedder: TextEmbedder instance.
        vector_store: VectorStore instance.
        config: RetrievalConfig with top_k, min_similarity, etc.
        bm25_store: Optional BM25Store for hybrid retrieval.
        reranker: Optional Reranker for precision improvement.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        config: RetrievalConfig,
        bm25_store: BM25Store | None = None,
        reranker: Any | None = None,  # Use Any to avoid circular import
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._config = config
        self._bm25_store = bm25_store
        self._reranker = reranker

    @staticmethod
    def _normalize_dedup_value(value: Any) -> str:
        """Normalize metadata values used for deduplication."""
        if value is None:
            return ""
        return " ".join(str(value).split()).casefold()

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
            # Determine if using hybrid search
            use_hybrid = (
                self._config.use_hybrid
                and self._bm25_store is not None
                and self._bm25_store.is_loaded
            )

            if use_hybrid:
                logger.debug("Using hybrid retrieval (vector + BM25)")
                raw_results = self._hybrid_search(
                    question=question,
                    top_k=initial_candidates if self._reranker else final_top_k,
                    filter_dict=filter_dict,
                )
            else:
                logger.debug("Using vector-only retrieval")
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

            # Step 4b: Deduplicate by text, position and logical source section.
            # Keep the highest-scoring copy (results are already score-sorted).
            # We intentionally use normalized book title + section path in addition
            # to book_id because duplicate ingestions can create distinct IDs for
            # the same logical source, which would otherwise leak duplicates.
            seen_texts: set[str] = set()
            seen_positions: set[tuple[str, int]] = set()
            seen_sections: set[tuple[str, str]] = set()
            deduped_results: list[dict] = []
            for result in filtered_results:
                meta = result["metadata"]
                normalized_book_id = self._normalize_dedup_value(meta.get("book_id", ""))
                normalized_book_title = self._normalize_dedup_value(
                    meta.get("book_title", "")
                )
                normalized_section_path = self._normalize_dedup_value(
                    meta.get("section_path", "")
                )
                position_key = (
                    normalized_book_id or normalized_book_title,
                    int(meta.get("chunk_index", -1)),
                )
                section_key = (normalized_book_title, normalized_section_path)
                text_key = self._normalize_dedup_value(result.get("text", ""))
                if (
                    position_key not in seen_positions
                    and text_key not in seen_texts
                    and (
                        not normalized_book_title
                        or not normalized_section_path
                        or section_key not in seen_sections
                    )
                ):
                    seen_positions.add(position_key)
                    seen_texts.add(text_key)
                    if normalized_book_title and normalized_section_path:
                        seen_sections.add(section_key)
                    deduped_results.append(result)
            if len(deduped_results) < len(filtered_results):
                logger.warning(
                    "Deduplicated %d → %d results (duplicate chunks detected; "
                    "consider re-ingesting affected books)",
                    len(filtered_results),
                    len(deduped_results),
                )
            filtered_results = deduped_results[:final_top_k]

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

    def _hybrid_search(
        self,
        question: str,
        top_k: int,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search combining vector and BM25 results.

        Uses Reciprocal Rank Fusion (RRF) to merge ranked lists from both
        retrieval methods.

        Args:
            question: The search query.
            top_k: Number of candidates to retrieve from each method.
            filter_dict: Optional metadata filters.

        Returns:
            Merged and re-ranked list of results.
        """
        # Get vector search results
        logger.debug("Hybrid search: performing vector search")
        query_embedding = self._embedder.embed_query(question)
        vector_results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict,
        )

        # Get BM25 search results
        logger.debug("Hybrid search: performing BM25 search")
        bm25_results = self._bm25_store.search(  # type: ignore
            query=question,
            top_k=top_k,
            filter_dict=filter_dict,
        )

        # Merge using Reciprocal Rank Fusion
        merged_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            vector_weight=self._config.vector_weight,
            bm25_weight=self._config.bm25_weight,
        )

        logger.debug(
            "Hybrid search: merged %d vector + %d BM25 → %d results",
            len(vector_results),
            len(bm25_results),
            len(merged_results),
        )

        return merged_results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Merge two ranked lists using Reciprocal Rank Fusion (RRF).

        RRF formula: score(d) = Σ weight / (k + rank(d))

        Where:
        - rank(d) is the rank of document d in a given list (1-indexed)
        - k is a constant (default 60, standard in RRF)
        - weight is the importance of each retrieval method

        Args:
            vector_results: Results from vector search.
            bm25_results: Results from BM25 search.
            vector_weight: Weight for vector search (default 0.7).
            bm25_weight: Weight for BM25 search (default 0.3).
            k: RRF constant (default 60).

        Returns:
            Merged list of results, sorted by RRF score (highest first).
        """
        # Build lookup dicts: chunk_id -> (rank, original_result)
        vector_ranks: dict[str, tuple[int, dict[str, Any]]] = {
            result["id"]: (rank + 1, result)
            for rank, result in enumerate(vector_results)
        }

        bm25_ranks: dict[str, tuple[int, dict[str, Any]]] = {
            result["id"]: (rank + 1, result)
            for rank, result in enumerate(bm25_results)
        }

        # Compute RRF scores for all unique chunk IDs
        all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores: dict[str, float] = {}

        for chunk_id in all_chunk_ids:
            score = 0.0

            # Add vector contribution
            if chunk_id in vector_ranks:
                rank, _ = vector_ranks[chunk_id]
                score += vector_weight / (k + rank)

            # Add BM25 contribution
            if chunk_id in bm25_ranks:
                rank, _ = bm25_ranks[chunk_id]
                score += bm25_weight / (k + rank)

            rrf_scores[chunk_id] = score

        # Sort by RRF score (highest first)
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        # Build final results list
        merged: list[dict[str, Any]] = []

        for chunk_id in sorted_ids:
            # Prefer vector result if available (has embedding-based metadata)
            if chunk_id in vector_ranks:
                _, result = vector_ranks[chunk_id]
            else:
                _, result = bm25_ranks[chunk_id]

            # Create new result dict with RRF score
            merged_result = {
                "id": result["id"],
                "text": result["text"],
                "metadata": result["metadata"],
                "score": rrf_scores[chunk_id],  # Use RRF score
                "original_vector_score": vector_ranks.get(chunk_id, (None, {}))[1].get("score"),
                "original_bm25_score": bm25_ranks.get(chunk_id, (None, {}))[1].get("score"),
            }
            merged.append(merged_result)

        return merged
