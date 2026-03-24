"""Tests for the Retriever service."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.config import RetrievalConfig
from src.embeddings.embedder import TextEmbedder
from src.models.chunk import Chunk
from src.models.query_result import RetrievalResult
from src.retrieval.bm25_store import BM25Store
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore


@pytest.fixture
def config() -> RetrievalConfig:
    """Default retrieval configuration for tests."""
    return RetrievalConfig(
        top_k=5,
        initial_candidates=20,
        min_similarity=0.3,
        use_reranker=False,
        use_hybrid=False,
        vector_weight=0.7,
        bm25_weight=0.3,
        bm25_dir="./db/bm25",
    )


@pytest.fixture
def hybrid_config() -> RetrievalConfig:
    """Hybrid retrieval configuration for tests."""
    return RetrievalConfig(
        top_k=5,
        initial_candidates=20,
        min_similarity=0.3,
        use_reranker=False,
        use_hybrid=True,
        vector_weight=0.7,
        bm25_weight=0.3,
        bm25_dir="./db/bm25",
    )


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock embedder that returns a fixed embedding."""
    embedder = MagicMock(spec=TextEmbedder)
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    return embedder


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Mock vector store with sample chunks."""
    store = MagicMock(spec=VectorStore)
    store.is_initialized = True

    # Default search results
    store.search.return_value = [
        {
            "id": "chunk-1",
            "score": 0.9,
            "text": "תוכן ראשון",
            "metadata": {
                "book_id": "book-1",
                "book_title": "ספר א",
                "book_author": "רבי א",
                "section_path": "סימן א",
                "section_type": "siman",
                "language": "he",
                "char_start": 0,
                "char_end": 100,
                "token_count": 20,
                "chunk_index": 0,
                "total_chunks_in_section": 3,
            },
        },
        {
            "id": "chunk-2",
            "score": 0.7,
            "text": "תוכן שני",
            "metadata": {
                "book_id": "book-1",
                "book_title": "ספר א",
                "book_author": "רבי א",
                "section_path": "סימן ב",
                "section_type": "siman",
                "language": "he",
                "char_start": 100,
                "char_end": 200,
                "token_count": 25,
                "chunk_index": 1,
                "total_chunks_in_section": 3,
            },
        },
        {
            "id": "chunk-3",
            "score": 0.5,
            "text": "תוכן שלישי",
            "metadata": {
                "book_id": "book-2",
                "book_title": "ספר ב",
                "book_author": "רבי ב",
                "section_path": "סימן א",
                "section_type": "siman",
                "language": "he",
                "char_start": 0,
                "char_end": 150,
                "token_count": 30,
                "chunk_index": 0,
                "total_chunks_in_section": 2,
            },
        },
    ]

    # Default neighboring chunks
    store.get_neighboring_chunks.return_value = {
        "previous": [
            {
                "id": "chunk-prev",
                "text": "תוכן קודם",
                "metadata": {"chunk_index": -1},
            }
        ],
        "next": [
            {
                "id": "chunk-next",
                "text": "תוכן הבא",
                "metadata": {"chunk_index": 1},
            }
        ],
    }

    return store


@pytest.fixture
def mock_reranker() -> MagicMock:
    """Mock reranker that adds rerank scores."""
    reranker = MagicMock(spec=Reranker)

    def rerank_side_effect(
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        # Add rerank scores and reverse order
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = 1.0 - (i * 0.1)

        # Reverse order to simulate reranking
        reranked = list(reversed(candidates))
        if top_k:
            reranked = reranked[:top_k]
        return reranked

    reranker.rerank.side_effect = rerank_side_effect
    return reranker


@pytest.fixture
def mock_bm25_store() -> MagicMock:
    """Mock BM25Store for hybrid retrieval tests."""
    store = MagicMock(spec=BM25Store)
    store.is_loaded = True

    # Default BM25 search results (different from vector results)
    store.search.return_value = [
        {
            "id": "chunk-3",  # Different from top vector result
            "score": 12.5,
            "text": "תוכן שלישי",
            "metadata": {
                "book_id": "book-2",
                "book_title": "ספר ב",
                "book_author": "רבי ב",
                "section_path": "סימן א",
                "section_type": "siman",
                "language": "he",
                "char_start": 0,
                "char_end": 150,
                "token_count": 30,
                "chunk_index": 0,
                "total_chunks_in_section": 2,
            },
        },
        {
            "id": "chunk-1",
            "score": 10.2,
            "text": "תוכן ראשון",
            "metadata": {
                "book_id": "book-1",
                "book_title": "ספר א",
                "book_author": "רבי א",
                "section_path": "סימן א",
                "section_type": "siman",
                "language": "he",
                "char_start": 0,
                "char_end": 100,
                "token_count": 20,
                "chunk_index": 0,
                "total_chunks_in_section": 3,
            },
        },
        {
            "id": "chunk-4",  # Unique to BM25
            "score": 8.7,
            "text": "תוכן רביעי",
            "metadata": {
                "book_id": "book-2",
                "book_title": "ספר ב",
                "book_author": "רבי ב",
                "section_path": "סימן ב",
                "section_type": "siman",
                "language": "he",
                "char_start": 150,
                "char_end": 250,
                "token_count": 28,
                "chunk_index": 1,
                "total_chunks_in_section": 2,
            },
        },
    ]

    return store


class TestRetrieverBasicSearch:
    """Test basic search functionality."""

    def test_search_with_default_parameters_returns_results(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test basic search returns results."""
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        results = retriever.search("מה הדין?")

        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].chunk.text == "תוכן ראשון"
        assert results[0].similarity_score == 0.9

    def test_search_with_empty_query_returns_empty_list(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test empty query returns empty list."""
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        results = retriever.search("")

        assert results == []
        mock_embedder.embed_query.assert_not_called()

    def test_search_with_no_vector_store_results_returns_empty_list(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test no results from vector store returns empty list."""
        mock_vector_store.search.return_value = []
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        results = retriever.search("שאלה")

        assert results == []

    def test_query_embedding_uses_correct_prefix(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test query embedding uses embed_query method."""
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        retriever.search("מה הדין?")

        mock_embedder.embed_query.assert_called_once_with("מה הדין?")

    def test_results_are_sorted_by_score_descending(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test results remain sorted by score (descending)."""
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        results = retriever.search("שאלה")

        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestRetrieverScoreFiltering:
    """Test score filtering behavior."""

    def test_min_similarity_filters_out_low_scores(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test min_similarity filters low-score results."""
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        results = retriever.search("שאלה", min_similarity=0.6)

        assert len(results) == 2  # Only scores 0.9 and 0.7 pass
        assert all(r.similarity_score >= 0.6 for r in results)

    def test_all_results_returned_when_min_similarity_zero(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test all results returned when min_similarity=0."""
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        results = retriever.search("שאלה", min_similarity=0.0)

        assert len(results) == 3


class TestRetrieverTopKControl:
    """Test top_k parameter controls result count."""

    def test_top_k_parameter_limits_results(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test top_k limits results."""
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        results = retriever.search("שאלה", top_k=2)

        # Vector store should be called with top_k=2 (no reranker)
        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs["top_k"] == 2

    def test_top_k_one_returns_single_best_result(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test top_k=1 returns single result."""
        # Adjust mock to return 1 result when top_k=1
        def search_side_effect(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            top_k = kwargs.get("top_k", 5)
            all_results = mock_vector_store.search.return_value
            return all_results[:top_k]

        mock_vector_store.search.side_effect = search_side_effect

        retriever = Retriever(mock_embedder, mock_vector_store, config)
        results = retriever.search("שאלה", top_k=1)

        assert len(results) == 1
        assert results[0].similarity_score == 0.9

    def test_top_k_larger_than_available_returns_all(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test top_k larger than available results returns all."""
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        results = retriever.search("שאלה", top_k=100)

        assert len(results) == 3  # Only 3 results available


class TestRetrieverMetadataFiltering:
    """Test metadata filtering."""

    def test_filter_dict_passed_to_vector_store(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test filter_dict is passed to vector store."""
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        filter_dict = {"book_id": "book-1"}
        retriever.search("שאלה", filter_dict=filter_dict)

        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs["filter_dict"] == filter_dict

    def test_filter_by_book_id_works(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test filtering by book_id."""
        # Mock filtered results
        mock_vector_store.search.return_value = [
            mock_vector_store.search.return_value[0],
            mock_vector_store.search.return_value[1],
        ]

        retriever = Retriever(mock_embedder, mock_vector_store, config)
        results = retriever.search("שאלה", filter_dict={"book_id": "book-1"})

        assert len(results) == 2
        assert all(r.chunk.book_id == "book-1" for r in results)


class TestRetrieverContextEnrichment:
    """Test neighboring context enrichment."""

    def test_include_context_true_fetches_neighboring_chunks(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test include_context=True fetches neighboring chunks."""
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        results = retriever.search("שאלה", include_context=True)

        # Should call get_neighboring_chunks for each result
        assert mock_vector_store.get_neighboring_chunks.call_count == 3
        assert results[0].context_before == "תוכן קודם"
        assert results[0].context_after == "תוכן הבא"

    def test_include_context_false_skips_context_fetching(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test include_context=False skips context fetching."""
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        results = retriever.search("שאלה", include_context=False)

        mock_vector_store.get_neighboring_chunks.assert_not_called()
        assert all(r.context_before is None for r in results)
        assert all(r.context_after is None for r in results)

    def test_missing_neighbors_handled_gracefully(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test missing neighbors result in None values."""
        mock_vector_store.get_neighboring_chunks.return_value = {
            "previous": [],
            "next": [],
        }

        retriever = Retriever(mock_embedder, mock_vector_store, config)
        results = retriever.search("שאלה", include_context=True)

        assert all(r.context_before is None for r in results)
        assert all(r.context_after is None for r in results)


class TestRetrieverReranking:
    """Test reranking functionality."""

    def test_reranker_called_when_provided(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_reranker: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test reranker is called when provided."""
        retriever = Retriever(mock_embedder, mock_vector_store, config, mock_reranker)

        retriever.search("שאלה")

        mock_reranker.rerank.assert_called_once()

    def test_reranked_scores_stored_in_rerank_score_field(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_reranker: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test reranked scores are stored."""
        retriever = Retriever(mock_embedder, mock_vector_store, config, mock_reranker)

        results = retriever.search("שאלה")

        # Reranker reverses order, so check rerank_score exists
        assert all(r.rerank_score is not None for r in results)

    def test_initial_candidates_used_when_reranker_provided(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        """Test initial_candidates used with reranker."""
        config = RetrievalConfig(
            top_k=5,
            initial_candidates=20,
            min_similarity=0.3,
            use_reranker=True,
        )
        retriever = Retriever(mock_embedder, mock_vector_store, config, mock_reranker)

        retriever.search("שאלה")

        # Vector store should be called with initial_candidates
        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs["top_k"] == 20


class TestRetrieverErrorHandling:
    """Test error handling."""

    def test_vector_store_not_initialized_raises_error(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test uninitialized vector store raises error."""
        mock_vector_store.is_initialized = False
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        with pytest.raises(RuntimeError, match="not initialized"):
            retriever.search("שאלה")

    def test_embedder_failure_propagates_exception(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test embedder failure propagates."""
        mock_embedder.embed_query.side_effect = RuntimeError("Embedding failed")
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        with pytest.raises(RuntimeError, match="Embedding failed"):
            retriever.search("שאלה")

    def test_context_enrichment_failure_uses_none(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        config: RetrievalConfig,
    ) -> None:
        """Test context enrichment failure results in None."""
        mock_vector_store.get_neighboring_chunks.side_effect = RuntimeError("Failed")
        retriever = Retriever(mock_embedder, mock_vector_store, config)

        # Should not raise, should use None for context
        results = retriever.search("שאלה", include_context=True)

        assert all(r.context_before is None for r in results)
        assert all(r.context_after is None for r in results)


class TestRetrieverHybridSearch:
    """Test hybrid retrieval with BM25 + vector search."""

    def test_hybrid_search_uses_both_stores(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_bm25_store: MagicMock,
        hybrid_config: RetrievalConfig,
    ) -> None:
        """Test hybrid search calls both vector store and BM25 store."""
        retriever = Retriever(
            mock_embedder,
            mock_vector_store,
            hybrid_config,
            bm25_store=mock_bm25_store,
        )

        results = retriever.search("שאלה")

        # Both stores should be called
        mock_vector_store.search.assert_called_once()
        mock_bm25_store.search.assert_called_once()
        assert len(results) > 0

    def test_hybrid_search_merges_results_from_both_sources(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_bm25_store: MagicMock,
        hybrid_config: RetrievalConfig,
    ) -> None:
        """Test hybrid search merges vector and BM25 results."""
        retriever = Retriever(
            mock_embedder,
            mock_vector_store,
            hybrid_config,
            bm25_store=mock_bm25_store,
        )

        results = retriever.search("שאלה", include_context=False)

        # Should have results from both sources merged
        result_ids = {r.chunk.id for r in results}
        # chunk-1 and chunk-3 appear in both, chunk-2 only in vector, chunk-4 only in BM25
        assert "chunk-1" in result_ids or "chunk-3" in result_ids

    def test_hybrid_disabled_when_bm25_not_loaded(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_bm25_store: MagicMock,
        hybrid_config: RetrievalConfig,
    ) -> None:
        """Test hybrid falls back to vector-only when BM25 not loaded."""
        mock_bm25_store.is_loaded = False
        retriever = Retriever(
            mock_embedder,
            mock_vector_store,
            hybrid_config,
            bm25_store=mock_bm25_store,
        )

        results = retriever.search("שאלה")

        # Only vector store should be called
        mock_vector_store.search.assert_called_once()
        mock_bm25_store.search.assert_not_called()
        assert len(results) > 0

    def test_hybrid_disabled_when_bm25_store_none(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        hybrid_config: RetrievalConfig,
    ) -> None:
        """Test hybrid falls back to vector-only when BM25 store is None."""
        retriever = Retriever(
            mock_embedder,
            mock_vector_store,
            hybrid_config,
            bm25_store=None,
        )

        results = retriever.search("שאלה")

        # Only vector store should be called
        mock_vector_store.search.assert_called_once()
        assert len(results) > 0

    def test_vector_only_search_when_hybrid_disabled_in_config(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_bm25_store: MagicMock,
        config: RetrievalConfig,  # Non-hybrid config
    ) -> None:
        """Test vector-only search when hybrid disabled in config."""
        retriever = Retriever(
            mock_embedder,
            mock_vector_store,
            config,
            bm25_store=mock_bm25_store,
        )

        results = retriever.search("שאלה")

        # Only vector store should be called
        mock_vector_store.search.assert_called_once()
        mock_bm25_store.search.assert_not_called()
        assert len(results) > 0

    def test_hybrid_search_passes_filters_to_both_stores(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_bm25_store: MagicMock,
        hybrid_config: RetrievalConfig,
    ) -> None:
        """Test hybrid search passes filter_dict to both stores."""
        retriever = Retriever(
            mock_embedder,
            mock_vector_store,
            hybrid_config,
            bm25_store=mock_bm25_store,
        )

        filter_dict = {"book_id": "book-1"}
        retriever.search("שאלה", filter_dict=filter_dict)

        # Both stores should receive the filter
        vector_call_args = mock_vector_store.search.call_args
        bm25_call_args = mock_bm25_store.search.call_args

        assert vector_call_args.kwargs["filter_dict"] == filter_dict
        assert bm25_call_args.kwargs["filter_dict"] == filter_dict

    def test_hybrid_search_respects_top_k(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_bm25_store: MagicMock,
        hybrid_config: RetrievalConfig,
    ) -> None:
        """Test hybrid search respects top_k parameter."""
        retriever = Retriever(
            mock_embedder,
            mock_vector_store,
            hybrid_config,
            bm25_store=mock_bm25_store,
        )

        retriever.search("שאלה", top_k=3)

        # Both stores should be called with top_k=3
        vector_call_args = mock_vector_store.search.call_args
        bm25_call_args = mock_bm25_store.search.call_args

        assert vector_call_args.kwargs["top_k"] == 3
        assert bm25_call_args.kwargs["top_k"] == 3

    def test_hybrid_search_with_reranker(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_bm25_store: MagicMock,
        mock_reranker: MagicMock,
        hybrid_config: RetrievalConfig,
    ) -> None:
        """Test hybrid search works with reranker."""
        retriever = Retriever(
            mock_embedder,
            mock_vector_store,
            hybrid_config,
            bm25_store=mock_bm25_store,
            reranker=mock_reranker,
        )

        results = retriever.search("שאלה")

        # Both stores should be called with initial_candidates
        vector_call_args = mock_vector_store.search.call_args
        bm25_call_args = mock_bm25_store.search.call_args

        assert vector_call_args.kwargs["top_k"] == hybrid_config.initial_candidates
        assert bm25_call_args.kwargs["top_k"] == hybrid_config.initial_candidates

        # Reranker should be called on merged results
        mock_reranker.rerank.assert_called_once()
        assert len(results) > 0

    def test_rrf_fusion_combines_rankings(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_bm25_store: MagicMock,
        hybrid_config: RetrievalConfig,
    ) -> None:
        """Test RRF properly combines rankings from both methods."""
        retriever = Retriever(
            mock_embedder,
            mock_vector_store,
            hybrid_config,
            bm25_store=mock_bm25_store,
        )

        results = retriever.search("שאלה", include_context=False)

        # Results should have scores
        assert all(r.similarity_score > 0 for r in results)
        # Scores should be sorted descending
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_search_empty_bm25_results(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_bm25_store: MagicMock,
        hybrid_config: RetrievalConfig,
    ) -> None:
        """Test hybrid search when BM25 returns no results."""
        mock_bm25_store.search.return_value = []
        retriever = Retriever(
            mock_embedder,
            mock_vector_store,
            hybrid_config,
            bm25_store=mock_bm25_store,
        )

        results = retriever.search("שאלה")

        # Should still return vector results
        assert len(results) > 0
        # All results should be from vector search
        result_ids = {r.chunk.id for r in results}
        assert "chunk-1" in result_ids or "chunk-2" in result_ids

    def test_hybrid_search_empty_vector_results(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_bm25_store: MagicMock,
        hybrid_config: RetrievalConfig,
    ) -> None:
        """Test hybrid search when vector search returns no results."""
        mock_vector_store.search.return_value = []
        retriever = Retriever(
            mock_embedder,
            mock_vector_store,
            hybrid_config,
            bm25_store=mock_bm25_store,
        )

        results = retriever.search("שאלה")

        # Should still return BM25 results
        assert len(results) > 0
        # All results should be from BM25 search
        result_ids = {r.chunk.id for r in results}
        assert "chunk-3" in result_ids or "chunk-4" in result_ids
