"""Cross-encoder reranking for improved retrieval precision."""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for improving retrieval precision.

    Uses a cross-encoder model to score query-passage pairs.
    More accurate but slower than bi-encoders.

    Args:
        model_name: Cross-encoder model (default: BAAI bge-reranker multilingual).
        device: Device for inference ("cpu", "cuda", "mps", "auto").
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._device_str = device
        self._model: Any = None  # Lazy loaded
        self._device: str | None = None

    def _load_model(self) -> None:
        """Lazy-load the cross-encoder model on first use."""
        if self._model is not None:
            return

        device = self._determine_device()
        logger.info(
            "Loading cross-encoder model '%s' on device '%s'",
            self._model_name,
            device,
        )

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name,
                device=device,
                max_length=512,
            )
            self._device = device
            logger.info("Cross-encoder model loaded successfully")

        except ImportError:
            logger.exception(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception:
            logger.exception("Failed to load cross-encoder model")
            raise

    def _determine_device(self) -> str:
        """Determine the best available device for the model.

        Returns:
            Device string: "cuda", "mps", or "cpu".
        """
        if self._device_str == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self._device_str

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using cross-encoder scoring.

        Args:
            query: User's question.
            candidates: List of dicts with 'id', 'text', 'score', 'metadata'.
            top_k: Number of results to return (default: all candidates).

        Returns:
            Reranked list of candidates with 'rerank_score' added to each.
            Sorted by rerank_score (descending).
        """
        if not candidates:
            return []

        if self._model is None:
            self._load_model()

        if top_k is None:
            top_k = len(candidates)

        try:
            # Prepare query-passage pairs for cross-encoder
            pairs = [[query, candidate["text"]] for candidate in candidates]

            logger.debug("Reranking %d candidates", len(pairs))

            # Score all pairs in batch
            scores = self._model.predict(pairs, show_progress_bar=False)

            # Add rerank scores to candidates
            for candidate, score in zip(candidates, scores):
                # Convert numpy float to Python float
                candidate["rerank_score"] = float(score)

            # Sort by rerank score (descending)
            reranked = sorted(
                candidates,
                key=lambda x: x["rerank_score"],
                reverse=True,
            )

            # Return top_k results
            return reranked[:top_k]

        except Exception:
            logger.exception("Failed to rerank candidates")
            # On error, return original candidates without reranking
            logger.warning("Falling back to original vector scores")
            return candidates[:top_k]
