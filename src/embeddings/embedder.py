"""Text embedding service for multilingual semantic search."""

import logging
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from src.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Generates embeddings for Hebrew, Aramaic, English, and mixed-language texts.

    Uses sentence-transformers with a multilingual model to produce dense
    vector representations suitable for semantic similarity search.

    Args:
        config: EmbeddingConfig with model name, device, and batch size settings.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None
        self._device: str | None = None

    def _load_model(self) -> None:
        """Lazy-load the embedding model on first use.

        Determines the device (CPU, CUDA, or MPS) and loads the model.
        This is called automatically on the first embed() call.
        """
        if self._model is not None:
            return

        device = self._determine_device()
        logger.info(
            "Loading embedding model '%s' on device '%s'",
            self._config.model,
            device,
        )

        try:
            self._model = SentenceTransformer(
                self._config.model,
                device=device,
            )
            self._device = device
            logger.info("Model loaded successfully")
        except Exception:
            logger.exception("Failed to load embedding model")
            raise

    def _determine_device(self) -> str:
        """Determine the best available device for the model.

        Returns:
            Device string: "cuda", "mps", or "cpu".
        """
        if self._config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self._config.device

    def embed(self, texts: str | list[str], show_progress: bool = False) -> list[list[float]]:
        """Generate embeddings for one or more texts.

        Args:
            texts: A single text string or a list of text strings.
            show_progress: Whether to display a progress bar for batch processing.

        Returns:
            List of embedding vectors. Each embedding is a list of floats.
            For a single text input, returns a list with one embedding.

        Raises:
            RuntimeError: If model loading fails.
        """
        if self._model is None:
            self._load_model()

        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        # Add instruction prefix for E5 models
        # E5 models perform better with "query: " or "passage: " prefixes
        prefixed_texts = [self._add_instruction_prefix(text) for text in texts]

        try:
            embeddings = self._model.encode(  # type: ignore[union-attr]
                prefixed_texts,
                batch_size=self._config.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
            )

            # Convert numpy arrays to lists for serialization
            return embeddings.tolist()  # type: ignore[union-attr]

        except Exception:
            logger.exception("Failed to generate embeddings")
            raise

    def _add_instruction_prefix(self, text: str) -> str:
        """Add instruction prefix for E5 models.

        E5 models are trained with task-specific prefixes. For retrieval:
        - Use "query: " for search queries
        - Use "passage: " for documents/chunks

        Since this embedder is used for both, we default to "passage: "
        for chunks and let the retriever add "query: " for queries.

        Args:
            text: The raw text.

        Returns:
            Text with instruction prefix.
        """
        # Check if already prefixed (to avoid double-prefixing)
        if text.startswith(("query: ", "passage: ")):
            return text

        # Default to passage prefix for chunk embedding
        return f"passage: {text}"

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding specifically for a search query.

        This is a convenience method that adds the "query: " prefix
        for E5 models.

        Args:
            query: The search query text.

        Returns:
            A single embedding vector as a list of floats.
        """
        if self._model is None:
            self._load_model()

        prefixed_query = f"query: {query}"

        try:
            embedding = self._model.encode(  # type: ignore[union-attr]
                prefixed_query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            return embedding.tolist()  # type: ignore[union-attr]

        except Exception:
            logger.exception("Failed to generate query embedding")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.

        Returns:
            The embedding dimension (e.g., 1024 for multilingual-e5-large).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            self._load_model()

        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]

    @property
    def model_name(self) -> str:
        """Get the name of the loaded model."""
        return self._config.model

    @property
    def device(self) -> str:
        """Get the device the model is running on.

        Returns:
            Device string, or "not loaded" if model hasn't been initialized.
        """
        return self._device or "not loaded"
