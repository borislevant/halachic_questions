"""Tests for text embedding service."""

from typing import Any

import pytest

from src.config import EmbeddingConfig
from src.embeddings.embedder import TextEmbedder


class _FakeArray:
    def __init__(self, data: Any) -> None:
        self._data = data

    def tolist(self) -> Any:
        return self._data


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.calls: list[dict[str, Any]] = []

    def encode(self, texts: Any, **kwargs: Any) -> _FakeArray:
        self.calls.append({"texts": texts, "kwargs": kwargs})

        if isinstance(texts, str):
            return _FakeArray([0.11, 0.22, 0.33])

        vectors = [[float(i), float(i + 1), float(i + 2)] for i, _ in enumerate(texts)]
        return _FakeArray(vectors)

    def get_sentence_embedding_dimension(self) -> int:
        return 3


@pytest.fixture
def config() -> EmbeddingConfig:
    return EmbeddingConfig(model="test-model", device="cpu", batch_size=8)


@pytest.fixture
def embedder(config: EmbeddingConfig, monkeypatch: pytest.MonkeyPatch) -> TextEmbedder:
    def _factory(model_name: str, device: str) -> _FakeSentenceTransformer:
        return _FakeSentenceTransformer(model_name=model_name, device=device)

    monkeypatch.setattr("src.embeddings.embedder.SentenceTransformer", _factory)
    return TextEmbedder(config=config)


class TestTextEmbedder:
    def test_embed_single_text_adds_passage_prefix(self, embedder: TextEmbedder) -> None:
        result = embedder.embed("שלום עולם")

        assert len(result) == 1
        model = embedder._model
        assert model is not None
        assert model.calls[0]["texts"] == ["passage: שלום עולם"]

    def test_embed_keeps_existing_prefix(self, embedder: TextEmbedder) -> None:
        embedder.embed(["query: שאלה", "passage: מקור"])

        model = embedder._model
        assert model is not None
        assert model.calls[0]["texts"] == ["query: שאלה", "passage: מקור"]

    def test_embed_empty_list_returns_empty(self, embedder: TextEmbedder) -> None:
        assert embedder.embed([]) == []

    def test_embed_query_uses_query_prefix(self, embedder: TextEmbedder) -> None:
        result = embedder.embed_query("האם מותר")

        assert result == [0.11, 0.22, 0.33]
        model = embedder._model
        assert model is not None
        assert model.calls[0]["texts"] == "query: האם מותר"

    def test_get_embedding_dimension(self, embedder: TextEmbedder) -> None:
        assert embedder.get_embedding_dimension() == 3

    def test_model_loads_only_once(self, embedder: TextEmbedder) -> None:
        embedder.embed("טקסט ראשון")
        first_model = embedder._model

        embedder.embed("טקסט שני")
        second_model = embedder._model

        assert first_model is second_model


class TestDetermineDevice:
    def test_auto_prefers_cuda(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = EmbeddingConfig(model="test", device="auto", batch_size=4)
        embedder = TextEmbedder(config=config)

        monkeypatch.setattr("src.embeddings.embedder.torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("src.embeddings.embedder.torch.backends.mps.is_available", lambda: False)

        assert embedder._determine_device() == "cuda"

    def test_auto_uses_cpu_when_no_accelerator(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = EmbeddingConfig(model="test", device="auto", batch_size=4)
        embedder = TextEmbedder(config=config)

        monkeypatch.setattr("src.embeddings.embedder.torch.cuda.is_available", lambda: False)
        monkeypatch.setattr("src.embeddings.embedder.torch.backends.mps.is_available", lambda: False)

        assert embedder._determine_device() == "cpu"

    def test_explicit_device_is_respected(self) -> None:
        config = EmbeddingConfig(model="test", device="cpu", batch_size=4)
        embedder = TextEmbedder(config=config)

        assert embedder._determine_device() == "cpu"
