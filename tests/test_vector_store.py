"""Tests for vector store behavior."""

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from src.models.chunk import Chunk


# chromadb currently fails to import on Python 3.14 in some environments
# due to pydantic.v1 compatibility. Stub the module before importing VectorStore.
fake_chromadb = types.ModuleType("chromadb")
fake_chromadb_config = types.ModuleType("chromadb.config")


class _FakeSettings:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _client_not_configured(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("Test stub client should be monkeypatched in fixture")


fake_chromadb.Client = object
fake_chromadb.Collection = object
fake_chromadb.EphemeralClient = _client_not_configured
fake_chromadb.PersistentClient = _client_not_configured
fake_chromadb_config.Settings = _FakeSettings
fake_chromadb.config = fake_chromadb_config

sys.modules["chromadb"] = fake_chromadb
sys.modules["chromadb.config"] = fake_chromadb_config

from src.retrieval.vector_store import VectorStore


class _FakeCollection:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def count(self) -> int:
        return len(self.items)

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for item_id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self.items[item_id] = {
                "id": item_id,
                "embedding": emb,
                "document": doc,
                "metadata": meta,
            }

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None,
        include: list[str],
    ) -> dict[str, Any]:
        del query_embeddings
        del include

        filtered = self._apply_where(where)
        ids = [item["id"] for item in filtered[:n_results]]
        documents = [item["document"] for item in filtered[:n_results]]
        metadatas = [item["metadata"] for item in filtered[:n_results]]
        distances = [0.1 + (i * 0.1) for i in range(len(ids))]

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        del include

        if ids is not None:
            selected = [self.items[item_id] for item_id in ids if item_id in self.items]
            return {
                "ids": [item["id"] for item in selected],
                "documents": [item["document"] for item in selected],
                "metadatas": [item["metadata"] for item in selected],
            }

        selected = self._apply_where(where)
        return {
            "ids": [item["id"] for item in selected],
            "documents": [item["document"] for item in selected],
            "metadatas": [item["metadata"] for item in selected],
        }

    def delete(self, ids: list[str]) -> None:
        for item_id in ids:
            self.items.pop(item_id, None)

    def _apply_where(self, where: dict[str, Any] | None) -> list[dict[str, Any]]:
        items = list(self.items.values())
        if where is None:
            return items

        if "$and" in where:
            conditions = where["$and"]
            return [
                item
                for item in items
                if all(self._matches(item["metadata"], condition) for condition in conditions)
            ]

        return [item for item in items if self._matches(item["metadata"], where)]

    @staticmethod
    def _matches(metadata: dict[str, Any], condition: dict[str, Any]) -> bool:
        for key, value in condition.items():
            if metadata.get(key) != value:
                return False
        return True


class _FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(self, name: str, metadata: dict[str, Any]) -> _FakeCollection:
        del metadata
        if name not in self.collections:
            self.collections[name] = _FakeCollection()
        return self.collections[name]

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    client = _FakeClient()

    monkeypatch.setattr("src.retrieval.vector_store.chromadb.EphemeralClient", lambda settings: client)
    monkeypatch.setattr("src.retrieval.vector_store.chromadb.PersistentClient", lambda path, settings: client)
    return client


@pytest.fixture
def store(fake_client: _FakeClient, tmp_path: Path) -> VectorStore:
    del fake_client
    vector_store = VectorStore(persist_directory=tmp_path, use_persistent=False)
    vector_store.initialize()
    return vector_store


def _make_chunk(
    chunk_id: str,
    text: str,
    book_id: str = "book-1",
    section_path: str = "אורח חיים > סימן א",
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        book_id=book_id,
        book_title="שולחן ערוך",
        book_author="רבי יוסף קארו",
        section_path=section_path,
        section_type="siman",
        chunk_index=chunk_index,
        total_chunks_in_section=3,
        language="he",
        char_start=0,
        char_end=len(text),
        token_count=max(1, len(text.split())),
    )


class TestVectorStoreGuards:
    def test_requires_initialize_before_add(self, tmp_path: Path) -> None:
        store = VectorStore(persist_directory=tmp_path, use_persistent=False)
        with pytest.raises(RuntimeError, match="not initialized"):
            store.add_chunks([], [])

    def test_requires_initialize_before_search(self, tmp_path: Path) -> None:
        store = VectorStore(persist_directory=tmp_path, use_persistent=False)
        with pytest.raises(RuntimeError, match="not initialized"):
            store.search([0.1, 0.2, 0.3])

    def test_add_chunks_length_mismatch_raises(self, store: VectorStore) -> None:
        chunks = [_make_chunk("c1", "טקסט")]
        embeddings = [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]
        with pytest.raises(ValueError, match="same length"):
            store.add_chunks(chunks, embeddings)


class TestVectorStoreOperations:
    def test_add_and_count(self, store: VectorStore) -> None:
        chunks = [_make_chunk("c1", "מקור ראשון"), _make_chunk("c2", "מקור שני", chunk_index=1)]
        embeddings = [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]

        store.add_chunks(chunks, embeddings)

        assert store.count() == 2

    def test_search_returns_similarity_scores(self, store: VectorStore) -> None:
        chunks = [_make_chunk("c1", "מקור ראשון"), _make_chunk("c2", "מקור שני", chunk_index=1)]
        embeddings = [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]
        store.add_chunks(chunks, embeddings)

        results = store.search(query_embedding=[0.2, 0.2, 0.2], top_k=2)

        assert len(results) == 2
        assert results[0]["id"] == "c1"
        assert results[0]["score"] == pytest.approx(0.9)
        assert results[1]["score"] == pytest.approx(0.8)

    def test_delete_by_book_id(self, store: VectorStore) -> None:
        chunks = [
            _make_chunk("a1", "טקסט א", book_id="book-a"),
            _make_chunk("a2", "טקסט ב", book_id="book-a", chunk_index=1),
            _make_chunk("b1", "טקסט ג", book_id="book-b"),
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
        store.add_chunks(chunks, embeddings)

        deleted = store.delete_by_book_id("book-a")

        assert deleted == 2
        assert store.count() == 1

    def test_get_chunk_by_id(self, store: VectorStore) -> None:
        chunk = _make_chunk("c1", "הלכה חשובה")
        store.add_chunks([chunk], [[0.1, 0.2, 0.3]])

        found = store.get_chunk_by_id("c1")
        missing = store.get_chunk_by_id("does-not-exist")

        assert found is not None
        assert found["id"] == "c1"
        assert "הלכה חשובה" in found["text"]
        assert missing is None

    def test_get_neighboring_chunks(self, store: VectorStore) -> None:
        chunks = [
            _make_chunk("c0", "קטע ראשון", chunk_index=0),
            _make_chunk("c1", "קטע שני", chunk_index=1),
            _make_chunk("c2", "קטע שלישי", chunk_index=2),
        ]
        embeddings = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]
        store.add_chunks(chunks, embeddings)

        context = store.get_neighboring_chunks("c1", context_size=1)

        assert [c["id"] for c in context["previous"]] == ["c0"]
        assert [c["id"] for c in context["next"]] == ["c2"]

    def test_reset_clears_collection(self, store: VectorStore) -> None:
        chunk = _make_chunk("c1", "טקסט לבדיקה")
        store.add_chunks([chunk], [[0.1, 0.2, 0.3]])
        assert store.count() == 1

        store.reset()

        assert store.count() == 0
        assert store.collection_name == "halachic_texts"
