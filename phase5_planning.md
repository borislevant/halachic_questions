# Phase 5: Retrieval Service (Query & Retrieve Chunks)

## Context

Phases 1–4 are complete:
- ✓ Config, models, database
- ✓ Parsing and chunking
- ✓ Embeddings and vector store
- ✓ Full ingestion pipeline

Phase 5 builds the **retrieval service layer** — a high-level API that takes user questions, converts them to embeddings, searches the vector store, optionally reranks, and returns enriched retrieval results with neighboring context.

---

## Current State

### Already implemented (Phase 3)
- `VectorStore` in `src/retrieval/vector_store.py`:
  - `search()` — basic vector similarity search
  - `get_neighboring_chunks()` — fetch context before/after a chunk
  - Filtering by metadata (book_id, etc.)

### Missing (Phase 5)
- High-level `Retriever` class that orchestrates:
  1. Query text → embedding (with correct "query: " prefix)
  2. Vector search with configurable parameters
  3. Optional reranking with cross-encoder
  4. Score filtering (min_similarity threshold)
  5. Context enrichment (previous/next chunks)
  6. Return structured `RetrievalResult` objects
- Optional `Reranker` class for precision improvements
- Tests for retriever behavior

---

## Phase 5 Goals

1. Implement `Retriever` class — the main interface for question answering
2. (Optional) Implement `Reranker` class for cross-encoder reranking
3. Handle edge cases: no results, low scores, missing context
4. Add comprehensive tests for retrieval behavior
5. Export retriever API from package

---

## Architecture

```
User Question (text)
      ↓
[Retriever.search(question)]
      ↓
1. TextEmbedder.embed_query(question)  ← adds "query: " prefix
      ↓
2. VectorStore.search(embedding, top_k=initial_candidates)
      ↓
3. (Optional) Reranker.rerank(question, candidates)
      ↓
4. Filter by min_similarity threshold
      ↓
5. Enrich with neighboring context (VectorStore.get_neighboring_chunks)
      ↓
Return list[RetrievalResult]
```

---

## Files to Create/Modify

### 1) `src/retrieval/retriever.py` (new)

Main retrieval orchestrator.

**Class: `Retriever`**

```python
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
        reranker: Reranker | None = None,
    ):
        ...
    
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
        ...
    
    def _enrich_with_context(
        self,
        chunks: list[Chunk],
        chunk_ids: list[str],
    ) -> tuple[list[str | None], list[str | None]]:
        """Fetch previous and next chunks for context.
        
        Returns:
            (context_before_list, context_after_list)
        """
        ...
```

**Key implementation details:**
- Use `embedder.embed_query(question)` to get query embedding with "query: " prefix
- Retrieve `initial_candidates` from vector store (configurable, default 20)
- If reranker provided → rerank candidates → take top_k
- Filter by `min_similarity` threshold
- Enrich with context if `include_context=True`
- Convert to `RetrievalResult` objects with all metadata
- Handle empty results gracefully (return empty list, log warning)

### 2) `src/retrieval/reranker.py` (new, optional)

Cross-encoder reranking for improved precision.

**Class: `Reranker`**

```python
class Reranker:
    """Cross-encoder reranker for improving retrieval precision.
    
    Uses a cross-encoder model to score query-passage pairs.
    More accurate but slower than bi-encoders.
    
    Args:
        model_name: Cross-encoder model (default: ms-marco-multilingual-MiniLMv2-L6-v2).
        device: Device for inference ("cpu", "cuda", "mps", "auto").
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-multilingual-MiniLMv2-L6-v2",
        device: str = "auto",
    ):
        ...
    
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
        """
        ...
```

**Implementation notes:**
- Use `sentence-transformers.CrossEncoder` (lazy load on first use)
- Score all query-passage pairs in batch
- Add `rerank_score` to each candidate dict
- Sort by rerank_score (descending)
- Return top_k results

### 3) `src/retrieval/__init__.py` (modify)

Export new classes:
```python
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker  # optional
from src.retrieval.vector_store import VectorStore

__all__ = ["Retriever", "Reranker", "VectorStore"]
```

### 4) `tests/test_retriever.py` (new)

Comprehensive test suite for retrieval behavior.

---

## Implementation Steps (Execution Order)

### Step 1 — Retriever core logic
- Create `src/retrieval/retriever.py`
- Implement `Retriever.__init__` and `Retriever.search`
- Handle query embedding with correct prefix
- Call vector_store.search with initial_candidates
- Filter by min_similarity
- Return list[RetrievalResult]

### Step 2 — Context enrichment
- Implement `_enrich_with_context` method
- Use `vector_store.get_neighboring_chunks` for each result
- Handle missing neighbors gracefully (None values)

### Step 3 — Optional reranking
- Create `src/retrieval/reranker.py`
- Implement lazy model loading
- Implement batch reranking logic
- Integrate reranker in Retriever.search (if provided)

### Step 4 — Edge cases & error handling
- Empty query → warning + empty results
- No results from vector store → empty list
- Vector store not initialized → clear error
- Reranker model load failure → fallback to no reranking

### Step 5 — Tests
- Create `tests/test_retriever.py`
- Mock embedder, vector_store, and reranker
- Test all code paths and edge cases

### Step 6 — Package exports
- Update `src/retrieval/__init__.py`
- Verify imports work

---

## Test Plan (`tests/test_retriever.py`)

Target: ~18–22 tests

### Unit tests (with mocks)

**Basic search flow:**
1. ✓ Search with default parameters returns results
2. ✓ Search with empty query returns empty list
3. ✓ Search with no vector store results returns empty list
4. ✓ Query embedding uses correct "query: " prefix
5. ✓ Results are sorted by score (descending)

**Score filtering:**
6. ✓ min_similarity filters out low-score results
7. ✓ All results returned when min_similarity=0.0

**Top-k control:**
8. ✓ top_k parameter limits results
9. ✓ top_k=1 returns single best result
10. ✓ top_k larger than available results returns all

**Metadata filtering:**
11. ✓ filter_dict passed to vector_store.search
12. ✓ Filter by book_id works

**Context enrichment:**
13. ✓ include_context=True fetches neighboring chunks
14. ✓ include_context=False skips context fetching
15. ✓ Missing neighbors handled gracefully (None values)

**Reranking (optional, mock reranker):**
16. ✓ Reranker called when provided
17. ✓ Reranked scores stored in rerank_score field
18. ✓ Results sorted by rerank_score when reranker used

**Error handling:**
19. ✓ Vector store not initialized raises clear error
20. ✓ Embedder failure propagates exception
21. ✓ Reranker failure falls back to vector scores

### Integration test
22. ✓ End-to-end with real embedder + fake vector store
    - Add chunks to vector store
    - Search with question
    - Verify correct chunks returned with scores

---

## Data Flow Example

```python
# User code (later in Phase 6 or UI)
from src.retrieval import Retriever
from src.embeddings import TextEmbedder
from src.retrieval import VectorStore
from src.config import load_config

config = load_config()
embedder = TextEmbedder(config.embedding)
vector_store = VectorStore(config.storage.chroma_dir)
vector_store.initialize()

retriever = Retriever(
    embedder=embedder,
    vector_store=vector_store,
    config=config.retrieval,
)

# Search
results = retriever.search(
    question="מה הדין של ברכת הנהנין?",
    top_k=5,
    min_similarity=0.3,
)

for result in results:
    print(f"Score: {result.similarity_score:.3f}")
    print(f"Book: {result.chunk.book_title}")
    print(f"Section: {result.chunk.section_path}")
    print(f"Text: {result.chunk.text[:100]}...")
    if result.context_before:
        print(f"Context before: {result.context_before[:50]}...")
```

---

## Acceptance Criteria

Phase 5 is complete when:
- ✓ `Retriever` class works end-to-end
- ✓ Query embedding uses correct "query: " prefix
- ✓ Score filtering and top_k limiting work correctly
- ✓ Context enrichment adds previous/next chunks
- ✓ (Optional) Reranker improves precision when enabled
- ✓ All edge cases handled gracefully
- ✓ Test suite passes with good coverage
- ✓ Exports work from `src.retrieval` package

---

## Verification Commands

```bash
# Run retriever tests
pytest tests/test_retriever.py -v

# Run all tests
pytest tests/ -v

# Type check
mypy src/retrieval/retriever.py

# Lint
ruff check src/retrieval/

# Integration smoke test (after implementation)
python -c "
from src.config import load_config
from src.embeddings.embedder import TextEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever

config = load_config()
embedder = TextEmbedder(config.embedding)
vector_store = VectorStore(config.storage.chroma_dir, use_persistent=False)
vector_store.initialize()

retriever = Retriever(embedder, vector_store, config.retrieval)
print('Retriever initialized successfully')
"
```

---

## Notes / Scope Control

### In scope for Phase 5:
- Retrieval orchestration and context enrichment
- Score filtering and ranking
- Optional reranking support
- Comprehensive tests

### Out of scope (future phases):
- LLM answer generation (Phase 6)
- Citation parsing (Phase 6)
- UI components (Phase 7)
- Query history persistence (Phase 8)

### Dependencies
- Reranker requires `sentence-transformers` (already in requirements.txt)
- No new external dependencies needed

---

## Next Steps After Phase 5

Once Phase 5 is complete, we move to:
- **Phase 6**: Generation layer (prompt builder, summarizer, citation parser)
- **Phase 7**: Streamlit UI
- **Phase 8**: Query history
- **Phase 9-10**: Testing and polish
