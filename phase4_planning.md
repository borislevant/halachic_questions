# Phase 4: Full Ingestion Pipeline (Orchestration + Persistence)

## Context

Phases 1–3 are implemented in codebase:
- Config models and loader (`src/config.py`)
- Core models (`src/models/*` including `ParsedBook`, `Chunk`, `Book`)
- Parsing and chunking (`src/ingestion/parser.py`, `src/ingestion/chunker.py`)
- Embeddings + vector storage (`src/embeddings/embedder.py`, `src/retrieval/vector_store.py`)
- Initial pipeline skeleton exists (`src/ingestion/pipeline.py`)

This plan upgrades the current pipeline into a production-ready Phase 4 ingestion flow, aligned with the requirements and architecture.

---

## Current State Analysis (from existing code)

### Already implemented
- `IngestionPipeline.ingest_book()` parses → chunks → embeds → writes to Chroma.
- `IngestionReport` exists with warnings/errors and timing.
- `remove_book()` deletes from vector store by `book_id`.
- `create_ingestion_pipeline(config)` factory exists and initializes dependencies.

### Work to complete in Phase 4
1. **Connect pipeline to SQLite book registry**
   - Save/update/remove book metadata from `pipeline.py` in the `books` table.
2. **Add batch ingestion API**
   - Implement `ingest_directory()` as defined in the architecture.
3. **Complete re-index flow**
   - Ensure `reindex_book()` preserves the same `book_id` from start to end.
4. **Add ingestion pipeline tests**
   - Add tests for orchestration behavior (not only parser/chunker/embedder/vector store).
5. **Complete ingestion package exports**
   - Export pipeline interfaces from `src/ingestion/__init__.py`.

---

## Phase 4 Goals

1. Deliver robust end-to-end ingestion orchestration.
2. Persist book metadata and status in SQLite (`books` table).
3. Support batch ingest and reliable re-index/remove lifecycle.
4. Add unit/integration tests for pipeline behavior and failure modes.
5. Keep dependency injection pattern and type-safe interfaces.

---

## Files to Modify

### 1) `src/storage/database.py`
Add small repository-style functions for the `books` table (keep it simple, no new abstractions):
- `upsert_book(db_path, book: Book) -> None`
- `get_book_by_id(db_path, book_id: str) -> Book | None`
- `list_books(db_path) -> list[Book]`
- `delete_book(db_path, book_id: str) -> bool`
- `update_book_status(db_path, book_id: str, status: str) -> None`

Notes:
- Keep using existing `get_connection()`.
- Use parameterized SQL.
- Keep compatibility with current schema.

### 2) `src/ingestion/pipeline.py`
Complete and harden the pipeline API:
- `ingest_book(file_path, author="", book_id: str | None = None, show_progress=True)`
  - If `book_id` provided, use it (needed for true re-index).
  - Insert/update `books` row with `status='ingesting'` before heavy work.
  - On success: persist final metadata (`chunk_count`, `status='active'`, timestamps).
  - On failure: persist `status='error'` when possible.
- `ingest_directory(dir_path, author="", recursive=False, show_progress=True) -> list[IngestionReport]`
  - Scan supported formats from parser support map.
  - Continue processing on per-file errors; each file gets its own report.
- `remove_book(book_id)`
  - Delete from vector store + delete metadata row from SQLite.
  - Return structured success outcome.
- `reindex_book(file_path, book_id, author="", show_progress=True)`
  - Remove old vectors + metadata, then ingest with same `book_id`.

Optional (recommended) cleanup:
- Convert `IngestionReport` to `@dataclass` for clarity/typing.

### 3) `src/ingestion/__init__.py`
Export:
- `IngestionPipeline`
- `IngestionReport`
- `create_ingestion_pipeline`

### 4) `tests/test_pipeline.py` (new)
Add focused test suite for Phase 4 orchestration.

---

## Implementation Steps (Execution Order)

### Step 1 — SQLite Book Operations
- Implement CRUD helpers for books metadata in `storage/database.py`.
- Keep strict mapping between DB rows and `Book` model.

### Step 2 — Complete `ingest_book` lifecycle
- Add pre-ingest status write (`ingesting`).
- Keep existing parse/chunk/embed/store flow.
- Add post-ingest metadata persist (`active`, chunk count).
- Add failure path update (`error`) and preserve report details.

### Step 3 — Batch ingestion
- Implement `ingest_directory()` with deterministic order and optional recursion.
- Skip unsupported files safely (warning report entry, no crash).

### Step 4 — Correct remove/reindex semantics
- Ensure `remove_book` deletes both vector chunks and SQLite row.
- Ensure `reindex_book` reuses same `book_id`.

### Step 5 — Exports + tests
- Update ingestion package exports.
- Add `tests/test_pipeline.py` with mocks/fakes for fast deterministic tests.

---

## Test Plan (`tests/test_pipeline.py`)

Target: 14–20 focused tests.

### Unit tests (mock dependencies)
1. Successful `ingest_book` writes vectors and DB metadata.
2. Empty parsed text returns failure report and no vector writes.
3. Empty chunks returns failure report.
4. Embedding count mismatch returns failure report.
5. Parser exception captured into report errors.
6. Vector store exception captured into report errors.
7. `book_id` injection preserved in chunks and DB row.
8. Author override is applied.
9. Hebrew-text warning behavior for low Hebrew char count.

### Lifecycle tests
10. `remove_book` removes from vector store and DB.
11. `reindex_book` removes old and ingests with same `book_id`.
12. `get_stats` returns expected keys and values.

### Directory ingestion tests
13. `ingest_directory` processes supported files only.
14. `ingest_directory` continues after one file fails.
15. Recursive vs non-recursive behavior.

### Lightweight integration test
16. End-to-end with tiny real TXT fixture + fake embedder/vector store, validates full flow.

---

## Acceptance Criteria

Phase 4 is complete when:
- `ingest_book`, `ingest_directory`, `remove_book`, `reindex_book` all work end-to-end.
- `books` table reflects ingestion lifecycle and final chunk counts.
- Re-index keeps stable `book_id`.
- Per-file failures do not stop batch ingestion.
- New pipeline tests pass, and existing Phase 1–3 tests remain green.

---

## Verification Commands

```bash
pytest tests/test_pipeline.py -v
pytest tests/ -v
ruff check src/ tests/
mypy src/ingestion/pipeline.py src/storage/database.py
```

Optional smoke check:
```bash
python -c "from src.config import load_config; from src.ingestion.pipeline import create_ingestion_pipeline; p=create_ingestion_pipeline(load_config()); print(p.get_stats())"
```

---

## Notes / Scope Control

- Keep Phase 4 focused only on ingestion orchestration and persistence.
- Do not implement retrieval (`retriever.py`) or generation components in this phase.
- Do not change chunking/embedding logic unless required for pipeline correctness.
