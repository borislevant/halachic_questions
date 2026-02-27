# Phase 7: UI Module Planning

## Overview

Phase 7 implements the Streamlit web interface in `src/ui/`. It wires together all
previously built services (Retriever, Summarizer, IngestionPipeline) into a usable
desktop application served at `localhost:8501`.

Phase 8 (query history sidebar) is explicitly excluded from this phase.

**Inputs** (from prior phases):
- `Retriever` — searches ChromaDB for relevant chunks
- `Summarizer` — calls LLM to generate a grounded answer
- `IngestionPipeline` — parses, chunks, embeds, and stores a new book
- `database` — lists and deletes books from SQLite

**User-facing flows**:
1. Ask a Halachic question → see sources + AI answer
2. Upload and ingest a new book file
3. View and manage the ingested library

---

## Files to Create

```
src/ui/
├── __init__.py              (already exists, currently empty — update exports)
├── app.py                   # Main Streamlit application
├── services.py              # Cached service initialization
├── style.css                # RTL layout + Hebrew font rules
└── components/
    ├── __init__.py          (already exists — update exports)
    ├── answer.py            # Rendered answer + citations
    ├── sources.py           # Source result cards
    ├── ingestion.py         # File upload + ingestion widget
    └── book_list.py         # Library sidebar (books + delete)
```

---

## Layout Wireframe

```
┌────────────────┬──────────────────────────────────────┐
│   SIDEBAR      │   MAIN AREA                          │
│                │                                      │
│ 📚 Library     │  # Halachic Q&A                      │
│ ─────────────  │  ⚠️ Research only — not a Rabbinical  │
│ Book A  [🗑]   │     ruling.                           │
│  120 chunks    │                                      │
│ Book B  [🗑]   │  ┌──────────────────────────────┐   │
│  80 chunks     │  │  Enter your question here... │   │
│                │  └──────────────────────────────┘   │
│ ─────────────  │  [🔍 Search]                         │
│ Upload a Book  │                                      │
│ [file picker]  │  ── Answer ──────────────────────── │
│ [Ingest]       │  Answer text with [1][2] markers     │
│                │                                      │
│                │  Citations: [1] Book A — OC 1:1      │
│                │             [2] Book B — YD 2:3      │
│                │                                      │
│                │  ── Sources ─────────────────────── │
│                │  ▶ [1] Book A — OC 1:1  (0.92)      │
│                │    "Chunk text excerpt..."           │
│                │  ▶ [2] Book B — YD 2:3  (0.87)      │
└────────────────┴──────────────────────────────────────┘
```

---

## 1. `src/ui/services.py` — Cached Service Initialization

### Responsibility
Instantiates all heavy services exactly once per Streamlit server lifetime using
`@st.cache_resource`. Avoids reloading the embedding model (≈1–2 GB) on every rerender.

### Design

```python
import streamlit as st
from src.config import AppConfig, load_config
from src.embeddings.embedder import TextEmbedder
from src.generation.citation_parser import CitationParser
from src.generation.prompt_builder import PromptBuilder
from src.generation.summarizer import Summarizer
from src.ingestion.chunker import HalachicChunker
from src.ingestion.parser import BookParser
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore


@st.cache_resource(show_spinner="Loading models…")
def load_services() -> dict:
    """Load and cache all application services.

    Returns a dict with keys:
        config, retriever, summarizer, pipeline
    """
    config = load_config()
    embedder = TextEmbedder(config.embedding)
    vector_store = VectorStore(config.storage.chroma_dir)
    vector_store.initialize()
    retriever = Retriever(embedder, vector_store, config.retrieval)
    summarizer = Summarizer(
        config=config.generation,
        prompt_builder=PromptBuilder(),
        citation_parser=CitationParser(),
        anthropic_api_key=config.anthropic_api_key,
        openai_api_key=config.openai_api_key,
    )
    pipeline = IngestionPipeline(
        config=config,
        parser=BookParser(),
        chunker=HalachicChunker(config.chunking),
        embedder=embedder,
        vector_store=vector_store,
    )
    return {
        "config": config,
        "retriever": retriever,
        "summarizer": summarizer,
        "pipeline": pipeline,
    }
```

### Notes
- `@st.cache_resource` persists across rerenders and browser refreshes
- A single `TextEmbedder` and `VectorStore` instance is shared between
  `Retriever` and `IngestionPipeline` so embeddings stay consistent

---

## 2. `src/ui/app.py` — Main Application

### Responsibility
Entry point for `streamlit run src/ui/app.py`. Configures the page, loads CSS,
initializes `st.session_state`, and delegates rendering to component functions.

### Session State Keys

| Key | Type | Description |
|-----|------|-------------|
| `query_result` | `QueryResult \| None` | Most recent Q&A result |
| `is_searching` | `bool` | Search in progress |
| `search_error` | `str \| None` | Error message to display |

### Structure

```python
def main() -> None:
    st.set_page_config(
        page_title="Halachic Q&A",
        page_icon="📚",
        layout="wide",
    )
    _load_css()
    _init_session_state()

    services = load_services()

    with st.sidebar:
        render_book_list(services["config"])
        st.divider()
        render_ingestion(services["pipeline"], services["config"])

    render_header()
    render_question_form(services["retriever"], services["summarizer"], services["config"])

    if st.session_state.search_error:
        st.error(st.session_state.search_error)

    if st.session_state.query_result:
        render_answer(st.session_state.query_result)
        render_sources(st.session_state.query_result.sources)
```

### Search Action

```python
def _handle_search(question: str, retriever: Retriever, summarizer: Summarizer, config: AppConfig) -> None:
    """Run retrieval + generation, update session state."""
    st.session_state.search_error = None
    st.session_state.query_result = None

    sources = retriever.search(question)

    answer = None
    if sources:
        answer = summarizer.generate(question, sources)
        # answer may be None if all LLM providers fail — sources still shown

    st.session_state.query_result = QueryResult(
        question=question,
        sources=sources,
        answer=answer,
    )
```

The form uses `st.spinner` for feedback during both retrieval and generation.

---

## 3. `src/ui/components/answer.py` — Answer Display

### Responsibility
Renders the `GeneratedAnswer` when present, or a "sources only" fallback when the
LLM was unavailable. Highlights `[N]` citation markers in the answer text.

### Functions

```python
def render_answer(query_result: QueryResult) -> None:
    """Render the answer section of the results page."""

def _render_answer_text(answer: GeneratedAnswer) -> None:
    """Display answer text with [N] markers styled as superscripts."""

def _render_citations(citations: list[Citation]) -> None:
    """List citations below the answer; flag invalid ones."""

def _render_llm_unavailable_notice() -> None:
    """Show a warning when answer is None (LLM failed)."""

def _render_answer_metadata(answer: GeneratedAnswer) -> None:
    """Show model name, token count, latency in a collapsed expander."""
```

### Citation Highlighting

Transform `[N]` markers in the answer text into styled HTML superscripts before
rendering with `st.markdown(unsafe_allow_html=True)`:

```python
import re
_CITATION_RE = re.compile(r'\[(\d+)\]')

def _highlight_citations(text: str) -> str:
    return _CITATION_RE.sub(
        r'<sup class="citation">[\1]</sup>', text
    )
```

### Citation List Format

```
📖 Citations
  [1] Shulchan Aruch — Orach Chaim 1:1
  [2] Mishnah Berurah — Siman 1
  ⚠️ [3] Unknown source (not found in provided sources)
```

Invalid citations (`is_valid=False`) shown with a warning icon and dimmed style.

### LLM Unavailable Fallback

When `query_result.answer is None`:
```
⚠️ LLM generation failed — showing retrieved sources only.
   Check your API key configuration or try again.
```

---

## 4. `src/ui/components/sources.py` — Source Cards

### Responsibility
Renders each `RetrievalResult` as a card. Cards are collapsed by default to keep
the page compact; expanding reveals the full chunk text.

### Functions

```python
def render_sources(sources: list[RetrievalResult]) -> None:
    """Render all source cards under a 'Sources' heading."""

def _render_source_card(index: int, result: RetrievalResult) -> None:
    """Render a single expandable source card."""
```

### Card Structure

```
▶ [1] Shulchan Aruch — Orach Chaim 1:1    Score: 0.92
   ──────────────────────────────────────────────────
   [Full chunk text, Hebrew RTL if applicable]
   ──────────────────────────────────────────────────
   [context ↑] Preceding sentence...
   [context ↓] Following sentence...
```

Uses `st.expander` with the header line as the label. Score shown as a muted badge.
RTL class applied to the text container based on `chunk.language == "he"`.

---

## 5. `src/ui/components/ingestion.py` — Book Ingestion Widget

### Responsibility
Provides the sidebar "Upload a Book" form. Saves the uploaded file to a temp path,
calls `IngestionPipeline.ingest()`, and reports success or failure.

### Functions

```python
def render_ingestion(pipeline: IngestionPipeline, config: AppConfig) -> None:
    """Render the upload form and handle ingestion."""

def _save_uploaded_file(uploaded_file: UploadedFile, dest_dir: Path) -> Path:
    """Write Streamlit UploadedFile bytes to disk. Returns the saved path."""
```

### Form Layout

```
Upload a Book
─────────────
[File picker: PDF, TXT, DOCX, HTML]
Title (optional override): [text input]
Author (optional): [text input]
[Ingest] button
```

### Ingestion Flow

```
1. st.file_uploader → UploadedFile
2. Write bytes to config.storage.books_dir / filename
3. pipeline.ingest(source_path=saved_path, title=..., author=...)
4a. Success → st.success(f"✅ Ingested '{report.book_title}' ({report.chunks_created} chunks)")
4b. Failure → st.error(f"❌ Ingestion failed: {report.errors[0]}")
5. st.rerun() to refresh book list
```

Wrapped in `st.spinner("Ingesting book…")` during step 3.

---

## 6. `src/ui/components/book_list.py` — Library Sidebar

### Responsibility
Lists all ingested books from SQLite. Each entry shows title, author,
chunk count, and a delete button.

### Functions

```python
def render_book_list(config: AppConfig) -> None:
    """Render the library panel in the sidebar."""

def _render_book_row(book: Book, config: AppConfig) -> None:
    """Render one book entry with a delete button."""
```

### Layout

```
📚 Library  (N books)
─────────────────────
Shulchan Aruch
Yosef Karo · 340 chunks
                    [🗑]

Mishnah Berurah
Israel Meir Kagan · 220 chunks
                    [🗑]
─────────────────────
(empty state: "No books ingested yet.")
```

Delete button calls `database.delete_book()` then `st.rerun()`.

Note: Deleting from SQLite removes the metadata record. The ChromaDB vectors
are **not** deleted in Phase 7 (full vector purge is a Phase 8+ concern).
A warning is shown to the user.

---

## 7. `src/ui/style.css` — Styling

### Purpose
RTL layout support for Hebrew text, Hebrew-appropriate fonts,
and citation superscript styling.

### Rules

```css
/* Hebrew / RTL text blocks */
.rtl-text {
    direction: rtl;
    text-align: right;
    font-family: "Noto Sans Hebrew", "David", "Frank Ruehl", serif;
    line-height: 1.8;
}

/* Inline citation badges */
sup.citation {
    color: #1a73e8;
    font-size: 0.75em;
    font-weight: 600;
    cursor: default;
}

/* Invalid citation */
.citation-invalid {
    color: #d93025;
}

/* Source card score badge */
.score-badge {
    font-size: 0.8em;
    color: #5f6368;
    background: #f1f3f4;
    padding: 2px 6px;
    border-radius: 4px;
}

/* Disclaimer banner */
.disclaimer {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    padding: 8px 12px;
    font-size: 0.9em;
}
```

The CSS is loaded in `app.py` via:
```python
def _load_css() -> None:
    css_path = Path(__file__).parent / "style.css"
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
```

---

## 8. `src/ui/__init__.py` and `src/ui/components/__init__.py`

### `src/ui/__init__.py`
```python
"""Streamlit UI package."""
```
(Stays minimal — `app.py` is the entry point, not imported as a module.)

### `src/ui/components/__init__.py`
```python
"""UI component functions."""

from src.ui.components.answer import render_answer
from src.ui.components.book_list import render_book_list
from src.ui.components.ingestion import render_ingestion
from src.ui.components.sources import render_sources

__all__ = ["render_answer", "render_book_list", "render_ingestion", "render_sources"]
```

---

## State Management Summary

All mutable state lives in `st.session_state`. Components are pure render functions
that read from session state and may call `st.rerun()` to trigger a full re-render.

```
session_state.query_result   → QueryResult | None   (set by _handle_search)
session_state.is_searching   → bool                 (spinner guard)
session_state.search_error   → str | None            (error display)
```

---

## Error Handling in the UI

| Scenario | Behaviour |
|---|---|
| Vector store empty (no books) | `st.info("Ingest at least one book to begin.")` |
| No sources above threshold | Show `st.warning` + no answer |
| LLM all providers fail | Show sources + `st.warning` about LLM unavailability |
| Ingestion parse error | `st.error` with error message from `IngestionReport.errors` |
| File already ingested | `IngestionPipeline` skips (idempotent); UI shows success |

---

## Config Changes Required

None. All paths and settings already exist in `AppConfig`.

---

## Dependencies

All already in `requirements.txt`:
- `streamlit>=1.38`

No new packages needed.

---

## Implementation Order Within Phase 7

Build in this order — each step is independently runnable:

1. `services.py` — service wiring (can be tested by importing without running Streamlit)
2. `style.css` — static file, no logic
3. `components/book_list.py` — read-only, simplest component
4. `components/ingestion.py` — file upload + pipeline call
5. `components/sources.py` — pure display of retrieval results
6. `components/answer.py` — most complex (citation highlighting, fallback)
7. `app.py` — wires everything together; app is runnable after this step

---

## Integration Verification (manual, no new tests)

Phase 7 is UI code; unit-testing Streamlit components requires additional tooling
not in the current stack. Verification is done by running the app:

```bash
python run.py
# or
streamlit run src/ui/app.py
```

Checklist:
- [ ] App loads without error on cold start (model download on first run)
- [ ] Ingesting a sample TXT file produces a success banner and appears in library
- [ ] Asking a question against an ingested book returns sources
- [ ] Answer appears with [N] markers when LLM key is configured
- [ ] Warning appears (no crash) when LLM key is absent
- [ ] Deleting a book removes it from the sidebar
- [ ] Hebrew text renders RTL in source cards
- [ ] Disclaimer is visible on page load
