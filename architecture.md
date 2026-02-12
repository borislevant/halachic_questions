# Halachic Q&A Desktop Application — Architecture Document

## 1. Project Overview

### 1.1 Purpose
A desktop application that enables users to ask questions about Jewish Halacha (religious law). The system searches through a curated library of Halachic books, retrieves the most relevant source passages, displays them to the user, and generates an AI-powered summary answer grounded in those sources.

### 1.2 Core Principles
- **Source fidelity**: Every answer must be traceable to specific book/chapter/section references.
- **Hebrew-first**: The system must handle Hebrew, Aramaic, and mixed-script texts natively (RTL support, proper tokenization).
- **Offline-capable**: The core retrieval pipeline works locally. Only the LLM summarization step requires an API call (with an optional local LLM fallback).
- **Cloud-ready**: All components are chosen to have direct cloud-scale equivalents for Phase 2.

### 1.3 High-Level Flow
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  User types  │────▶│  Embed the   │────▶│  Vector      │────▶│  Display     │
│  a question  │     │  question    │     │  similarity  │     │  top sources │
└──────────────┘     └──────────────┘     │  search      │     └──────┬───────┘
                                          └──────────────┘            │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │  LLM summary │
                                                               │  with cited  │
                                                               │  sources     │
                                                               └──────────────┘
```

---

## 2. System Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DESKTOP APPLICATION                          │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     PRESENTATION LAYER                        │  │
│  │                                                               │  │
│  │   Streamlit Web UI (localhost)                                │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │  │
│  │   │ Question     │  │ Sources     │  │ Summary Answer      │  │  │
│  │   │ Input Panel  │  │ Display     │  │ Panel               │  │  │
│  │   │              │  │ Panel       │  │                     │  │  │
│  │   └─────────────┘  └─────────────┘  └─────────────────────┘  │  │
│  │   ┌─────────────┐  ┌─────────────────────────────────────┐   │  │
│  │   │ Book Manager │  │ Settings / Configuration            │   │  │
│  │   │ (add/remove) │  │ (API keys, model selection, etc.)   │   │  │
│  │   └─────────────┘  └─────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     APPLICATION LAYER                         │  │
│  │                                                               │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │  │
│  │   │ Query        │  │ Retriever    │  │ Summarizer       │   │  │
│  │   │ Engine       │  │ Service      │  │ Service          │   │  │
│  │   │              │  │              │  │                  │   │  │
│  │   │ - preprocess │  │ - vector     │  │ - prompt builder │   │  │
│  │   │ - embed      │  │   search     │  │ - LLM call       │   │  │
│  │   │ - rerank     │  │ - filtering  │  │ - citation parse │   │  │
│  │   └──────────────┘  └──────────────┘  └──────────────────┘   │  │
│  │                                                               │  │
│  │   ┌──────────────────────────────────────────────────────┐    │  │
│  │   │ Ingestion Pipeline                                    │    │  │
│  │   │                                                       │    │  │
│  │   │ - Book parser (PDF, TXT, DOCX, HTML)                 │    │  │
│  │   │ - Structure-aware chunker (Sefer → Siman → Seif)     │    │  │
│  │   │ - Metadata extractor                                  │    │  │
│  │   │ - Embedding generator                                 │    │  │
│  │   │ - Vector DB writer                                    │    │  │
│  │   └──────────────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                       DATA LAYER                              │  │
│  │                                                               │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │  │
│  │   │ ChromaDB     │  │ SQLite       │  │ File System      │   │  │
│  │   │ (vectors +   │  │ (metadata,   │  │ (raw books,      │   │  │
│  │   │  embeddings) │  │  settings,   │  │  config files)   │   │  │
│  │   │              │  │  history)    │  │                  │   │  │
│  │   └──────────────┘  └──────────────┘  └──────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (API calls)
                 ┌─────────────────────────┐
                 │  External Services       │
                 │  - Anthropic Claude API  │
                 │  - (optional) OpenAI API │
                 └─────────────────────────┘
```

### 2.2 Directory Structure

```
halachic-qa/
├── README.md
├── requirements.txt
├── config.yaml                     # App configuration
├── .env                            # API keys (gitignored)
├── run.py                          # Entry point
│
├── data/
│   ├── books/                      # Raw book files (PDF, TXT, DOCX)
│   └── processed/                  # Chunked + metadata JSON files
│
├── db/
│   ├── chroma/                     # ChromaDB persistent storage
│   └── app.db                      # SQLite for app state
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py                   # Configuration loader (YAML + env)
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py               # File format parsers (PDF, TXT, DOCX, HTML)
│   │   ├── chunker.py              # Structure-aware text chunking
│   │   ├── metadata.py             # Metadata extraction (book, siman, seif)
│   │   └── pipeline.py             # Orchestrates full ingestion flow
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py             # Embedding model wrapper
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py         # ChromaDB interface
│   │   ├── retriever.py            # Search + filtering + reranking
│   │   └── reranker.py             # Optional cross-encoder reranker
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py       # Constructs LLM prompts with sources
│   │   ├── summarizer.py           # LLM API client (Claude / OpenAI / local)
│   │   └── citation_parser.py      # Extracts and validates citations
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── book.py                 # Book dataclass
│   │   ├── chunk.py                # Chunk dataclass
│   │   ├── query_result.py         # Search result dataclass
│   │   └── answer.py               # Final answer dataclass
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── database.py             # SQLite operations
│   │   └── history.py              # Query history management
│   │
│   └── ui/
│       ├── __init__.py
│       ├── app.py                  # Main Streamlit app
│       ├── components/
│       │   ├── __init__.py
│       │   ├── question_input.py   # Question input component
│       │   ├── sources_display.py  # Source passages display
│       │   ├── answer_display.py   # Summary answer display
│       │   ├── book_manager.py     # Book add/remove UI
│       │   ├── history_sidebar.py  # Past queries sidebar
│       │   └── settings_page.py    # Settings UI
│       └── styles/
│           └── custom.css          # RTL support, Hebrew fonts
│
└── tests/
    ├── __init__.py
    ├── test_chunker.py
    ├── test_retriever.py
    ├── test_summarizer.py
    └── fixtures/
        └── sample_texts/           # Test Hebrew text samples
```

---

## 3. Detailed Component Specifications

### 3.1 Ingestion Pipeline

#### 3.1.1 Book Parser (`src/ingestion/parser.py`)

**Responsibility**: Convert raw book files into clean text with positional metadata.

**Supported formats**:
| Format | Library | Notes |
|--------|---------|-------|
| PDF | `pymupdf` (fitz) | Best for Hebrew PDFs; preserves layout |
| TXT / Markdown | Built-in | Direct read with encoding detection |
| DOCX | `python-docx` | Preserves heading structure |
| HTML | `beautifulsoup4` | For Sefaria or online sources |

**Interface**:
```python
class BookParser:
    def parse(self, file_path: Path) -> ParsedBook:
        """Returns ParsedBook with raw text and structural hints."""

@dataclass
class ParsedBook:
    title: str
    author: str
    language: str              # "he", "arc" (Aramaic), "en", "mixed"
    raw_text: str
    sections: list[Section]    # Detected structural divisions
    source_path: Path
```

#### 3.1.2 Structure-Aware Chunker (`src/ingestion/chunker.py`)

**Responsibility**: Split parsed books into semantically meaningful chunks that respect Halachic text structure.

**Chunking strategy** (priority order):
1. **Structural chunking**: Split on detected section boundaries (סימן, סעיף, הלכה, etc.)
2. **Paragraph chunking**: If no structure detected, split on paragraph boundaries.
3. **Sliding window fallback**: For unstructured text, use 512-token chunks with 100-token overlap.

**Hebrew structure patterns to detect**:
```python
STRUCTURE_PATTERNS = {
    "siman": r"סימן\s+[א-ת]{1,4}",           # Siman א, Siman ב, etc.
    "seif": r"סעיף\s+[א-ת]{1,4}",             # Seif within a Siman
    "halacha": r"הלכה\s+[א-ת]{1,4}",          # Halacha numbering
    "chapter": r"פרק\s+[א-ת]{1,4}",           # Chapter
    "paragraph": r"\n\s*\n",                     # Double newline
    "siman_katan": r"ס[\"״]ק\s+[א-ת]{1,4}",  # Sub-section
}
```

**Chunk metadata**:
```python
@dataclass
class Chunk:
    id: str                    # UUID
    text: str                  # The actual text content
    book_title: str            # e.g., "שולחן ערוך"
    book_author: str           # e.g., "רבי יוסף קארו"
    section_path: str          # e.g., "אורח חיים > סימן שכח > סעיף ב"
    section_type: str          # "siman", "seif", "halacha", etc.
    chunk_index: int           # Position within the section
    total_chunks_in_section: int
    language: str
    char_start: int            # Position in original text
    char_end: int
    token_count: int           # Approximate token count
```

**Chunk size targets**:
- **Target**: 300–600 tokens per chunk
- **Maximum**: 800 tokens
- **Minimum**: 50 tokens (merge small chunks with neighbors)
- **Overlap**: 50 tokens when splitting within a section

#### 3.1.3 Ingestion Orchestrator (`src/ingestion/pipeline.py`)

```python
class IngestionPipeline:
    def __init__(self, parser, chunker, embedder, vector_store, db):
        ...

    def ingest_book(self, file_path: Path, book_metadata: dict) -> IngestionReport:
        """
        Full pipeline:
        1. Parse the book file
        2. Chunk with structure awareness
        3. Generate embeddings for all chunks
        4. Store in vector DB with metadata
        5. Record in SQLite book registry
        6. Return report (chunk count, errors, warnings)
        """

    def ingest_directory(self, dir_path: Path) -> list[IngestionReport]:
        """Batch ingest all books in a directory."""

    def remove_book(self, book_id: str):
        """Remove a book and all its chunks from the system."""
```

---

### 3.2 Embedding System

#### 3.2.1 Embedder (`src/embeddings/embedder.py`)

**Primary model**: `intfloat/multilingual-e5-large`
- 1024-dimensional embeddings
- Excellent Hebrew support
- Runs locally on CPU (slower) or GPU (fast)
- ~2.2GB model size

**Fallback model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- 768-dimensional embeddings
- Lighter weight (~1GB)
- Slightly lower quality for Hebrew

```python
class Embedder:
    def __init__(self, model_name: str, device: str = "auto"):
        """Load the embedding model. Device: 'cpu', 'cuda', or 'auto'."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a batch of texts efficiently."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a query (may use different prefix for E5 models).
        E5 models require 'query: ' prefix for queries and
        'passage: ' prefix for documents."""
```

**Important note for E5 models**: Query embeddings must be prefixed with `"query: "` and document embeddings with `"passage: "` for optimal performance.

---

### 3.3 Retrieval System

#### 3.3.1 Vector Store (`src/retrieval/vector_store.py`)

**Technology**: ChromaDB (persistent local storage)

```python
class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "halachic_chunks"):
        """Initialize ChromaDB with persistent storage."""

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]):
        """Add embedded chunks to the vector store."""

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None
    ) -> list[SearchResult]:
        """
        Semantic search with optional metadata filters.
        Filters example: {"book_title": "שולחן ערוך", "section_type": "seif"}
        """

    def delete_by_book(self, book_title: str):
        """Remove all chunks for a specific book."""

    def get_collection_stats(self) -> dict:
        """Return stats: total chunks, books indexed, etc."""
```

#### 3.3.2 Retriever (`src/retrieval/retriever.py`)

```python
class Retriever:
    def __init__(self, vector_store, embedder, reranker=None):
        ...

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        initial_candidates: int = 20,
        book_filter: list[str] | None = None,
        min_score: float = 0.3
    ) -> list[RetrievalResult]:
        """
        Full retrieval pipeline:
        1. Embed the query
        2. Vector search for initial_candidates results
        3. Filter by min_score threshold
        4. (Optional) Rerank with cross-encoder
        5. Return top_k final results
        """
```

**RetrievalResult**:
```python
@dataclass
class RetrievalResult:
    chunk: Chunk
    similarity_score: float
    rerank_score: float | None
    context_before: str | None    # Previous chunk text for context
    context_after: str | None     # Next chunk text for context
```

#### 3.3.3 Reranker (Optional) (`src/retrieval/reranker.py`)

**Model**: `cross-encoder/ms-marco-multilingual-MiniLMv2-L6-v2`

```python
class Reranker:
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Rerank results using cross-encoder for higher precision."""
```

---

### 3.4 Generation / Summarization

#### 3.4.1 Prompt Builder (`src/generation/prompt_builder.py`)

```python
class PromptBuilder:
    def build_answer_prompt(
        self,
        question: str,
        sources: list[RetrievalResult],
        language: str = "he"
    ) -> str:
        """Build the full prompt with system instructions and sources."""
```

**System prompt template**:

```
You are a knowledgeable assistant specializing in Jewish Halacha (religious law).

TASK: Answer the user's question based ONLY on the provided source passages.

RULES:
1. Base your answer EXCLUSIVELY on the provided sources. Do not use external knowledge.
2. For every claim, cite the source using the format: [Book, Section Path].
   Example: [שולחן ערוך, אורח חיים, סימן שכח, סעיף ב]
3. If the sources contain conflicting opinions, present all opinions with their sources.
4. If the sources do not contain sufficient information to answer, clearly state this.
5. Maintain respectful treatment of all sources and opinions.
6. Respond in the same language the user used for their question.

SOURCES:
---
Source 1: {book_title} — {section_path}
{chunk_text}
---
Source 2: {book_title} — {section_path}
{chunk_text}
---
[... additional sources ...]

USER QUESTION:
{question}
```

#### 3.4.2 Summarizer (`src/generation/summarizer.py`)

```python
class Summarizer:
    def __init__(self, provider: str, model: str, api_key: str):
        """
        Providers: "anthropic", "openai", "local"
        Models:
          - anthropic: "claude-sonnet-4-20250514"
          - openai: "gpt-4o"
          - local: "ollama/mistral" (via Ollama)
        """

    async def generate_answer(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.2
    ) -> GeneratedAnswer:
        """Call LLM and return structured answer."""

@dataclass
class GeneratedAnswer:
    text: str
    citations: list[Citation]
    model_used: str
    tokens_used: int
    latency_ms: int
```

#### 3.4.3 Citation Parser (`src/generation/citation_parser.py`)

```python
class CitationParser:
    def extract_citations(self, answer_text: str, sources: list[RetrievalResult]) -> list[Citation]:
        """Parse [Book, Section] citations from the LLM output and validate
        them against the actual sources that were provided."""

@dataclass
class Citation:
    book_title: str
    section_path: str
    source_chunk_id: str       # Links back to the chunk
    is_valid: bool             # Whether it matches a real provided source
```

---

### 3.5 Data Models (`src/models/`)

```python
# src/models/book.py
@dataclass
class Book:
    id: str                    # UUID
    title: str
    author: str
    language: str
    source_path: str
    file_format: str
    chunk_count: int
    ingested_at: datetime
    status: str                # "active", "ingesting", "error"

# src/models/query_result.py
@dataclass
class QueryResult:
    id: str
    question: str
    sources: list[RetrievalResult]
    answer: GeneratedAnswer
    timestamp: datetime
    feedback: str | None       # User feedback (thumbs up/down)
```

---

### 3.6 Storage (`src/storage/`)

#### SQLite Schema

```sql
-- Books registry
CREATE TABLE books (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT,
    language TEXT DEFAULT 'he',
    source_path TEXT NOT NULL,
    file_format TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active'
);

-- Query history
CREATE TABLE query_history (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer_text TEXT,
    sources_json TEXT,          -- JSON serialized source references
    model_used TEXT,
    tokens_used INTEGER,
    latency_ms INTEGER,
    feedback TEXT,              -- 'positive', 'negative', NULL
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Application settings
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

### 3.7 User Interface (`src/ui/`)

**Technology**: Streamlit (for rapid prototyping and clean UI)

#### Pages / Views:

**Main Page** (`app.py`):
- Question input text area (with Hebrew RTL support)
- "Ask" button
- Loading indicator during processing
- Sources panel: expandable cards showing each retrieved passage with book/section metadata and relevance score
- Answer panel: the generated summary with inline citations that link back to the sources panel
- Toggle to show/hide sources

**Book Manager** (`book_manager.py`):
- List of ingested books with stats (chunk count, date added)
- Upload new book button (file picker)
- Remove book button (with confirmation)
- Re-index book button
- Ingestion progress bar

**History Sidebar** (`history_sidebar.py`):
- List of past questions (most recent first)
- Click to reload a past question and its answer
- Search within history

**Settings** (`settings_page.py`):
- LLM provider selection (Anthropic / OpenAI / Local)
- API key input
- Model selection
- Number of sources to retrieve (top_k slider, default 5)
- Language preference
- Embedding model selection

#### RTL / Hebrew CSS:

```css
/* custom.css */
.hebrew-text {
    direction: rtl;
    text-align: right;
    font-family: 'David', 'Frank Ruehl', 'Noto Sans Hebrew', serif;
    line-height: 1.8;
    font-size: 1.1em;
}

.source-card {
    direction: rtl;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    background-color: #fafaf5;
}

.citation-link {
    color: #1a5276;
    cursor: pointer;
    text-decoration: underline;
    font-weight: bold;
}
```

---

## 4. Configuration

### 4.1 config.yaml

```yaml
app:
  name: "Halachic Q&A"
  version: "1.0.0"
  language: "he"             # Default UI language

embedding:
  model: "intfloat/multilingual-e5-large"
  device: "auto"             # "cpu", "cuda", "mps", "auto"
  batch_size: 32

chunking:
  target_tokens: 450
  max_tokens: 800
  min_tokens: 50
  overlap_tokens: 50

retrieval:
  top_k: 5                  # Final number of sources shown
  initial_candidates: 20    # Pre-reranking candidates
  min_similarity: 0.3       # Minimum cosine similarity threshold
  use_reranker: false       # Enable cross-encoder reranking

generation:
  provider: "anthropic"     # "anthropic", "openai", "local"
  model: "claude-sonnet-4-20250514"
  max_tokens: 2000
  temperature: 0.2

storage:
  chroma_dir: "./db/chroma"
  sqlite_path: "./db/app.db"
  books_dir: "./data/books"
  processed_dir: "./data/processed"
```

### 4.2 .env

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...        # Optional
```

---

## 5. Data Flow — End-to-End Query

```
Step 1: User types question
        "האם מותר לחמם מים בשבת?"

Step 2: Query preprocessing
        - Detect language (Hebrew)
        - Normalize text (remove nikud if needed for matching)

Step 3: Embed query
        - Prefix with "query: " for E5 model
        - Generate 1024-dim vector

Step 4: Vector search
        - Search ChromaDB for top 20 candidates
        - Filter by min similarity 0.3

Step 5: (Optional) Rerank
        - Cross-encoder scores query-passage pairs
        - Select top 5

Step 6: Context assembly
        - For each result, optionally fetch neighboring chunks
        - Build source display data

Step 7: Prompt construction
        - System prompt + sources + user question

Step 8: LLM API call
        - Send to Claude API
        - Receive generated answer with citations

Step 9: Citation validation
        - Parse citations from LLM output
        - Validate each against provided sources
        - Mark invalid citations

Step 10: Display
        - Show sources panel with expandable cards
        - Show answer with highlighted citation links
        - Save to query history
```

---

## 6. Error Handling Strategy

| Scenario | Handling |
|----------|----------|
| Book parsing fails | Log error, skip file, report in ingestion report |
| No relevant chunks found | Display "No relevant sources found" message, suggest rephrasing |
| LLM API unavailable | Fall back to displaying sources without summary, or use local model |
| LLM hallucinates citations | Citation parser marks them as invalid, UI shows warning |
| API rate limit hit | Retry with exponential backoff (3 attempts), then show sources only |
| Embedding model fails to load | Show clear error with instructions to check model download |
| Corrupt vector DB | Offer re-indexing option; keep raw books for recovery |

---

## 7. Performance Targets

| Metric | Target |
|--------|--------|
| Ingestion speed | ~100 pages/minute |
| Query embedding | < 200ms |
| Vector search | < 100ms (for up to 100K chunks) |
| Reranking | < 500ms |
| LLM response | 2–8 seconds (API dependent) |
| Total query-to-answer | < 10 seconds |
| Memory usage | < 4GB RAM (with E5-large model loaded) |

---

## 8. Future Cloud Migration Path

| Desktop Component | Cloud Equivalent |
|-------------------|-----------------|
| ChromaDB local | Pinecone / Weaviate / pgvector on RDS |
| SQLite | PostgreSQL on RDS |
| Streamlit local | Streamlit Cloud / FastAPI + React on ECS |
| Local file system | S3 for books + processed data |
| Single user | Auth0/Cognito + multi-tenant data isolation |
| Direct API calls | API Gateway + Lambda/ECS for async processing |
| Local embedding model | SageMaker endpoint or replicate.com |
