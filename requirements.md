# Halachic Q&A Desktop Application — Requirements Document

## 1. Project Summary

**Project Name**: Halachic Q&A  
**Type**: Desktop application (Phase 1), with planned cloud migration (Phase 2)  
**Language**: Python 3.11+  
**Primary Users**: Students, scholars, and anyone seeking Halachic guidance  

The application allows users to query a curated library of Halachic books. It retrieves the most relevant source passages using semantic search (RAG — Retrieval-Augmented Generation), displays them to the user, and generates an AI-powered summary answer grounded exclusively in those sources.

---

## 2. Functional Requirements

### 2.1 Book Ingestion

| ID | Requirement | Priority |
|----|-------------|----------|
| F-ING-01 | System shall accept books in PDF, TXT, DOCX, and HTML formats | Must |
| F-ING-02 | System shall parse Hebrew, Aramaic, and mixed Hebrew/English text correctly | Must |
| F-ING-03 | System shall detect Halachic text structure (Sefer, Siman, Seif, Halacha, Perek) and use it for chunking boundaries | Must |
| F-ING-04 | System shall fall back to paragraph-based chunking when structure is not detected | Must |
| F-ING-05 | System shall generate embeddings for each chunk using a multilingual model | Must |
| F-ING-06 | System shall store chunks with full metadata (book, author, section path, position) in a local vector database | Must |
| F-ING-07 | System shall show a progress bar during book ingestion | Should |
| F-ING-08 | System shall produce an ingestion report (chunks created, warnings, errors) | Should |
| F-ING-09 | System shall allow removing a book and all its indexed data | Must |
| F-ING-10 | System shall allow re-indexing a book (remove old chunks, re-ingest) | Should |
| F-ING-11 | System shall validate file encoding and handle UTF-8/UTF-16/Windows-1255 encoded files | Must |

### 2.2 Question & Retrieval

| ID | Requirement | Priority |
|----|-------------|----------|
| F-QRY-01 | System shall accept free-text questions in Hebrew or English | Must |
| F-QRY-02 | System shall embed the user's question and perform semantic similarity search against all indexed chunks | Must |
| F-QRY-03 | System shall return the top K most relevant source passages (configurable, default 5) | Must |
| F-QRY-04 | System shall display each retrieved source with: book title, author, section path (e.g., "שולחן ערוך, אורח חיים, סימן שכח, סעיף ב"), relevance score, and the full passage text | Must |
| F-QRY-05 | System shall allow the user to filter retrieval by specific book(s) | Should |
| F-QRY-06 | System shall allow the user to expand a source to see neighboring context (previous/next chunks) | Should |
| F-QRY-07 | System shall support optional cross-encoder reranking for improved precision | Could |
| F-QRY-08 | System shall handle the case where no relevant sources are found (similarity below threshold) with a clear message | Must |

### 2.3 Answer Generation

| ID | Requirement | Priority |
|----|-------------|----------|
| F-GEN-01 | System shall send retrieved sources + user question to an LLM to generate a summarized answer | Must |
| F-GEN-02 | Generated answers must include inline citations referencing specific book and section for each claim | Must |
| F-GEN-03 | System shall validate citations against the actual provided sources and flag any that don't match | Must |
| F-GEN-04 | System shall display the answer in the same language the user asked the question in | Must |
| F-GEN-05 | System shall present conflicting Halachic opinions from different sources when they exist | Must |
| F-GEN-06 | System shall clearly state when the available sources are insufficient to answer the question | Must |
| F-GEN-07 | System shall support multiple LLM providers: Anthropic Claude (primary), OpenAI (secondary), local via Ollama (offline fallback) | Should |
| F-GEN-08 | If the LLM API is unavailable, the system shall still display the retrieved sources without a summary | Must |

### 2.4 User Interface

| ID | Requirement | Priority |
|----|-------------|----------|
| F-UI-01 | Application shall run as a local web interface (Streamlit on localhost) | Must |
| F-UI-02 | UI shall support right-to-left (RTL) text display for Hebrew content | Must |
| F-UI-03 | UI shall use Hebrew-appropriate fonts (David, Frank Ruehl, Noto Sans Hebrew, or equivalent) | Must |
| F-UI-04 | UI shall have a main page with: question input, sources panel, and answer panel | Must |
| F-UI-05 | Sources shall be displayed as expandable cards with metadata headers | Must |
| F-UI-06 | Citation links in the answer shall visually connect to their corresponding source cards | Should |
| F-UI-07 | UI shall have a Book Manager page to add, remove, and view ingested books | Must |
| F-UI-08 | UI shall have a Settings page for API keys, model selection, and retrieval parameters | Must |
| F-UI-09 | UI shall have a sidebar showing query history (clickable to reload past Q&A) | Should |
| F-UI-10 | UI shall show a loading spinner/progress during retrieval and generation | Must |
| F-UI-11 | User shall be able to provide thumbs up/down feedback on answers | Could |

### 2.5 History & Persistence

| ID | Requirement | Priority |
|----|-------------|----------|
| F-HIS-01 | System shall save every query, its retrieved sources, and the generated answer to a local SQLite database | Must |
| F-HIS-02 | System shall allow browsing and searching past queries | Should |
| F-HIS-03 | System shall persist all settings between sessions | Must |
| F-HIS-04 | System shall persist the vector database between sessions (no re-indexing on restart) | Must |

---

## 3. Non-Functional Requirements

### 3.1 Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NF-PERF-01 | Book ingestion speed | ≥ 100 pages/minute |
| NF-PERF-02 | Query embedding latency | < 200ms |
| NF-PERF-03 | Vector search latency (up to 100K chunks) | < 100ms |
| NF-PERF-04 | Total query-to-answer time (including LLM) | < 10 seconds |
| NF-PERF-05 | Application memory usage | < 4GB RAM |
| NF-PERF-06 | Application startup time | < 15 seconds |

### 3.2 Reliability

| ID | Requirement |
|----|-------------|
| NF-REL-01 | Application shall not crash on malformed book files; errors are logged and reported |
| NF-REL-02 | Application shall handle LLM API failures gracefully (show sources without summary) |
| NF-REL-03 | Application shall retry failed API calls with exponential backoff (max 3 attempts) |
| NF-REL-04 | Vector database corruption shall be recoverable via re-indexing from raw book files |

### 3.3 Usability

| ID | Requirement |
|----|-------------|
| NF-USE-01 | Application shall be installable by running a single setup command (`pip install -r requirements.txt`) |
| NF-USE-02 | Application shall be launchable with a single command (`python run.py` or `streamlit run src/ui/app.py`) |
| NF-USE-03 | All user-facing text shall support both Hebrew and English |
| NF-USE-04 | Error messages shall be clear and actionable (not raw tracebacks) |

### 3.4 Security

| ID | Requirement |
|----|-------------|
| NF-SEC-01 | API keys shall be stored in `.env` file (gitignored), never hardcoded |
| NF-SEC-02 | Application runs entirely on localhost; no external network access except LLM API calls |
| NF-SEC-03 | `.env` and `db/` directories shall be included in `.gitignore` |

### 3.5 Maintainability

| ID | Requirement |
|----|-------------|
| NF-MNT-01 | Code shall use type hints throughout |
| NF-MNT-02 | All public methods shall have docstrings |
| NF-MNT-03 | Project shall include unit tests for core components (chunker, retriever, citation parser) |
| NF-MNT-04 | Configuration shall be externalized in YAML (not hardcoded) |
| NF-MNT-05 | All components shall use dependency injection (passed via constructor, not imported globals) |

---

## 4. Technology Stack

### 4.1 Core Dependencies

| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **UI** | `streamlit` | ≥ 1.38 | Web-based desktop UI |
| **Vector DB** | `chromadb` | ≥ 0.5 | Local vector storage and search |
| **Embeddings** | `sentence-transformers` | ≥ 3.0 | Local embedding models |
| **LLM Client** | `anthropic` | ≥ 0.39 | Claude API client |
| **LLM Client** | `openai` | ≥ 1.50 | OpenAI API client (optional) |
| **PDF Parsing** | `pymupdf` | ≥ 1.24 | PDF text extraction |
| **DOCX Parsing** | `python-docx` | ≥ 1.1 | Word document parsing |
| **HTML Parsing** | `beautifulsoup4` | ≥ 4.12 | HTML content extraction |
| **Config** | `pyyaml` | ≥ 6.0 | YAML configuration |
| **Env** | `python-dotenv` | ≥ 1.0 | Environment variable management |
| **Data** | `pydantic` | ≥ 2.0 | Data validation and models |

### 4.2 Optional / Enhancement Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Required by sentence-transformers (CPU or CUDA) |
| `chardet` | Automatic file encoding detection |
| `ollama` | Local LLM inference (offline mode) |
| `tqdm` | Progress bars for batch operations |

### 4.3 Development Dependencies

| Package | Purpose |
|---------|---------|
| `pytest` | Testing framework |
| `pytest-asyncio` | Async test support |
| `ruff` | Linter and formatter |
| `mypy` | Static type checking |

---

## 5. requirements.txt

```
# Core
streamlit>=1.38
chromadb>=0.5
sentence-transformers>=3.0
anthropic>=0.39
pymupdf>=1.24
python-docx>=1.1
beautifulsoup4>=4.12
lxml>=5.0
pyyaml>=6.0
python-dotenv>=1.0
pydantic>=2.0

# Optional LLM providers
openai>=1.50

# Utilities
chardet>=5.0
tqdm>=4.66

# ML backend (sentence-transformers dependency)
torch>=2.0

# Dev
pytest>=8.0
pytest-asyncio>=0.24
ruff>=0.7
mypy>=1.11
```

---

## 6. Setup & Running Instructions

### 6.1 First-Time Setup

```bash
# 1. Clone / create the project
cd halachic-qa

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. Place books in data/books/
# Supported: .pdf, .txt, .docx, .html

# 6. Run the application
python run.py
# This will:
#   a) Initialize databases if first run
#   b) Launch Streamlit on http://localhost:8501
```

### 6.2 run.py Entry Point

```python
"""Entry point for the Halachic Q&A application."""
import subprocess
import sys
from pathlib import Path
from src.config import load_config
from src.storage.database import initialize_database

def main():
    config = load_config()
    # Ensure directories exist
    Path(config["storage"]["books_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["storage"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["storage"]["chroma_dir"]).mkdir(parents=True, exist_ok=True)
    # Initialize SQLite if needed
    initialize_database(config["storage"]["sqlite_path"])
    # Launch Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/ui/app.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    main()
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

| Test File | What It Tests |
|-----------|---------------|
| `test_chunker.py` | Hebrew structure detection, chunk size boundaries, overlap logic, edge cases (empty text, single line) |
| `test_retriever.py` | Vector search returns correct results, filtering works, min_score threshold applied |
| `test_citation_parser.py` | Citations extracted correctly, invalid citations flagged, edge cases (no citations, malformed) |
| `test_parser.py` | PDF, DOCX, TXT parsing produces expected output |
| `test_prompt_builder.py` | Prompt includes all sources, respects token limits |

### 7.2 Integration Tests

| Test | What It Tests |
|------|---------------|
| Ingestion end-to-end | Book file → chunks in ChromaDB with correct metadata |
| Query end-to-end | Question → sources retrieved → answer generated (mock LLM) |

### 7.3 Test Fixtures

Place sample Hebrew texts in `tests/fixtures/sample_texts/`:
- `sample_shulchan_aruch.txt` — A few simanim with seifim
- `sample_mishna_berura.txt` — Corresponding commentary
- `sample_unstructured.txt` — Plain Hebrew text without section markers

---

## 8. Implementation Order

Build the application in this order so each step is independently testable:

| Phase | Components | Deliverable |
|-------|-----------|-------------|
| **1** | `config.py`, `models/`, `storage/database.py` | Configuration and data models working |
| **2** | `ingestion/parser.py`, `ingestion/chunker.py` | Can parse books and produce chunks |
| **3** | `embeddings/embedder.py`, `retrieval/vector_store.py` | Can embed and store chunks in ChromaDB |
| **4** | `ingestion/pipeline.py` | Full ingestion pipeline working end-to-end |
| **5** | `retrieval/retriever.py` | Can query and retrieve relevant chunks |
| **6** | `generation/prompt_builder.py`, `generation/summarizer.py`, `generation/citation_parser.py` | Can generate answers with citations |
| **7** | `ui/app.py` + all components | Full UI working |
| **8** | `storage/history.py`, `ui/components/history_sidebar.py` | Query history working |
| **9** | Tests | Unit and integration tests passing |
| **10** | Polish | Error handling, edge cases, UX improvements |

---

## 9. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hebrew embedding quality is poor | Irrelevant sources retrieved | Test multiple models early; `multilingual-e5-large` is the current best. Evaluate with sample queries before building UI. |
| Halachic structure detection fails | Chunks split mid-sentence or mid-halacha | Build comprehensive regex patterns; fall back to paragraph chunking; allow manual section markers in text files. |
| LLM hallucinates Halachic rulings | Users get incorrect guidance | Citation validation is mandatory; system prompt forces source-only answers; disclaimer in UI that this is a research tool, not a psak halacha. |
| PDF parsing loses Hebrew text | Missing or garbled content | Use pymupdf which has strong RTL support; add a verification step that checks for Hebrew characters in output. |
| Large embedding model doesn't fit in RAM | App crashes on low-spec machines | Offer `mpnet-base` as a lighter alternative in config; document minimum specs (8GB RAM recommended). |

---

## 10. Important Disclaimers (Must Display in UI)

The application **must** display a clear disclaimer:

> **הערה חשובה**: יישום זה הוא כלי מחקר בלבד ואינו מהווה תחליף לפסק הלכה מרב מוסמך.
> ניתן להשתמש בו כדי למצוא מקורות ולהבין נושאים הלכתיים, אך יש לפנות לרב מוסמך לקבלת פסק הלכה.
>
> **Important Note**: This application is a research tool only and does not constitute a substitute for a Halachic ruling from a qualified Rabbi. Use it to find sources and understand Halachic topics, but consult a qualified Rabbi for actual Halachic rulings.
