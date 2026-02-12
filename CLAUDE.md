# CLAUDE.md — Halachic Q&A Project

## Project Overview

Desktop application (Python 3.11+) for querying a curated library of Halachic (Jewish law) books. Uses RAG (Retrieval-Augmented Generation) to retrieve relevant source passages via semantic search and generate AI-powered answers grounded exclusively in those sources.

**Status**: Pre-development — only `requirements.md` and `architecture.md` exist. No code written yet.

## Tech Stack

- **Language**: Python 3.11+
- **UI**: Streamlit (localhost web interface)
- **Vector DB**: ChromaDB (local persistent storage)
- **Embeddings**: sentence-transformers (`intfloat/multilingual-e5-large`)
- **LLM**: Anthropic Claude (primary), OpenAI (secondary), Ollama (offline fallback)
- **Database**: SQLite (app state, query history, settings)
- **Config**: YAML (`config.yaml`) + `.env` for secrets
- **Data validation**: Pydantic v2

## Project Structure

```
halachic-qa/
├── run.py                          # Entry point
├── config.yaml                     # App configuration
├── .env                            # API keys (gitignored)
├── data/
│   ├── books/                      # Raw book files (PDF, TXT, DOCX, HTML)
│   └── processed/                  # Chunked + metadata JSON files
├── db/
│   ├── chroma/                     # ChromaDB persistent storage
│   └── app.db                      # SQLite
├── src/
│   ├── config.py                   # Config loader (YAML + env)
│   ├── ingestion/                  # Book parsing, chunking, ingestion pipeline
│   ├── embeddings/                 # Embedding model wrapper
│   ├── retrieval/                  # Vector store, retriever, reranker
│   ├── generation/                 # Prompt builder, summarizer, citation parser
│   ├── models/                     # Pydantic/dataclass models
│   ├── storage/                    # SQLite operations, history management
│   └── ui/                         # Streamlit app + components + CSS
└── tests/
    ├── fixtures/sample_texts/      # Test Hebrew text samples
    ├── test_chunker.py
    ├── test_retriever.py
    ├── test_summarizer.py
    └── ...
```

## Key Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run
python run.py
# or: streamlit run src/ui/app.py

# Tests
pytest
pytest tests/test_chunker.py -v

# Linting & formatting
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/
```

## Development Guidelines

- **Type hints** on all code
- **Docstrings** on all public methods
- **Dependency injection** — pass dependencies via constructors, no imported globals
- **Config externalized** in YAML, never hardcoded
- **API keys** in `.env` only, never committed
- `.env` and `db/` directories must be in `.gitignore`

## Hebrew / RTL Considerations

- The app handles Hebrew, Aramaic, and mixed Hebrew/English text
- UI must support RTL text display
- Use Hebrew-appropriate fonts (David, Frank Ruehl, Noto Sans Hebrew)
- Embedding model uses E5 prefix convention: `"query: "` for queries, `"passage: "` for documents
- Structure-aware chunking detects Halachic text patterns: סימן, סעיף, הלכה, פרק, ס"ק

## Implementation Order

Build in this order (each phase independently testable):

1. `config.py`, `models/`, `storage/database.py` — config and data models
2. `ingestion/parser.py`, `ingestion/chunker.py` — book parsing and chunking
3. `embeddings/embedder.py`, `retrieval/vector_store.py` — embed and store chunks
4. `ingestion/pipeline.py` — full ingestion pipeline
5. `retrieval/retriever.py` — query and retrieve chunks
6. `generation/` — prompt builder, summarizer, citation parser
7. `ui/app.py` + components — full UI
8. `storage/history.py` + history sidebar — query history
9. Tests — unit and integration
10. Polish — error handling, edge cases, UX

## Important Notes

- This is a **research tool**, not a substitute for a qualified Rabbi's ruling. The UI must display this disclaimer.
- Citation validation is mandatory — every LLM claim must be traceable to provided sources.
- When LLM API is unavailable, still display retrieved sources without a summary.
- Retry failed API calls with exponential backoff (max 3 attempts).
