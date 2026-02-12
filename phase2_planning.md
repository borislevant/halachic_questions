# Phase 2: Book Parsing and Chunking

## Context

Phase 1 is complete — config loader, Pydantic data models (Book, Chunk, QueryResult), SQLite initialization, and the entry point are all working with 28 passing tests. Phase 2 builds the ingestion layer: parsing book files (PDF, TXT, DOCX, HTML) into structured text, then splitting that text into semantically meaningful chunks that respect Halachic text structure.

The chunker consumes `ChunkingConfig` from `src/config.py` (target=450, max=800, min=50, overlap=50 tokens) and produces `Chunk` objects from `src/models/chunk.py`.

### Real Book Analysis: מנחת איש מסכת ברכות

An 880-page Hebrew PDF (`data/מנחת איש מסכת ברכות.pdf`) is available for development. Analysis reveals:
- **Main structure**: `פרק א`, `פרק ב`, ... `פרק כח` (chapters)
- **Halacha numbering**: `א.`, `ב.`, `.ג`, `.ד` (Hebrew letter + period, sometimes reversed) — these are the individual laws within each chapter
- **Sub-topic headers**: Plain Hebrew text lines like `נוסח הברכה והבנתה`, `אמירת הברכה וכבודה` — topical groupings within a chapter
- **Footnotes**: Numbered references at bottom of pages (`.1`, `.2`, etc.)
- **Page headers**: Each page has `מנחת איש` + page number + chapter reference
- **No סימן/סעיף structure** — this book uses פרק + numbered halachot, not the Shulchan Aruch format

This means the chunker must handle **both** classic Shulchan Aruch patterns (סימן/סעיף) **and** the פרק + numbered-halacha pattern used in this book.

## Files to Create/Modify

### Step 1: Intermediate data models — `src/models/parsed.py` (new)

- `Section` model: section_type, title, text, char_start, char_end, subsections (recursive), level
- `ParsedBook` model: title, author, language, raw_text, sections, source_path, file_format
- Both use Pydantic `BaseModel` to match existing conventions
- Update `src/models/__init__.py` to re-export `ParsedBook` and `Section`

### Step 2: Test fixtures — `tests/fixtures/sample_texts/`

Create 3 synthetic Hebrew text fixtures + use the real PDF:
- `sample_shulchan_aruch.txt` — 2 simanim with 2–3 seifim each (סימן/סעיף markers)
- `sample_mishna_berura.txt` — commentary with ס"ק markers
- `sample_unstructured.txt` — plain Hebrew paragraphs, no section markers (~800 words)
- Real PDF at `data/מנחת איש מסכת ברכות.pdf` — integration test for PDF parsing + פרק/halacha chunking

### Step 3: Book parser — `src/ingestion/parser.py` (new)

`BookParser` class with single `parse(file_path) -> ParsedBook` method dispatching to:
- `_parse_pdf` — `pymupdf` (fitz), `page.get_text("text")` per page
- `_parse_txt` — built-in open with UTF-8, fallback to `chardet` for Windows-1255/UTF-16
- `_parse_docx` — `python-docx`, join paragraphs with `\n\n`
- `_parse_html` — `BeautifulSoup` with lxml, strip scripts/styles
- `_detect_language` — count Hebrew chars (U+0590–U+05FF) vs Latin; >60% = "he", >60% Latin = "en", else "mixed"
- `_extract_title_from_text` — first non-empty line, fallback to filename

Note: Parser returns `ParsedBook` with `sections=[]`. Section detection is the chunker's job.

### Step 4: Chunker — `src/ingestion/chunker.py` (new)

`HalachicChunker(config: ChunkingConfig)` with `chunk(parsed_book, book_id=None) -> list[Chunk]`:

**Chunking strategy (priority order):**
1. **Structural** — detect פרק/סימן/סעיף/הלכה/ס"ק + numbered halachot via regex, build nested section tree, chunk recursively
2. **Paragraph** — split on `\n\s*\n`, merge small paragraphs, split large ones
3. **Sliding window** — token-based with overlap for unstructured blocks

**Key methods:**
- `_detect_sections(text)` → find all structure markers, sort by position
- `_build_section_tree(markers, text)` → stack-based nesting by hierarchy level
- `_chunk_sections(sections, ...)` → recursive chunking with section_path building ("פרק א > א")
- `_chunk_by_paragraphs(text, ...)` → double-newline splitting
- `_sliding_window_chunks(text, ...)` → target_tokens window with overlap_tokens overlap
- `_merge_small_chunks(chunks)` → merge consecutive chunks below min_tokens
- `estimate_tokens(text)` → standalone function, `len(text.split())`

**Hebrew structure patterns (updated for real book):**
```python
STRUCTURE_PATTERNS = {
    "perek": r"^פרק\s+[א-ת]{1,4}\s*$",                    # פרק א (on its own line)
    "siman": r"^סימן\s+[א-ת]{1,4}",                        # סימן שכח
    "seif": r"^סעיף\s+[א-ת]{1,4}",                         # סעיף ב
    "halacha": r"^\.?([א-ת]{1,3})\.\s*$",                   # .א or א. (numbered halacha)
    "siman_katan": r"ס[\"״׳׳]ק\s+[א-ת]{1,4}",            # ס"ק א
}

HIERARCHY_LEVELS = {
    "perek": 0,       # Chapter (highest)
    "siman": 1,       # Section
    "halacha": 2,     # Individual law
    "seif": 3,        # Sub-section
    "siman_katan": 4, # Commentary sub-section
}
```

### Step 5: Update `src/ingestion/__init__.py`

Export `BookParser`, `HalachicChunker`, `estimate_tokens`.

### Step 6: Tests

**`tests/test_parser.py`** (~15 tests):
- TXT parsing (UTF-8, Windows-1255, UTF-16, empty file)
- PDF parsing (mocked fitz for unit tests)
- DOCX parsing (mocked python-docx)
- HTML parsing (real files, strip tags/scripts)
- Language detection (Hebrew, English, mixed)
- Title extraction, format detection, error cases
- **Integration test**: parse the real `מנחת איש` PDF — verify it extracts Hebrew text, detects language as "he", has substantial raw_text

**`tests/test_chunker.py`** (~25 tests):
- Structural: siman/seif/perek/siman_katan/numbered-halacha detection, section_path hierarchy, max_tokens enforcement, overlap, small section merging
- Paragraph: double-newline splitting, merge small, split large
- Sliding window: window size, overlap between windows
- Edge cases: empty text, whitespace-only, single short section
- Metadata: chunk_index/total set, book_id/title propagated, token_count populated, UUIDs unique
- Fixture tests: end-to-end with all 3 sample txt files
- **Integration test**: parse the real PDF → chunk it → verify chunks have proper פרק-based section_paths, reasonable token counts, and no chunk exceeds max_tokens

## Verification

```bash
# Install parsing dependencies
pip install pymupdf python-docx beautifulsoup4 lxml chardet

# Run Phase 2 tests
pytest tests/test_parser.py tests/test_chunker.py -v

# Run full test suite (Phase 1 + Phase 2)
pytest tests/ -v

# Integration smoke test with real PDF
python -c "
from src.ingestion.parser import BookParser
from src.ingestion.chunker import HalachicChunker
from src.config import ChunkingConfig

parser = BookParser()
book = parser.parse('data/מנחת איש מסכת ברכות.pdf')
print(f'Parsed: {book.title}, {book.language}, {len(book.raw_text)} chars')

chunker = HalachicChunker(ChunkingConfig())
chunks = chunker.chunk(book)
print(f'Chunks: {len(chunks)}')
print(f'Sample path: {chunks[0].section_path}')
print(f'Max tokens: {max(c.token_count for c in chunks)}')
"

# Type checking
mypy src/models/parsed.py src/ingestion/parser.py src/ingestion/chunker.py

# Lint
ruff check src/ tests/
```
