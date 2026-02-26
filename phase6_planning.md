# Phase 6: Generation Module Planning

## Overview

Phase 6 implements the `src/generation/` module — the LLM layer that takes a user question and
retrieved source chunks, builds a grounded prompt, calls the LLM, and returns a validated
`GeneratedAnswer` with parsed citations.

**Inputs** (from Phase 5 Retriever):
- `question: str` — the user's question
- `sources: list[RetrievalResult]` — ranked chunks with metadata and optional context

**Outputs** (consumed by Phase 7 UI):
- `GeneratedAnswer` — LLM response text + validated `Citation` list + token/latency metadata
- `None` — when all LLM providers fail (UI still shows retrieved sources)

---

## Files to Create

```
src/generation/
├── __init__.py          (already exists, currently empty)
├── prompt_builder.py    # Formats sources into system + user prompts
├── summarizer.py        # LLM client: multi-provider, retry, fallback
└── citation_parser.py   # Extracts and validates [N] citations from response

tests/
└── test_summarizer.py   # Unit tests for all three classes
```

---

## Existing Interfaces (Key Contracts)

### `GenerationConfig` (from `src/config.py`)
```python
class GenerationConfig(BaseModel):
    provider: str = "anthropic"          # primary provider
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2000
    temperature: float = 0.2
```

### `GeneratedAnswer` (from `src/models/query_result.py`)
```python
class GeneratedAnswer(BaseModel):
    text: str
    citations: list[Citation] = []
    model_used: str = ""
    tokens_used: int = 0
    latency_ms: int = 0
```

### `Citation` (from `src/models/query_result.py`)
```python
class Citation(BaseModel):
    book_title: str
    section_path: str
    source_chunk_id: str = ""
    is_valid: bool = True
```

### `RetrievalResult` (from `src/models/query_result.py`)
```python
class RetrievalResult(BaseModel):
    chunk: Chunk                    # .book_title, .section_path, .text, .id
    similarity_score: float
    rerank_score: float | None
    context_before: str | None
    context_after: str | None
```

---

## 1. `src/generation/prompt_builder.py`

### Responsibility
Converts `(question, sources)` into `(system_prompt, user_prompt)` strings ready for the LLM.

### Citation Convention
Sources are numbered `[1]`, `[2]`, … `[N]` in the user prompt.
The LLM is instructed to cite inline using `[N]` markers.
`CitationParser` maps these indices back to `RetrievalResult` objects.

### Class Design

```python
class PromptBuilder:
    """Builds LLM prompts grounded in retrieved Halachic sources."""

    def build(
        self,
        question: str,
        sources: list[RetrievalResult],
    ) -> tuple[str, str]:
        """Return (system_prompt, user_prompt)."""

    def _system_prompt(self) -> str:
        """Static system instructions for grounded Halachic Q&A."""

    def _format_sources_block(self, sources: list[RetrievalResult]) -> str:
        """Render numbered source list with metadata and text."""

    def _format_single_source(
        self,
        index: int,
        result: RetrievalResult,
    ) -> str:
        """Format one source entry including optional context."""
```

### System Prompt Content
```
You are a Halachic research assistant. Answer the question below using ONLY the
provided sources. For every claim, cite the source number inline as [N].

Rules:
- Only use information found in the provided sources.
- If the sources do not contain sufficient information, say so explicitly.
- Preserve the original Hebrew/Aramaic text when quoting directly.
- Do NOT invent rulings, authorities, or page numbers.
- This answer is for research purposes only and does not replace a qualified Rabbi's ruling.
```

### User Prompt Structure
```
Sources:

[1] <Book Title> — <Section Path>
<context_before if present (lighter weight)>
<chunk text>
<context_after if present (lighter weight)>

[2] ...

---
Question: <user question>
```

### Edge Cases
- Empty `sources` list → include a note in the user prompt that no sources were found
- `context_before` / `context_after` → rendered in a dimmer format with `(context:)` prefix
- Hebrew text → passed through as-is; Claude handles RTL natively
- Very long chunk texts → no truncation at prompt level; `max_tokens` controls output size

---

## 2. `src/generation/citation_parser.py`

### Responsibility
Extracts `[N]` markers from the LLM response text, maps them to the corresponding
`RetrievalResult`, and returns a deduplicated list of `Citation` objects.

### Class Design

```python
class CitationParser:
    """Parses and validates inline [N] citations from LLM-generated text."""

    # Regex: matches [1], [2], ... [99]
    _CITATION_RE: ClassVar[re.Pattern] = re.compile(r'\[(\d+)\]')

    def parse(
        self,
        answer_text: str,
        sources: list[RetrievalResult],
    ) -> list[Citation]:
        """Return validated Citation objects for all [N] markers found."""

    def _extract_indices(self, text: str) -> list[int]:
        """Return sorted unique 1-based citation indices found in text."""

    def _build_citation(
        self,
        index: int,
        sources: list[RetrievalResult],
    ) -> Citation:
        """Map a 1-based index to a Citation; set is_valid=False if out of range."""
```

### Logic
1. Find all `[N]` markers in `answer_text` via regex
2. Deduplicate, sort
3. For each index `N`:
   - If `1 <= N <= len(sources)` → build `Citation` from `sources[N-1].chunk` with `is_valid=True`
   - Otherwise → build `Citation(book_title="Unknown", section_path="", is_valid=False)`
4. Return deduplicated list (same index cited multiple times → one Citation)

### Citation Validation Rules
- `is_valid=True` if the index maps to a real source
- `is_valid=False` if the LLM hallucinated a citation number beyond the source list
- `source_chunk_id` is populated from `chunk.id` for valid citations

---

## 3. `src/generation/summarizer.py`

### Responsibility
Orchestrates prompt construction → LLM call → citation parsing → returns `GeneratedAnswer`.
Implements multi-provider support with retry and graceful failure.

### Provider Priority
```
Primary:   config.provider  (e.g. "anthropic")
Secondary: next in ["anthropic", "openai", "ollama"] not equal to primary
Tertiary:  "ollama" (always available offline if running)
```

### Retry Strategy
Per provider: up to 3 attempts with exponential backoff (`1s → 2s → 4s`).
On provider exhaustion: log error, return `None`.

### Class Design

```python
class Summarizer:
    """LLM generation service with multi-provider support and retry."""

    def __init__(
        self,
        config: GenerationConfig,
        prompt_builder: PromptBuilder,
        citation_parser: CitationParser,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None: ...

    def generate(
        self,
        question: str,
        sources: list[RetrievalResult],
    ) -> GeneratedAnswer | None:
        """Generate an answer. Returns None if all providers fail."""

    def _provider_order(self) -> list[str]:
        """Return providers in priority order."""

    def _call_with_retry(
        self,
        provider: str,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, str, int] | None:
        """Try one provider up to 3 times. Returns (text, model_id, tokens) or None."""

    def _call_anthropic(
        self, system: str, user: str
    ) -> tuple[str, str, int]:
        """Call Anthropic API. Returns (text, model_id, tokens_used)."""

    def _call_openai(
        self, system: str, user: str
    ) -> tuple[str, str, int]:
        """Call OpenAI API. Returns (text, model_id, tokens_used)."""

    def _call_ollama(
        self, system: str, user: str
    ) -> tuple[str, str, int]:
        """Call local Ollama API via HTTP. Returns (text, model_id, tokens_used)."""
```

### `generate()` Control Flow

```
1. Build (system_prompt, user_prompt) from PromptBuilder
2. Record start_time = time.monotonic()
3. For each provider in _provider_order():
   a. result = _call_with_retry(provider, system_prompt, user_prompt)
   b. If result is not None → break
4. If no result → log error, return None
5. latency_ms = int((time.monotonic() - start_time) * 1000)
6. citations = CitationParser.parse(result.text, sources)
7. Return GeneratedAnswer(
       text=result.text,
       citations=citations,
       model_used=result.model_id,
       tokens_used=result.tokens,
       latency_ms=latency_ms,
   )
```

### `_call_with_retry()` Control Flow

```
for attempt in range(3):
    try:
        return _call_{provider}(system, user)
    except RateLimitError → sleep(2^attempt), continue
    except AuthError      → log, break (no point retrying)
    except Exception      → log, sleep(2^attempt), continue
return None
```

### Provider Implementation Notes

**Anthropic**
- Library: `anthropic` (`pip install anthropic`)
- API: `client.messages.create(...)`
- `tokens_used = response.usage.input_tokens + response.usage.output_tokens`
- Model from `config.model`

**OpenAI**
- Library: `openai` (`pip install openai`)
- API: `client.chat.completions.create(...)`
- `tokens_used = response.usage.total_tokens`
- Model: `"gpt-4o"` (hardcoded fallback, not from config)

**Ollama**
- HTTP POST to `{ollama_base_url}/api/chat`
- Body: `{"model": "llama3.2", "messages": [...], "stream": false}`
- No token count available → set `tokens_used = 0`
- Timeout: 120 seconds

---

## 4. `tests/test_summarizer.py`

### Test Cases

#### PromptBuilder
- `test_build_returns_system_and_user_prompts` — non-empty strings, correct structure
- `test_sources_numbered_correctly` — `[1]`, `[2]` appear in correct positions
- `test_context_included_when_present` — `context_before`/`context_after` appear
- `test_empty_sources_handled` — no IndexError, prompt contains a no-sources note
- `test_hebrew_text_preserved` — Hebrew chars survive round-trip through prompt

#### CitationParser
- `test_valid_citation_extracted` — `[1]` → correct book_title, section_path, is_valid=True
- `test_out_of_range_citation_marked_invalid` — `[99]` → is_valid=False
- `test_duplicate_citations_deduplicated` — `[1] [1]` → one Citation
- `test_no_citations_returns_empty_list`
- `test_source_chunk_id_populated` — Citation.source_chunk_id == chunk.id

#### Summarizer
- `test_generate_returns_generated_answer` — mock Anthropic call, check structure
- `test_generate_returns_none_on_all_provider_failure` — all providers raise Exception
- `test_retry_on_rate_limit` — first attempt raises rate limit, second succeeds
- `test_provider_fallback` — Anthropic fails, OpenAI succeeds
- `test_latency_ms_is_positive` — latency_ms > 0
- `test_empty_sources_returns_answer_without_citations`

### Test Infrastructure
- Use `unittest.mock.patch` to mock `anthropic.Anthropic`, `openai.OpenAI`
- Mock Ollama via `responses` library or `unittest.mock.patch("requests.post")`
- Fixture `make_retrieval_result(text, book_title, section_path)` for DRY source construction

---

## `__init__.py` Exports

```python
# src/generation/__init__.py
from src.generation.citation_parser import CitationParser
from src.generation.prompt_builder import PromptBuilder
from src.generation.summarizer import Summarizer

__all__ = ["CitationParser", "PromptBuilder", "Summarizer"]
```

---

## Config Changes Required

None — `GenerationConfig` already has all needed fields (`provider`, `model`, `max_tokens`,
`temperature`). `config.yaml` already has sensible defaults.

The `AppConfig` already exposes `anthropic_api_key` and `openai_api_key` loaded from `.env`.

---

## Dependencies to Add to `requirements.txt`

```
anthropic>=0.40.0
openai>=1.50.0
requests>=2.31.0      # for Ollama HTTP calls (likely already present)
```

---

## Integration Point (Phase 7 Preview)

The UI will wire these together like:

```python
config = load_config()
prompt_builder = PromptBuilder()
citation_parser = CitationParser()
summarizer = Summarizer(
    config=config.generation,
    prompt_builder=prompt_builder,
    citation_parser=citation_parser,
    anthropic_api_key=config.anthropic_api_key,
    openai_api_key=config.openai_api_key,
)

sources = retriever.search(question)
answer = summarizer.generate(question, sources)

query_result = QueryResult(
    question=question,
    sources=sources,
    answer=answer,   # None = show sources only, no LLM summary
)
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Citation format | `[N]` inline numbers | Simple regex extraction; language-agnostic |
| Multi-provider | Try-in-order with retry | Maximises availability; Ollama as offline safety net |
| Retry policy | 3 attempts, exponential backoff (1s/2s/4s) | Matches CLAUDE.md requirement |
| Return type on failure | `None` | UI still shows sources; avoids crashing |
| Prompt language | English system prompt | Claude handles multilingual content natively |
| Token tracking | Provider-specific extraction | Anthropic/OpenAI report tokens; Ollama does not |
| Context in prompt | Prefixed with `(context:)` label | Clear to LLM which text is primary vs surrounding |
| Dependency injection | All deps via `__init__` | Consistent with project convention; testable |
