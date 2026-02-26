"""Tests for the generation module: PromptBuilder, CitationParser, Summarizer."""

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.config import GenerationConfig
from src.generation.citation_parser import CitationParser
from src.generation.prompt_builder import PromptBuilder
from src.generation.summarizer import Summarizer
from src.models.chunk import Chunk
from src.models.query_result import RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_chunk(
    book_title: str = "Shulchan Aruch",
    section_path: str = "Orach Chaim 1:1",
    text: str = "Sample halachic text.",
    chunk_id: str = "chunk-1",
) -> Chunk:
    """Build a minimal Chunk for testing."""
    return Chunk(
        id=chunk_id,
        text=text,
        book_id="book-1",
        book_title=book_title,
        section_path=section_path,
    )


def make_result(
    book_title: str = "Shulchan Aruch",
    section_path: str = "Orach Chaim 1:1",
    text: str = "Sample halachic text.",
    chunk_id: str = "chunk-1",
    context_before: str | None = None,
    context_after: str | None = None,
) -> RetrievalResult:
    """Build a minimal RetrievalResult for testing."""
    return RetrievalResult(
        chunk=make_chunk(book_title, section_path, text, chunk_id),
        similarity_score=0.9,
        context_before=context_before,
        context_after=context_after,
    )


@pytest.fixture
def prompt_builder() -> PromptBuilder:
    return PromptBuilder()


@pytest.fixture
def citation_parser() -> CitationParser:
    return CitationParser()


@pytest.fixture
def generation_config() -> GenerationConfig:
    return GenerationConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        max_tokens=2000,
        temperature=0.2,
    )


@pytest.fixture
def summarizer(generation_config: GenerationConfig) -> Summarizer:
    return Summarizer(
        config=generation_config,
        prompt_builder=PromptBuilder(),
        citation_parser=CitationParser(),
        anthropic_api_key="test-key",
        openai_api_key="test-openai-key",
    )


# ---------------------------------------------------------------------------
# PromptBuilder tests
# ---------------------------------------------------------------------------


class TestPromptBuilder:
    def test_build_returns_two_nonempty_strings(self, prompt_builder: PromptBuilder) -> None:
        sources = [make_result()]
        system, user = prompt_builder.build("What is the law?", sources)
        assert isinstance(system, str) and len(system) > 0
        assert isinstance(user, str) and len(user) > 0

    def test_system_prompt_mentions_citation_format(
        self, prompt_builder: PromptBuilder
    ) -> None:
        system, _ = prompt_builder.build("q", [make_result()])
        assert "[N]" in system

    def test_system_prompt_contains_disclaimer(self, prompt_builder: PromptBuilder) -> None:
        system, _ = prompt_builder.build("q", [make_result()])
        assert "Rabbi" in system

    def test_sources_numbered_from_one(self, prompt_builder: PromptBuilder) -> None:
        sources = [make_result(chunk_id="a"), make_result(chunk_id="b")]
        _, user = prompt_builder.build("q", sources)
        assert "[1]" in user
        assert "[2]" in user

    def test_source_order_is_preserved(self, prompt_builder: PromptBuilder) -> None:
        s1 = make_result(book_title="Book A", chunk_id="a")
        s2 = make_result(book_title="Book B", chunk_id="b")
        _, user = prompt_builder.build("q", [s1, s2])
        pos_1 = user.index("[1]")
        pos_2 = user.index("[2]")
        assert pos_1 < pos_2

    def test_book_title_appears_in_user_prompt(self, prompt_builder: PromptBuilder) -> None:
        _, user = prompt_builder.build("q", [make_result(book_title="Mishnah Berurah")])
        assert "Mishnah Berurah" in user

    def test_section_path_appears_in_user_prompt(self, prompt_builder: PromptBuilder) -> None:
        _, user = prompt_builder.build("q", [make_result(section_path="סימן א")])
        assert "סימן א" in user

    def test_chunk_text_appears_in_user_prompt(self, prompt_builder: PromptBuilder) -> None:
        _, user = prompt_builder.build("q", [make_result(text="הלכה חשובה")])
        assert "הלכה חשובה" in user

    def test_context_before_included_when_present(self, prompt_builder: PromptBuilder) -> None:
        result = make_result(context_before="Preceding sentence.")
        _, user = prompt_builder.build("q", [result])
        assert "Preceding sentence." in user

    def test_context_after_included_when_present(self, prompt_builder: PromptBuilder) -> None:
        result = make_result(context_after="Following sentence.")
        _, user = prompt_builder.build("q", [result])
        assert "Following sentence." in user

    def test_question_appears_in_user_prompt(self, prompt_builder: PromptBuilder) -> None:
        _, user = prompt_builder.build("Is carrying permitted on Shabbat?", [make_result()])
        assert "Is carrying permitted on Shabbat?" in user

    def test_empty_sources_produces_no_source_note(self, prompt_builder: PromptBuilder) -> None:
        _, user = prompt_builder.build("q", [])
        assert "No relevant sources" in user
        assert "[1]" not in user

    def test_hebrew_text_preserved_in_prompt(self, prompt_builder: PromptBuilder) -> None:
        hebrew = "מותר לטלטל בשבת"
        _, user = prompt_builder.build(hebrew, [make_result(text="אסור")])
        assert hebrew in user
        assert "אסור" in user


# ---------------------------------------------------------------------------
# CitationParser tests
# ---------------------------------------------------------------------------


class TestCitationParser:
    def test_valid_citation_extracted(self, citation_parser: CitationParser) -> None:
        sources = [make_result(book_title="Shulchan Aruch", section_path="OC 1:1", chunk_id="c1")]
        citations = citation_parser.parse("The law is X [1].", sources)
        assert len(citations) == 1
        assert citations[0].book_title == "Shulchan Aruch"
        assert citations[0].section_path == "OC 1:1"
        assert citations[0].source_chunk_id == "c1"
        assert citations[0].is_valid is True

    def test_multiple_citations_extracted(self, citation_parser: CitationParser) -> None:
        sources = [make_result(chunk_id="a"), make_result(chunk_id="b")]
        citations = citation_parser.parse("X [1] and Y [2].", sources)
        assert len(citations) == 2
        assert {c.source_chunk_id for c in citations} == {"a", "b"}

    def test_duplicate_citations_deduplicated(self, citation_parser: CitationParser) -> None:
        sources = [make_result(chunk_id="a")]
        citations = citation_parser.parse("X [1]. Also [1].", sources)
        assert len(citations) == 1

    def test_out_of_range_citation_marked_invalid(self, citation_parser: CitationParser) -> None:
        sources = [make_result(chunk_id="a")]
        citations = citation_parser.parse("X [99].", sources)
        assert len(citations) == 1
        assert citations[0].is_valid is False
        assert citations[0].book_title == "Unknown"
        assert citations[0].source_chunk_id == ""

    def test_no_citations_returns_empty_list(self, citation_parser: CitationParser) -> None:
        sources = [make_result()]
        citations = citation_parser.parse("No markers here.", sources)
        assert citations == []

    def test_source_chunk_id_populated_for_valid(self, citation_parser: CitationParser) -> None:
        sources = [make_result(chunk_id="my-chunk-id")]
        citations = citation_parser.parse("[1]", sources)
        assert citations[0].source_chunk_id == "my-chunk-id"

    def test_zero_index_marked_invalid(self, citation_parser: CitationParser) -> None:
        sources = [make_result()]
        citations = citation_parser.parse("[0]", sources)
        assert citations[0].is_valid is False

    def test_citations_order_matches_first_appearance(
        self, citation_parser: CitationParser
    ) -> None:
        s1 = make_result(chunk_id="a")
        s2 = make_result(chunk_id="b")
        sources = [s1, s2]
        citations = citation_parser.parse("[2] then [1].", sources)
        assert citations[0].source_chunk_id == "b"
        assert citations[1].source_chunk_id == "a"

    def test_empty_sources_all_citations_invalid(self, citation_parser: CitationParser) -> None:
        citations = citation_parser.parse("[1] [2]", [])
        assert all(not c.is_valid for c in citations)

    def test_mixed_valid_and_invalid_citations(self, citation_parser: CitationParser) -> None:
        sources = [make_result(chunk_id="a")]
        citations = citation_parser.parse("[1] is valid, [5] is not.", sources)
        valid = [c for c in citations if c.is_valid]
        invalid = [c for c in citations if not c.is_valid]
        assert len(valid) == 1
        assert len(invalid) == 1


# ---------------------------------------------------------------------------
# Summarizer tests
# ---------------------------------------------------------------------------


def _make_anthropic_response(text: str, model: str = "claude-sonnet-4-6") -> MagicMock:
    """Build a mock Anthropic messages response."""
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    response.model = model
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    return response


def _make_openai_response(text: str, model: str = "gpt-4o") -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.model = model
    response.usage.total_tokens = 200
    return response


class TestSummarizer:
    def test_generate_returns_generated_answer(self, summarizer: Summarizer) -> None:
        sources = [make_result()]
        mock_response = _make_anthropic_response("The answer is X [1].")

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            answer = summarizer.generate("What is the law?", sources)

        assert answer is not None
        assert "The answer is X" in answer.text
        assert answer.model_used == "claude-sonnet-4-6"
        assert answer.tokens_used == 150
        assert answer.latency_ms >= 0

    def test_generate_parses_citations(self, summarizer: Summarizer) -> None:
        sources = [make_result(chunk_id="c1"), make_result(chunk_id="c2")]
        mock_response = _make_anthropic_response("X [1] and Y [2].")

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            answer = summarizer.generate("q", sources)

        assert answer is not None
        assert len(answer.citations) == 2
        assert {c.source_chunk_id for c in answer.citations} == {"c1", "c2"}

    def test_generate_returns_none_when_all_providers_fail(self) -> None:
        config = GenerationConfig(provider="anthropic")
        summarizer = Summarizer(
            config=config,
            prompt_builder=PromptBuilder(),
            citation_parser=CitationParser(),
            anthropic_api_key="bad-key",
        )

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.side_effect = Exception("network error")
            with patch("time.sleep"):  # suppress delays in tests
                answer = summarizer.generate("q", [make_result()])

        assert answer is None

    def test_retry_on_transient_error(self, summarizer: Summarizer) -> None:
        """First two calls fail, third succeeds."""
        sources = [make_result()]
        mock_response = _make_anthropic_response("Answer [1].")

        call_count = 0

        def side_effect(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("transient")
            return mock_response

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.side_effect = side_effect
            with patch("time.sleep"):
                answer = summarizer.generate("q", sources)

        assert answer is not None
        assert call_count == 3

    def test_auth_error_skips_remaining_retries(self) -> None:
        """AuthenticationError should not be retried — move to next provider."""
        import anthropic

        config = GenerationConfig(provider="anthropic")
        summarizer = Summarizer(
            config=config,
            prompt_builder=PromptBuilder(),
            citation_parser=CitationParser(),
            anthropic_api_key="bad",
            openai_api_key=None,  # no fallback available
        )

        call_count = 0

        def side_effect(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            raise anthropic.AuthenticationError(
                message="auth",
                response=MagicMock(status_code=401, headers={}),
                body={},
            )

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.side_effect = side_effect
            answer = summarizer.generate("q", [make_result()])

        assert answer is None
        assert call_count == 1  # No retries on auth error

    def test_provider_fallback_to_openai(self) -> None:
        """Anthropic fails entirely, OpenAI succeeds."""
        import anthropic

        config = GenerationConfig(provider="anthropic")
        summarizer = Summarizer(
            config=config,
            prompt_builder=PromptBuilder(),
            citation_parser=CitationParser(),
            anthropic_api_key="bad",
            openai_api_key="valid-openai-key",
        )

        mock_openai_response = _make_openai_response("OpenAI answer [1].")

        def anthropic_side_effect(**kwargs: Any) -> MagicMock:
            raise anthropic.AuthenticationError(
                message="auth",
                response=MagicMock(status_code=401, headers={}),
                body={},
            )

        with (
            patch("anthropic.Anthropic") as MockAnthropic,
            patch("openai.OpenAI") as MockOpenAI,
        ):
            MockAnthropic.return_value.messages.create.side_effect = anthropic_side_effect
            MockOpenAI.return_value.chat.completions.create.return_value = mock_openai_response

            answer = summarizer.generate("q", [make_result()])

        assert answer is not None
        assert "OpenAI answer" in answer.text
        assert answer.model_used == "gpt-4o"

    def test_latency_ms_is_non_negative(self, summarizer: Summarizer) -> None:
        mock_response = _make_anthropic_response("Answer.")

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            answer = summarizer.generate("q", [make_result()])

        assert answer is not None
        assert answer.latency_ms >= 0

    def test_empty_sources_returns_answer(self, summarizer: Summarizer) -> None:
        mock_response = _make_anthropic_response("No sources found.")

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            answer = summarizer.generate("q", [])

        assert answer is not None
        assert answer.citations == []

    def test_ollama_fallback_used_when_only_provider(self) -> None:
        config = GenerationConfig(provider="ollama")
        summarizer = Summarizer(
            config=config,
            prompt_builder=PromptBuilder(),
            citation_parser=CitationParser(),
            anthropic_api_key=None,
            openai_api_key=None,
            ollama_base_url="http://localhost:11434",
        )

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {
            "message": {"content": "Ollama answer [1]."}
        }
        mock_http_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_http_response):
            answer = summarizer.generate("q", [make_result(chunk_id="c1")])

        assert answer is not None
        assert "Ollama answer" in answer.text
        assert answer.tokens_used == 0  # Ollama doesn't report tokens
        assert answer.model_used == "llama3.2"
