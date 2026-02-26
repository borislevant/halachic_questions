"""LLM generation service with multi-provider support, retry, and fallback."""

import logging
import time

from src.config import GenerationConfig
from src.generation.citation_parser import CitationParser
from src.generation.prompt_builder import PromptBuilder
from src.models.query_result import GeneratedAnswer, RetrievalResult

logger = logging.getLogger(__name__)

# Provider resolution order (excluding the primary, which goes first)
_ALL_PROVIDERS = ["anthropic", "openai", "ollama"]

_MAX_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0  # seconds; doubles each attempt (1 → 2 → 4)

# Hardcoded fallback model identifiers for secondary providers
_OPENAI_FALLBACK_MODEL = "gpt-4o"
_OLLAMA_FALLBACK_MODEL = "llama3.2"


class Summarizer:
    """LLM generation service for Halachic Q&A.

    Orchestrates prompt construction → LLM call → citation parsing.
    Supports Anthropic (primary), OpenAI (secondary), and Ollama (offline
    fallback) with per-provider retry and graceful failure.

    Args:
        config: GenerationConfig with provider, model, max_tokens, temperature.
        prompt_builder: PromptBuilder instance.
        citation_parser: CitationParser instance.
        anthropic_api_key: Anthropic API key (or None to skip that provider).
        openai_api_key: OpenAI API key (or None to skip that provider).
        ollama_base_url: Base URL for the local Ollama server.
    """

    def __init__(
        self,
        config: GenerationConfig,
        prompt_builder: PromptBuilder,
        citation_parser: CitationParser,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self._config = config
        self._prompt_builder = prompt_builder
        self._citation_parser = citation_parser
        self._anthropic_api_key = anthropic_api_key
        self._openai_api_key = openai_api_key
        self._ollama_base_url = ollama_base_url.rstrip("/")

    def generate(
        self,
        question: str,
        sources: list[RetrievalResult],
    ) -> GeneratedAnswer | None:
        """Generate a grounded answer for the given question and sources.

        Tries each provider in priority order. If all fail, logs an error
        and returns None so the caller can still display retrieved sources.

        Args:
            question: The user's Halachic question.
            sources: Ranked retrieval results to ground the answer in.

        Returns:
            GeneratedAnswer on success, or None if all providers fail.
        """
        system_prompt, user_prompt = self._prompt_builder.build(question, sources)

        start = time.monotonic()
        result = None

        for provider in self._provider_order():
            result = self._call_with_retry(provider, system_prompt, user_prompt)
            if result is not None:
                break

        if result is None:
            logger.error(
                "All LLM providers failed for question: '%s'", question[:80]
            )
            return None

        text, model_id, tokens_used = result
        latency_ms = int((time.monotonic() - start) * 1000)
        citations = self._citation_parser.parse(text, sources)

        return GeneratedAnswer(
            text=text,
            citations=citations,
            model_used=model_id,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Provider orchestration
    # ------------------------------------------------------------------

    def _provider_order(self) -> list[str]:
        """Return providers in priority order.

        The configured primary provider goes first, followed by the remaining
        providers in the canonical order, skipping unavailable ones.
        """
        primary = self._config.provider
        others = [p for p in _ALL_PROVIDERS if p != primary]
        ordered = [primary] + others
        return [p for p in ordered if self._provider_available(p)]

    def _provider_available(self, provider: str) -> bool:
        """Return True if the provider has a usable credential/endpoint."""
        if provider == "anthropic":
            return bool(self._anthropic_api_key)
        if provider == "openai":
            return bool(self._openai_api_key)
        if provider == "ollama":
            return True  # Always assumed reachable; errors caught at call time
        return False

    def _call_with_retry(
        self,
        provider: str,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, str, int] | None:
        """Try one provider up to _MAX_RETRY_ATTEMPTS times with backoff.

        Args:
            provider: One of "anthropic", "openai", "ollama".
            system_prompt: The system message string.
            user_prompt: The user message string.

        Returns:
            (text, model_id, tokens_used) on success, or None on exhaustion.
        """
        call_fn = {
            "anthropic": self._call_anthropic,
            "openai": self._call_openai,
            "ollama": self._call_ollama,
        }[provider]

        for attempt in range(_MAX_RETRY_ATTEMPTS):
            try:
                return call_fn(system_prompt, user_prompt)

            except _AuthError:
                # Credential error — no point retrying this provider
                logger.error(
                    "Authentication failed for provider '%s', skipping", provider
                )
                return None

            except _RateLimitError:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Rate limit hit on '%s' (attempt %d/%d), retrying in %.1fs",
                    provider,
                    attempt + 1,
                    _MAX_RETRY_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)

            except Exception:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Error calling provider '%s' (attempt %d/%d), retrying in %.1fs",
                    provider,
                    attempt + 1,
                    _MAX_RETRY_ATTEMPTS,
                    delay,
                    exc_info=True,
                )
                time.sleep(delay)

        logger.error(
            "Provider '%s' exhausted after %d attempts", provider, _MAX_RETRY_ATTEMPTS
        )
        return None

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _call_anthropic(
        self,
        system: str,
        user: str,
    ) -> tuple[str, str, int]:
        """Call the Anthropic Messages API.

        Args:
            system: System prompt text.
            user: User message text.

        Returns:
            (response_text, model_id, total_tokens_used)

        Raises:
            _AuthError: On authentication failure.
            _RateLimitError: On rate limit (HTTP 429).
        """
        import anthropic

        try:
            client = anthropic.Anthropic(api_key=self._anthropic_api_key)
            response = client.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.AuthenticationError as exc:
            raise _AuthError("Anthropic authentication failed") from exc
        except anthropic.RateLimitError as exc:
            raise _RateLimitError("Anthropic rate limit exceeded") from exc

        text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, response.model, tokens

    def _call_openai(
        self,
        system: str,
        user: str,
    ) -> tuple[str, str, int]:
        """Call the OpenAI Chat Completions API.

        Args:
            system: System prompt text.
            user: User message text.

        Returns:
            (response_text, model_id, total_tokens_used)

        Raises:
            _AuthError: On authentication failure.
            _RateLimitError: On rate limit.
        """
        import openai

        try:
            client = openai.OpenAI(api_key=self._openai_api_key)
            response = client.chat.completions.create(
                model=_OPENAI_FALLBACK_MODEL,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except openai.AuthenticationError as exc:
            raise _AuthError("OpenAI authentication failed") from exc
        except openai.RateLimitError as exc:
            raise _RateLimitError("OpenAI rate limit exceeded") from exc

        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        model_id = response.model or _OPENAI_FALLBACK_MODEL
        return text, model_id, tokens

    def _call_ollama(
        self,
        system: str,
        user: str,
    ) -> tuple[str, str, int]:
        """Call a local Ollama server via its HTTP API.

        Args:
            system: System prompt text.
            user: User message text.

        Returns:
            (response_text, model_id, 0) — Ollama does not report token counts.

        Raises:
            Exception: On any HTTP or connection error.
        """
        import requests

        url = f"{self._ollama_base_url}/api/chat"
        payload = {
            "model": _OLLAMA_FALLBACK_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        text = data["message"]["content"]
        return text, _OLLAMA_FALLBACK_MODEL, 0


# ------------------------------------------------------------------
# Internal sentinel exceptions (not exposed publicly)
# ------------------------------------------------------------------


class _AuthError(Exception):
    """Raised when a provider returns an authentication error."""


class _RateLimitError(Exception):
    """Raised when a provider enforces a rate limit."""
