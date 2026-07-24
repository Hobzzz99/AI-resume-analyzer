"""Google Gemini generation adapter.

A second implementation of the :class:`~app.rag.base.LLMClient` protocol, added so
the engine can run on Gemini's free tier — which is far more generous per minute
than Groq's, effectively removing the rate-limit friction for local single-user
use.

It has the same two jobs as :class:`~app.llm.groq_client.GroqClient`: translate a
prompt into text, and normalise provider failures into this application's error
taxonomy (429 → ``LLMRateLimitError``, timeout → ``LLMTimeoutError``, else
``LLMError``). Because both satisfy ``LLMClient``, the rest of the system — the
structured generator, the repair loop, the pipeline — is identical regardless of
which provider is configured. That is the whole point of the protocol.

The model id is configuration, never hardcoded (FR-021, Principle VII), and is
reported by ``/health``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.utils.exceptions import ConfigurationError, LLMError, LLMRateLimitError, LLMTimeoutError
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Matched against the provider's error text. Gemini surfaces quota exhaustion as
# RESOURCE_EXHAUSTED / 429, and transient overload as UNAVAILABLE / 503; string
# matching is used for the same reason as in the Groq adapter — the concrete
# exception types shift across SDK versions more often than these markers do.
_RATE_LIMIT_MARKERS = ("resource_exhausted", "429", "quota", "rate limit", "too many requests")
_TIMEOUT_MARKERS = ("deadline", "timeout", "timed out", "504")


class GeminiClient:
    """Chat-completion client for Google Gemini via the ``google-genai`` SDK.

    Args:
        api_key: Google AI Studio API key.
        model: Model id, from configuration (e.g. ``gemini-2.0-flash``).
        temperature: Sampling temperature. Low by default — this is extraction.
        max_tokens: Response ceiling (``max_output_tokens``).
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> None:
        if not api_key.strip():
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. Get one free at https://aistudio.google.com/apikey "
                "and add it to your .env file.",
                details={"setting": "GEMINI_API_KEY"},
            )
        if not model.strip():
            raise ConfigurationError(
                "GEMINI_MODEL is not set.", details={"setting": "GEMINI_MODEL"}
            )

        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client: Any = None

    @property
    def model_name(self) -> str:
        """Resolved model identifier."""
        return self._model

    def _get_client(self) -> Any:
        """Construct the SDK client on first use.

        Lazy for the same reason the other providers are: the composition root
        must be able to build this object without any network or credential work
        happening at import time.
        """
        if self._client is None:
            from google import genai  # noqa: PLC0415

            # http_options carries the per-request timeout (milliseconds).
            self._client = genai.Client(
                api_key=self._api_key,
                http_options={"timeout": int(self._timeout * 1000)},
            )
        return self._client

    def _config(self, *, system: str | None, json_mode: bool) -> Any:
        """Build the generation config for one request."""
        from google.genai import types  # noqa: PLC0415

        return types.GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
            system_instruction=system or None,
            # Native JSON mode — Gemini constrains decoding to valid JSON, the
            # same guarantee JSON mode gives on Groq, which keeps the parser's
            # job to semantic validation rather than syntax recovery.
            response_mime_type="application/json" if json_mode else "text/plain",
        )

    def _translate(self, exc: Exception) -> LLMError:
        """Map an SDK exception onto this application's error taxonomy."""
        text = str(exc).lower()
        if any(marker in text for marker in _RATE_LIMIT_MARKERS):
            return LLMRateLimitError(
                "The generation provider rate-limited this request. Wait a moment and retry.",
                details={"model": self._model, "provider_error": str(exc)[:500]},
            )
        if any(marker in text for marker in _TIMEOUT_MARKERS):
            return LLMTimeoutError(
                f"Generation timed out after {self._timeout:.0f}s.",
                details={"model": self._model, "timeout_seconds": self._timeout},
            )
        return LLMError(
            f"The generation provider failed: {exc}",
            details={"model": self._model, "provider_error": str(exc)[:500]},
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull plain text out of a Gemini response.

        ``response.text`` is the happy path, but it is ``None`` when the model
        returns no candidate (e.g. a safety block). Falling back through the
        candidate parts — and returning ``""`` rather than raising — lets the
        structured parser report a clean "empty response" that the repair loop
        can act on, instead of an ``AttributeError`` escaping the provider layer.
        """
        text = getattr(response, "text", None)
        if text:
            return str(text)

        parts: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "text", None):
                    parts.append(str(part.text))
        return "".join(parts)

    def generate(self, prompt: str, *, system: str | None = None, json_mode: bool = False) -> str:
        """Produce a completion.

        Raises:
            LLMRateLimitError: Provider quota exhausted.
            LLMTimeoutError: Request exceeded the timeout.
            LLMError: Any other provider failure.
        """
        client = self._get_client()
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._config(system=system, json_mode=json_mode),
            )
        except Exception as exc:
            error = self._translate(exc)
            logger.error(
                "gemini generation failed",
                extra={"model": self._model, "code": error.code, "error": str(exc)[:300]},
            )
            raise error from exc

        content = self._extract_text(response)
        logger.debug(
            "gemini generation complete",
            extra={
                "model": self._model,
                "prompt_chars": len(prompt),
                "response_chars": len(content),
            },
        )
        return content

    def stream(
        self, prompt: str, *, system: str | None = None, json_mode: bool = False
    ) -> Iterator[str]:
        """Produce a completion as incremental deltas.

        Raises:
            LLMError: The provider failed mid-stream.
        """
        client = self._get_client()
        try:
            for chunk in client.models.generate_content_stream(
                model=self._model,
                contents=prompt,
                config=self._config(system=system, json_mode=json_mode),
            ):
                text = getattr(chunk, "text", None)
                if text:
                    yield str(text)
        except Exception as exc:
            error = self._translate(exc)
            logger.error(
                "gemini streaming failed",
                extra={"model": self._model, "code": error.code},
            )
            raise error from exc
