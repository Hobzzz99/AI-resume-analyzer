"""Groq generation adapter.

Implements :class:`~app.rag.base.LLMClient` over LangChain's ``ChatGroq``. Two
responsibilities, and nothing else:

1. **Translate.** Prompt in, string out. The engine never sees a message object,
   a provider response, or a token-usage dict.
2. **Normalise failures.** Provider exceptions arrive as a mix of transport
   errors, HTTP errors, and library-specific types. They leave here as
   ``LLMRateLimitError``, ``LLMTimeoutError``, or ``LLMError`` — three cases the
   API layer maps to 429, 504, and 502 without inspecting anything.

The model id is never hardcoded (FR-021, Principle VII); it arrives from
configuration and is reported by ``/health`` so the running model is always
knowable from outside the process.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.utils.exceptions import ConfigurationError, LLMError, LLMRateLimitError, LLMTimeoutError
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Substrings matched against the provider's error text. String matching is
# unattractive but is the only portable signal: the provider surfaces rate limits
# and timeouts through several exception types across transport layers, and
# depending on those concrete types would couple this module to a library's
# internals that change more often than its error messages do.
# A 413 "request too large" is NOT a rate limit: waiting does not shrink the
# request, so retrying is pointless. It must surface immediately with guidance to
# reduce the prompt budget. Checked before the rate-limit markers because the
# provider's 413 message mentions "tokens per minute", which would otherwise be
# misread as a transient limit and retried uselessly.
_TOO_LARGE_MARKERS = ("413", "request too large", "reduce your message", "please reduce")
_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "429", "too many requests", "quota")
_TIMEOUT_MARKERS = ("timeout", "timed out", "deadline")


class GroqClient:
    """Chat-completion client for Groq.

    Args:
        api_key: Groq API key.
        model: Model identifier, from configuration.
        temperature: Sampling temperature. Defaults low — this is structured
            extraction, where creativity is indistinguishable from fabrication.
        max_tokens: Response ceiling.
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
                "GROQ_API_KEY is not set. Add it to your .env file — see .env.example.",
                details={"setting": "GROQ_API_KEY"},
            )
        if not model.strip():
            raise ConfigurationError(
                "GROQ_MODEL is not set. Choose a model from https://console.groq.com/docs/models.",
                details={"setting": "GROQ_MODEL"},
            )

        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._clients: dict[bool, Any] = {}

    @property
    def model_name(self) -> str:
        """Resolved model identifier."""
        return self._model

    def _client(self, *, json_mode: bool) -> Any:
        """Return a client configured for the requested response format.

        Two instances are cached rather than one reconfigured, because
        ``model_kwargs`` is fixed at construction in ``ChatGroq``. JSON mode is
        requested at the provider level — it constrains decoding so malformed
        syntax is largely eliminated before the parser ever sees the text, which
        is strictly better than repairing it afterwards.
        """
        if json_mode in self._clients:
            return self._clients[json_mode]

        from langchain_groq import ChatGroq  # noqa: PLC0415

        kwargs: dict[str, Any] = {
            "api_key": self._api_key,
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "timeout": self._timeout,
            # Retries are owned by StructuredGenerator, which retries with a
            # *repair prompt*. A blind resample here would burn the same budget
            # on the same failure and hide it from the retry count we report.
            "max_retries": 0,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        self._clients[json_mode] = ChatGroq(**kwargs)
        return self._clients[json_mode]

    @staticmethod
    def _messages(prompt: str, system: str | None) -> list[tuple[str, str]]:
        """Build the message list, omitting an empty system turn."""
        if system and system.strip():
            return [("system", system), ("human", prompt)]
        return [("human", prompt)]

    def _translate(self, exc: Exception) -> LLMError:
        """Map a provider exception onto this application's error taxonomy."""
        text = str(exc).lower()
        if any(marker in text for marker in _TOO_LARGE_MARKERS):
            return LLMError(
                "The request exceeds the model's per-request token limit. Reduce "
                "MAX_CONTEXT_CHARS or LLM_MAX_TOKENS in your .env, then restart the service.",
                details={"model": self._model, "provider_error": str(exc)[:500]},
            )
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

    def generate(self, prompt: str, *, system: str | None = None, json_mode: bool = False) -> str:
        """Produce a completion.

        Raises:
            LLMRateLimitError: Provider rate limit.
            LLMTimeoutError: Request exceeded the timeout.
            LLMError: Any other provider failure.
        """
        try:
            response = self._client(json_mode=json_mode).invoke(self._messages(prompt, system))
        except Exception as exc:
            error = self._translate(exc)
            logger.error(
                "llm generation failed",
                extra={"model": self._model, "code": error.code, "error": str(exc)[:300]},
            )
            raise error from exc

        content = response.content
        # Multimodal-capable models can return a content list even for text-only
        # prompts; flatten it rather than letting a list reach the JSON parser.
        if isinstance(content, list):
            content = "".join(
                part if isinstance(part, str) else str(part.get("text", "")) for part in content
            )

        logger.debug(
            "llm generation complete",
            extra={
                "model": self._model,
                "prompt_chars": len(prompt),
                "response_chars": len(content),
            },
        )
        return str(content)

    def stream(
        self, prompt: str, *, system: str | None = None, json_mode: bool = False
    ) -> Iterator[str]:
        """Produce a completion as incremental deltas.

        Raises:
            LLMError: The provider failed mid-stream.
        """
        try:
            for chunk in self._client(json_mode=json_mode).stream(self._messages(prompt, system)):
                content = chunk.content
                if isinstance(content, list):
                    content = "".join(
                        part if isinstance(part, str) else str(part.get("text", ""))
                        for part in content
                    )
                if content:
                    yield str(content)
        except Exception as exc:
            error = self._translate(exc)
            logger.error("llm streaming failed", extra={"model": self._model, "code": error.code})
            raise error from exc
