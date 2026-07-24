"""Typed error hierarchy.

Every failure mode in this system is one of these classes. Each carries the
machine-readable ``code`` and the HTTP status it maps to, so the API layer needs
exactly one exception handler and no ``if isinstance`` ladder. Adding a failure
mode means adding a class here — it cannot be forgotten at the boundary.

This is the mechanism behind FR-031 ("distinct, meaningful failure responses"):
a caller can branch on ``code`` without parsing prose.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base for every expected failure in the application.

    Attributes:
        code: Stable machine-readable identifier, part of the public API contract.
        http_status: The status the API layer returns for this class of failure.
        message: Human-readable explanation, safe to show a user.
        details: Structured context for debugging. Must never contain secrets.
    """

    code: str = "INTERNAL_ERROR"
    http_status: int = 500

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self, request_id: str | None = None) -> dict[str, Any]:
        """Render the error envelope defined in ``contracts/api.md``."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "request_id": request_id,
            }
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# --------------------------------------------------------------- ingestion ---


class UnsupportedFileTypeError(AppError):
    """The submitted file's extension or content type is not accepted."""

    code = "UNSUPPORTED_FILE_TYPE"
    http_status = 415


class FileTooLargeError(AppError):
    """The submitted file exceeds the configured size ceiling."""

    code = "FILE_TOO_LARGE"
    http_status = 413


class InvalidDocumentError(AppError):
    """The file could not be parsed: corrupt, encrypted, or malformed."""

    code = "INVALID_DOCUMENT"
    http_status = 422


class EmptyDocumentError(AppError):
    """The file parsed successfully but yielded no usable text.

    Almost always an image-only scan. OCR is deliberately out of scope, so this
    is reported as a distinct, actionable failure rather than an empty analysis.
    """

    code = "EMPTY_DOCUMENT"
    http_status = 422


class DocumentNotFoundError(AppError):
    """The referenced document id is not present in the index."""

    code = "DOCUMENT_NOT_FOUND"
    http_status = 404


# ----------------------------------------------------------------- engine ---


class EmbeddingError(AppError):
    """The embedding model failed to load or to encode a batch."""

    code = "EMBEDDING_FAILED"
    http_status = 503


class VectorStoreError(AppError):
    """The vector store was unreachable, or a read/write failed."""

    code = "VECTOR_STORE_ERROR"
    http_status = 503


class InsufficientContextError(AppError):
    """Retrieval returned too few passages to ground an answer.

    Raised instead of prompting the model with a near-empty context block, which
    is the situation in which models fabricate most freely (Principle IV).
    """

    code = "INSUFFICIENT_CONTEXT"
    http_status = 422


class ContextBudgetExceededError(AppError):
    """More context was assembled than the configured prompt budget permits.

    A programming error rather than a user error: the pipeline is expected to
    truncate to budget before the builder sees the chunks. Surfacing it loudly
    is what keeps Principle II ("never the full document") honest.
    """

    code = "CONTEXT_BUDGET_EXCEEDED"
    http_status = 500


# -------------------------------------------------------------------- llm ---


class LLMError(AppError):
    """The generation provider returned an error or was unreachable."""

    code = "LLM_ERROR"
    http_status = 502


class LLMRateLimitError(LLMError):
    """The generation provider rate-limited the request.

    Surfaced rather than retried indefinitely: on a free tier, silent retry loops
    turn one slow request into a sustained ban.
    """

    code = "LLM_RATE_LIMITED"
    http_status = 429


class LLMTimeoutError(LLMError):
    """Generation exceeded the configured timeout."""

    code = "LLM_TIMEOUT"
    http_status = 504


class OutputValidationError(AppError):
    """The model's output failed schema validation and the repair budget ran out.

    The last set of validation errors travels in ``details`` so an operator can
    tell "the model refused" apart from "the schema is too strict".
    """

    code = "OUTPUT_VALIDATION_FAILED"
    http_status = 422


class ConfigurationError(AppError):
    """A required setting is missing or invalid."""

    code = "CONFIGURATION_ERROR"
    http_status = 500


class PromptTemplateNotFoundError(AppError):
    """The requested prompt template is not registered."""

    code = "PROMPT_TEMPLATE_NOT_FOUND"
    http_status = 404
