"""HTTP request and response envelopes.

Separate from the domain schemas on purpose. ``ResumeAnalysis`` is the contract
with the *model*; these are the contract with the *client*. Fusing them would
mean a field added for the UI leaks into the prompt schema, and a prompt tweak
becomes a breaking API change.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from app.config.settings import RetrievalStrategy


class ErrorDetail(BaseModel):
    """The body of an error response."""

    code: str = Field(description="Stable machine-readable error identifier.")
    message: str = Field(description="Human-readable explanation, safe to display.")
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Uniform error envelope for every failure.

    One shape for every error means a client writes one error handler and
    branches on ``code`` — instead of guessing whether a 422 carries FastAPI's
    validation shape or the application's.
    """

    error: ErrorDetail


class JobDescriptionRequest(BaseModel):
    """Pasted job description text."""

    text: str = Field(min_length=1, max_length=200_000)
    title: str = Field(default="", max_length=200, description="Role title, used in the prompt.")


class AnalyzeRequest(BaseModel):
    """A request to analyze one resume against one job description."""

    resume_document_id: str = Field(min_length=1)
    job_document_id: str = Field(min_length=1)
    job_title: str = Field(default="", max_length=200)
    top_k: int | None = Field(default=None, ge=1, le=20, description="Passages per facet.")
    strategy: RetrievalStrategy | None = None
    prompt_template: str | None = None
    use_cache: bool = True

    @model_validator(mode="after")
    def _distinct_documents(self) -> Self:
        """Reject analysing a document against itself.

        Not a hypothetical: because document ids are content fingerprints,
        uploading the same file to both endpoints yields the same id, and the
        analysis would score a document against itself and return a meaningless
        near-perfect match. Catching it here gives a clear message instead.

        Raises:
            ValueError: The two ids are identical.
        """
        if self.resume_document_id == self.job_document_id:
            msg = (
                "resume_document_id and job_document_id are identical — the same document "
                "cannot be analysed against itself"
            )
            raise ValueError(msg)
        return self


class ChatRequest(BaseModel):
    """A follow-up question over ingested documents."""

    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] = Field(min_length=1, max_length=10)
    top_k: int | None = Field(default=None, ge=1, le=20)


class ChatResponseBody(BaseModel):
    """A grounded answer with its provenance."""

    session_id: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    retrieval: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)


class ComponentStatus(BaseModel):
    """Health of one dependency."""

    name: str
    status: str = Field(description="ok | degraded | unavailable")
    detail: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Service health and resolved configuration.

    Reporting the *resolved* model ids is the point: FR-021 forbids hardcoding
    them, so this endpoint is how anyone confirms which model a running process
    is actually using without reading its environment.
    """

    status: str = Field(description="ok | degraded")
    version: str
    environment: str
    uptime_seconds: float
    llm: dict[str, Any]
    embeddings: dict[str, Any]
    vector_store: dict[str, Any]
    retrieval: dict[str, Any]
    prompts: dict[str, Any]
    cache: dict[str, Any] = Field(default_factory=dict)


class DeleteResponse(BaseModel):
    """Outcome of a document deletion."""

    document_id: str
    chunks_removed: int
