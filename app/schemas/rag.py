"""Domain-free data structures for the retrieval engine.

Constitution Principle I: nothing in this module knows what a resume is. The one
concession is :class:`DocumentType`, whose members happen to name the two roles
this application uses — but no engine code branches on the value; it is only ever
passed through to a metadata filter. An adopter in another domain uses
``GENERIC`` or extends the enum without touching engine logic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.config.settings import RetrievalStrategy

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "DocumentManifest",
    "DocumentType",
    "RetrievalPlan",
    "RetrievalPlanStep",
    "RetrievalStepTrace",
    "RetrievalStrategy",
    "RetrievalTrace",
    "RetrievedChunk",
    "SourceDocument",
    "StageTimings",
]


class DocumentType(StrEnum):
    """The role a document plays in a retrieval plan.

    Used exclusively as a metadata filter value. The engine never inspects which
    member it is.
    """

    RESUME = "resume"
    JOB_DESCRIPTION = "job_description"
    GENERIC = "generic"


class ChunkMetadata(BaseModel):
    """Provenance carried by every indexed passage.

    Every field here exists to answer a question in production: *which document
    did this come from* (``document_id``, ``filename``), *where in it* (``page``,
    ``chunk_index``), *can I filter on it* (``doc_type``), *is it stale*
    (``ingested_at``), and *how much prompt budget does it consume*
    (``char_count``).
    """

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Content fingerprint of the owning document.")
    filename: str = Field(description="Original filename, or a synthetic name for pasted text.")
    doc_type: DocumentType = Field(
        description="Role of the document; the primary retrieval filter."
    )
    page: int = Field(default=0, ge=0, description="1-based page number; 0 when unpaginated.")
    chunk_index: int = Field(default=0, ge=0, description="0-based position within the document.")
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    char_count: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chunk_id(self) -> str:
        """Stable, human-readable identifier.

        Deterministic from content and position, which is what makes re-ingestion
        idempotent: the same document produces the same ids and overwrites rather
        than duplicating.
        """
        return f"{self.document_id}:{self.page}:{self.chunk_index}"

    def to_store(self) -> dict[str, str | int]:
        """Flatten to primitives for a vector store's metadata column.

        Chroma rejects enums and datetimes. Doing the conversion here — once —
        keeps every call site free of serialisation concerns and guarantees the
        round trip is symmetric with :meth:`from_store`.
        """
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "doc_type": self.doc_type.value,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "ingested_at": self.ingested_at.isoformat(),
            "char_count": self.char_count,
        }

    @classmethod
    def from_store(cls, raw: dict[str, Any]) -> Self:
        """Rebuild from a store's flattened metadata."""
        ingested = raw.get("ingested_at")
        return cls(
            document_id=str(raw["document_id"]),
            filename=str(raw.get("filename", "unknown")),
            doc_type=DocumentType(raw.get("doc_type", DocumentType.GENERIC.value)),
            page=int(raw.get("page", 0)),
            chunk_index=int(raw.get("chunk_index", 0)),
            ingested_at=datetime.fromisoformat(ingested) if ingested else datetime.now(UTC),
            char_count=int(raw.get("char_count", 0)),
        )


class Chunk(BaseModel):
    """A contiguous passage of a document, with its provenance."""

    model_config = ConfigDict(frozen=True)

    text: str
    metadata: ChunkMetadata

    @property
    def chunk_id(self) -> str:
        """Shorthand for ``metadata.chunk_id``."""
        return self.metadata.chunk_id


class RetrievedChunk(BaseModel):
    """A chunk returned for a query, with its relevance and origin."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(description="Relevance, normalised so higher is always better.")
    rank: int = Field(ge=0, description="0-based position in the returned ordering.")
    retriever: str = Field(default="unknown", description="Which retriever produced this hit.")

    @property
    def chunk_id(self) -> str:
        """Shorthand for the underlying chunk id."""
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        """Shorthand for the underlying chunk text."""
        return self.chunk.text

    @property
    def citation(self) -> str:
        """The handle placed in the prompt and echoed back by the model.

        Format is chosen to be simultaneously readable by a human reviewer,
        greppable by a test asserting SC-004, and cheap in tokens.
        """
        meta = self.chunk.metadata
        return f"[{meta.filename} p.{meta.page} #{meta.chunk_index}]"


class RetrievalPlanStep(BaseModel):
    """One facet query within a retrieval plan.

    ``top_k`` and ``strategy`` are optional so a plan can express "use whatever
    is configured" for most steps and override it for the one that needs to.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Facet label; becomes a section header in the prompt.")
    query: str
    doc_type: DocumentType | None = None
    document_id: str | None = None
    top_k: int | None = Field(default=None, ge=1)
    strategy: RetrievalStrategy | None = None

    def filters(self) -> dict[str, Any]:
        """Metadata filter for this step, omitting unset keys."""
        built: dict[str, Any] = {}
        if self.doc_type is not None:
            built["doc_type"] = self.doc_type.value
        if self.document_id is not None:
            built["document_id"] = self.document_id
        return built


class RetrievalPlan(BaseModel):
    """An ordered set of facet queries.

    The plan is *data*. That is the whole trick behind Principle I: the resume
    domain expresses "look for programming languages, then cloud experience, then
    certifications" as a value the engine executes, not as code inside the engine.
    """

    model_config = ConfigDict(frozen=True)

    steps: tuple[RetrievalPlanStep, ...]

    def __len__(self) -> int:
        return len(self.steps)


class RetrievalStepTrace(BaseModel):
    """What one plan step actually did."""

    name: str
    query: str
    strategy: str
    candidates: int = Field(default=0, description="Chunks considered before top-k truncation.")
    returned: int = Field(default=0)
    duration_ms: float = 0.0
    cached: bool = False
    top_score: float | None = None


class RetrievalTrace(BaseModel):
    """Per-request retrieval diagnostics.

    Returned in the API response and written to the log. This is the artefact
    that turns "the analysis is wrong" into "the analysis is wrong *because the
    cloud-experience step returned three chunks about a university course*",
    which is the only version of that report anyone can act on.
    """

    steps: list[RetrievalStepTrace] = Field(default_factory=list)
    total_chunks: int = 0
    unique_chunks: int = 0
    deduplicated: int = 0
    budget_truncated: bool = False
    reranked: bool = False

    def add(self, step: RetrievalStepTrace) -> None:
        """Record a completed step."""
        self.steps.append(step)
        self.total_chunks += step.returned


class StageTimings(BaseModel):
    """Wall-clock duration of every pipeline stage, in milliseconds.

    ``None`` means "this stage did not run" — which is itself information: a
    cached ingestion has ``embed_ms is None``, and that is exactly how SC-006 is
    verified from the outside.
    """

    load_ms: float | None = None
    clean_ms: float | None = None
    split_ms: float | None = None
    embed_ms: float | None = None
    store_ms: float | None = None
    retrieve_ms: float | None = None
    rerank_ms: float | None = None
    prompt_ms: float | None = None
    llm_ms: float | None = None
    parse_ms: float | None = None
    total_ms: float | None = None

    def record(self, stage: str, duration_ms: float) -> None:
        """Accumulate a duration for ``stage``.

        Accumulates rather than overwrites: a plan with six steps calls
        ``record("retrieve", ...)`` six times and the total is what matters.
        Unknown stage names are ignored rather than raising, so instrumenting a
        new stage can never break a request path.
        """
        field = f"{stage}_ms"
        if field in type(self).model_fields:
            current = getattr(self, field) or 0.0
            setattr(self, field, round(current + duration_ms, 3))

    def as_reported(self) -> dict[str, float]:
        """Non-null timings only, for logging."""
        return {
            key: value
            for key, value in self.model_dump().items()
            if isinstance(value, (int, float))
        }


class SourceDocument(BaseModel):
    """Text extracted from one physical source, before splitting."""

    model_config = ConfigDict(frozen=True)

    text: str
    filename: str
    page: int = Field(default=0, ge=0)


class DocumentManifest(BaseModel):
    """The record of an ingested document.

    Persisted as JSON and returned from the upload endpoints. Chroma is a poor
    place to ask "what documents exist?" — that question would require scanning
    the whole collection — so the manifest is the index of the index.
    """

    document_id: str
    filename: str
    doc_type: DocumentType
    page_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    char_count: int = Field(default=0, ge=0)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cached: bool = Field(
        default=False,
        description="True when the fingerprint was already indexed and no embedding work ran.",
    )
    timings: StageTimings = Field(default_factory=StageTimings)
