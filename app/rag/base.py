"""Structural interfaces for every replaceable engine component.

Constitution Principle V. These are :class:`typing.Protocol` classes, not
abstract base classes, and that choice is deliberate:

* **No inheritance coupling.** A third-party object satisfies ``Embedder`` by
  having the right methods. An adopter can wrap OpenAI embeddings without
  importing anything from this package.
* **Fakes stay honest.** ``HashingEmbedder`` in the test suite is checked
  structurally against the same protocol the production class satisfies, so a
  signature change breaks the fake at type-check time rather than at runtime in
  a test that was quietly passing against a stale double.

All protocols are ``@runtime_checkable`` so the composition root can assert its
wiring, and every one of them is what makes SC-009 (a suite that runs offline)
achievable rather than aspirational.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from app.schemas.rag import Chunk, RetrievedChunk, SourceDocument

SchemaT = TypeVar("SchemaT", bound=BaseModel)
"""The answer shape a pipeline produces.

Bound to ``BaseModel`` because Constitution Principle III makes validation
non-negotiable: an unvalidatable answer type is not a legal target for this
engine.
"""


@runtime_checkable
class DocumentLoader(Protocol):
    """Extracts page-attributed text from a source file."""

    def supports(self, path_or_name: str) -> bool:
        """Whether this loader handles the given filename or path."""
        ...

    def load(self, path: str) -> list[SourceDocument]:
        """Extract the file into one :class:`SourceDocument` per page.

        Raises:
            InvalidDocumentError: The file could not be parsed.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors.

    Queries and documents are embedded through separate methods because
    asymmetric models (and future instruction-tuned embedders) prefix them
    differently. MiniLM treats them identically, but encoding that assumption
    into a single ``embed()`` would make swapping in an asymmetric model a
    breaking change across every call site.
    """

    @property
    def dimension(self) -> int:
        """Vector dimensionality."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier of the underlying model, for reporting through /health."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of passages."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Persists chunks with their vectors and searches them."""

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or overwrite chunks. Idempotent on ``chunk_id``."""
        ...

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Nearest neighbours, ordered best-first with higher-is-better scores.

        Normalising the score direction here — rather than leaving cosine
        *distance* for callers to reinterpret — means no downstream code has to
        know which way its backend's numbers run.
        """
        ...

    def get_by_document(self, document_id: str) -> list[Chunk]:
        """All chunks belonging to one document, ordered by position."""
        ...

    def all_chunks(self, filters: dict[str, Any] | None = None) -> list[Chunk]:
        """Every stored chunk matching ``filters``. Used to build the BM25 index."""
        ...

    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Number of stored chunks matching ``filters``."""
        ...

    def delete_document(self, document_id: str) -> int:
        """Remove a document's chunks. Returns how many were deleted."""
        ...

    def health(self) -> bool:
        """Whether the backend is reachable."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Selects relevant chunks for a query."""

    @property
    def name(self) -> str:
        """Strategy label, recorded in the retrieval trace."""
        ...

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks, best-first."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders retrieved candidates by a more precise relevance judgement."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return the best ``top_k`` candidates in the new order."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    """Generates text from a prompt.

    Kept intentionally narrow: two methods, plain strings in and out. Anything
    richer would leak a provider's request shape into the engine and make the
    ``ScriptedLLMClient`` fake a burden to maintain.
    """

    @property
    def model_name(self) -> str:
        """Resolved model identifier, reported by /health."""
        ...

    def generate(self, prompt: str, *, system: str | None = None, json_mode: bool = False) -> str:
        """Produce a completion."""
        ...

    def stream(
        self, prompt: str, *, system: str | None = None, json_mode: bool = False
    ) -> Iterator[str]:
        """Produce a completion as incremental deltas."""
        ...


@runtime_checkable
class PromptTemplateProvider(Protocol):
    """Supplies named, versioned prompt templates."""

    def get(self, name: str) -> Any:
        """Return the template registered under ``name``.

        Raises:
            PromptTemplateNotFoundError: No such template.
        """
        ...


@dataclass(slots=True)
class StructuredResult(Generic[SchemaT]):
    """A validated model response with the provenance of how it was obtained.

    ``retry_count`` and ``attempts`` are part of the return value rather than log
    lines because they are product-relevant: an analysis that needed two repair
    rounds is one a reviewer should look at more carefully, and the API surfaces
    the number for exactly that reason.
    """

    value: SchemaT
    raw: str
    retry_count: int = 0
    attempts: list[str] = field(default_factory=list)


@runtime_checkable
class StructuredGenerator(Protocol):
    """Produces a validated instance of a caller-supplied schema.

    This is the seam that keeps the generic pipeline free of any dependency on
    the LLM package: the pipeline is handed something that turns a prompt into a
    validated object, and never learns which provider, JSON mode, or repair
    strategy is behind it.
    """

    @property
    def model_name(self) -> str:
        """Resolved model identifier."""
        ...

    def generate(
        self,
        prompt: str,
        schema: type[SchemaT],
        *,
        system: str | None = None,
    ) -> StructuredResult[SchemaT]:
        """Generate and validate.

        Raises:
            OutputValidationError: The repair budget was exhausted.
            LLMError: The provider failed.
        """
        ...


@runtime_checkable
class ResultCache(Protocol):
    """Stores completed results against a key."""

    def get(self, key: str) -> Any | None:
        """Return the cached value, or ``None`` on a miss."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Store a value."""
        ...

    def clear(self) -> None:
        """Drop everything."""
        ...
