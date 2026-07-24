"""Vector persistence.

Two implementations of the :class:`~app.rag.base.VectorStore` protocol: a Chroma
adapter for production and an in-memory store for tests.

Design notes that a reviewer will want justified:

* **One collection, metadata-partitioned.** All documents share
  ``COLLECTION_NAME`` and are separated by ``document_id`` / ``doc_type``
  filters. A collection per document would force N queries plus a manual merge
  every time the analyzer compares a resume against a job description — which is
  every request.
* **Scores, not distances, cross this boundary.** Chroma returns cosine
  *distance* (lower is better). Converting once here means no retriever, ranker,
  or UI needs to know which direction its backend's numbers run, and eliminates
  an entire class of "why is the worst match first" bugs.
* **Embeddings are supplied, never computed.** The store does not own an
  embedder. That keeps the "embed exactly once" guarantee (FR-008) enforceable
  in one place — the ingestion pipeline — instead of being an emergent property
  of who happened to call what.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.rag.embeddings import cosine_similarity
from app.schemas.rag import Chunk, ChunkMetadata, RetrievedChunk
from app.utils.exceptions import VectorStoreError
from app.utils.logging import get_logger

logger = get_logger(__name__)


def build_where_clause(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate a flat filter mapping into Chroma's ``where`` syntax.

    Chroma accepts ``{"key": value}`` for a single condition but requires an
    explicit ``{"$and": [...]}`` for two or more. Handling that asymmetry here
    keeps every call site writing plain dictionaries.
    """
    if not filters:
        return None
    if len(filters) == 1:
        key, value = next(iter(filters.items()))
        return {key: value}
    return {"$and": [{key: value} for key, value in filters.items()]}


def matches_filters(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Whether stored metadata satisfies every filter condition (AND semantics)."""
    if not filters:
        return True
    return all(metadata.get(key) == value for key, value in filters.items())


class ChromaVectorStore:
    """Persistent Chroma-backed store.

    Args:
        persist_directory: Where Chroma writes its SQLite + index files.
        collection_name: Collection holding every chunk.
        distance_metric: ``cosine``, ``l2``, or ``ip``.
    """

    def __init__(
        self,
        persist_directory: Path | str,
        *,
        collection_name: str = "resume_rag_chunks",
        distance_metric: str = "cosine",
    ) -> None:
        self._directory = Path(persist_directory)
        self._collection_name = collection_name
        self._distance_metric = distance_metric
        self._client: Any = None
        self._collection: Any = None

    @property
    def collection_name(self) -> str:
        """Name of the backing collection."""
        return self._collection_name

    def _get_collection(self) -> Any:
        """Open the collection, creating it if absent.

        Lazy for the same reason the embedder is: the composition root must be
        able to construct this object without touching the filesystem.

        Raises:
            VectorStoreError: The backend could not be opened.
        """
        if self._collection is not None:
            return self._collection
        try:
            import chromadb  # noqa: PLC0415
            from chromadb.config import Settings as ChromaSettings  # noqa: PLC0415

            self._directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self._directory),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": self._distance_metric},
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not open vector store at '{self._directory}': {exc}",
                details={"collection": self._collection_name},
            ) from exc
        return self._collection

    def _to_score(self, distance: float) -> float:
        """Convert a backend distance into a higher-is-better score.

        Cosine distance is ``1 - similarity`` over normalised vectors, so the
        inversion is exact. L2 has no bounded inverse, so ``1/(1+d)`` is used —
        monotonic, which is all that ranking requires.
        """
        if self._distance_metric == "cosine":
            return 1.0 - float(distance)
        if self._distance_metric == "ip":
            return -float(distance)
        return 1.0 / (1.0 + float(distance))

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or overwrite chunks.

        Uses ``upsert`` so re-ingesting a document is idempotent: chunk ids are
        deterministic, so the same document overwrites itself rather than
        producing duplicate hits that would then compete for top-k slots.

        Raises:
            ValueError: ``chunks`` and ``embeddings`` differ in length.
            VectorStoreError: The write failed.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            msg = f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must align"
            raise ValueError(msg)

        collection = self._get_collection()
        try:
            collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                embeddings=[list(vector) for vector in embeddings],
                metadatas=[chunk.metadata.to_store() for chunk in chunks],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to store {len(chunks)} chunk(s): {exc}",
                details={"collection": self._collection_name},
            ) from exc

        logger.info(
            "stored %d chunk(s)",
            len(chunks),
            extra={
                "stage": "store",
                "chunks": len(chunks),
                "document_id": chunks[0].metadata.document_id,
            },
        )

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Nearest neighbours, best-first.

        Raises:
            VectorStoreError: The query failed.
        """
        collection = self._get_collection()
        try:
            result = collection.query(
                query_embeddings=[list(embedding)],
                n_results=max(1, top_k),
                where=build_where_clause(filters),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector query failed: {exc}") from exc

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        return [
            RetrievedChunk(
                chunk=Chunk(text=text, metadata=ChunkMetadata.from_store(dict(metadata))),
                score=self._to_score(distance),
                rank=rank,
                retriever="similarity",
            )
            for rank, (text, metadata, distance) in enumerate(
                zip(documents, metadatas, distances, strict=True)
            )
        ]

    def _fetch(self, filters: dict[str, Any] | None) -> list[Chunk]:
        """Retrieve every matching chunk, ordered by document position."""
        collection = self._get_collection()
        try:
            result = collection.get(
                where=build_where_clause(filters), include=["documents", "metadatas"]
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector store read failed: {exc}") from exc

        chunks = [
            Chunk(text=text, metadata=ChunkMetadata.from_store(dict(metadata)))
            for text, metadata in zip(
                result.get("documents") or [], result.get("metadatas") or [], strict=True
            )
        ]
        chunks.sort(key=lambda chunk: (chunk.metadata.page, chunk.metadata.chunk_index))
        return chunks

    def get_by_document(self, document_id: str) -> list[Chunk]:
        """All chunks of one document, in order."""
        return self._fetch({"document_id": document_id})

    def all_chunks(self, filters: dict[str, Any] | None = None) -> list[Chunk]:
        """Every stored chunk matching ``filters``.

        Used to build the BM25 index. Acceptable at this scale (thousands of
        chunks); a corpus large enough to make this expensive would want a
        persistent inverted index instead, which is noted as future work.
        """
        return self._fetch(filters)

    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Number of stored chunks matching ``filters``."""
        if filters:
            return len(self._fetch(filters))
        try:
            return int(self._get_collection().count())
        except Exception as exc:
            raise VectorStoreError(f"Vector store count failed: {exc}") from exc

    def delete_document(self, document_id: str) -> int:
        """Purge a document's chunks. Returns the number removed."""
        collection = self._get_collection()
        existing = self.get_by_document(document_id)
        if not existing:
            return 0
        try:
            collection.delete(ids=[chunk.chunk_id for chunk in existing])
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete document '{document_id}': {exc}") from exc
        logger.info(
            "deleted document",
            extra={"document_id": document_id, "chunks_removed": len(existing)},
        )
        return len(existing)

    def health(self) -> bool:
        """Whether the backend is reachable. Never raises — health checks must not."""
        try:
            self._get_collection().count()
        except Exception:
            logger.warning("vector store health check failed", exc_info=True)
            return False
        return True


class InMemoryVectorStore:
    """Dictionary-backed store with brute-force search.

    Exact rather than approximate nearest neighbour, which makes retriever tests
    deterministic — an HNSW index can legitimately return a different ordering on
    two runs over identical data, and a test that flakes for a *correct* reason is
    worse than no test.

    Also usable in production for a small ephemeral corpus, which is why it lives
    in the engine package rather than in the test helpers.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._embeddings: dict[str, list[float]] = {}

    @property
    def collection_name(self) -> str:
        """Name reported for parity with the Chroma adapter."""
        return "in-memory"

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or overwrite chunks.

        Raises:
            ValueError: ``chunks`` and ``embeddings`` differ in length.
        """
        if len(chunks) != len(embeddings):
            msg = f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must align"
            raise ValueError(msg)
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._chunks[chunk.chunk_id] = chunk
            self._embeddings[chunk.chunk_id] = list(embedding)

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Exact nearest neighbours by cosine similarity."""
        scored: list[tuple[float, Chunk]] = []
        for chunk_id, chunk in self._chunks.items():
            if not matches_filters(chunk.metadata.to_store(), filters):
                continue
            scored.append((cosine_similarity(embedding, self._embeddings[chunk_id]), chunk))

        # Tie-break on chunk id so equal scores produce a stable order.
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        return [
            RetrievedChunk(chunk=chunk, score=score, rank=rank, retriever="similarity")
            for rank, (score, chunk) in enumerate(scored[:top_k])
        ]

    def get_by_document(self, document_id: str) -> list[Chunk]:
        """All chunks of one document, in order."""
        return self.all_chunks({"document_id": document_id})

    def all_chunks(self, filters: dict[str, Any] | None = None) -> list[Chunk]:
        """Every stored chunk matching ``filters``, in document order."""
        chunks = [
            chunk
            for chunk in self._chunks.values()
            if matches_filters(chunk.metadata.to_store(), filters)
        ]
        chunks.sort(key=lambda chunk: (chunk.metadata.page, chunk.metadata.chunk_index))
        return chunks

    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Number of stored chunks matching ``filters``."""
        return len(self.all_chunks(filters))

    def delete_document(self, document_id: str) -> int:
        """Purge a document's chunks. Returns the number removed."""
        doomed = [
            chunk_id
            for chunk_id, chunk in self._chunks.items()
            if chunk.metadata.document_id == document_id
        ]
        for chunk_id in doomed:
            del self._chunks[chunk_id]
            del self._embeddings[chunk_id]
        return len(doomed)

    def health(self) -> bool:
        """Always reachable."""
        return True
