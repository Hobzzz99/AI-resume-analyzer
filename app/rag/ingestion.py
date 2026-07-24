"""Ingestion pipeline: load → clean → split → fingerprint → embed → store.

The ordering is not arbitrary. Cleaning precedes fingerprinting so that the same
resume exported twice with different whitespace produces the *same* document id.
Fingerprinting precedes embedding so the cache check happens before the only
expensive stage. Storing happens last and in one call, so a failure anywhere
earlier leaves no partial index behind (an invariant the spec's edge cases
require).

Constitution Principle I: nothing here knows what kind of document it is handling.
``doc_type`` is an opaque label carried through to metadata.
"""

from __future__ import annotations

from pathlib import Path

from app.rag.base import Embedder, VectorStore
from app.rag.cleaner import TextCleaner
from app.rag.loaders import LoaderRegistry, RawTextLoader
from app.rag.splitter import DocumentSplitter
from app.schemas.rag import Chunk, DocumentManifest, DocumentType, SourceDocument, StageTimings
from app.utils.exceptions import EmptyDocumentError
from app.utils.hashing import content_fingerprint
from app.utils.logging import get_logger
from app.utils.timing import Stopwatch

logger = get_logger(__name__)


class IngestionResult:
    """Outcome of one ingestion: the manifest plus the chunks that were produced.

    Chunks are returned alongside the manifest so callers that want to inspect
    what was indexed do not have to query the store back — and so tests can
    assert on chunk content without reaching into persistence.
    """

    __slots__ = ("chunks", "manifest")

    def __init__(self, manifest: DocumentManifest, chunks: list[Chunk]) -> None:
        self.manifest = manifest
        self.chunks = chunks


class IngestionPipeline:
    """Turns a source file or string into indexed, retrievable chunks.

    Args:
        loaders: Extraction registry, dispatching by file type.
        cleaner: Normalisation stage.
        splitter: Chunking stage.
        embedder: Vector encoder.
        store: Persistence backend.
        chunking_signature: Configuration identity folded into the document
            fingerprint, so a change to chunk size invalidates cached documents
            rather than mixing two chunk geometries in one collection.
    """

    def __init__(
        self,
        *,
        loaders: LoaderRegistry,
        cleaner: TextCleaner,
        splitter: DocumentSplitter,
        embedder: Embedder,
        store: VectorStore,
        chunking_signature: str = "",
    ) -> None:
        self._loaders = loaders
        self._cleaner = cleaner
        self._splitter = splitter
        self._embedder = embedder
        self._store = store
        self._signature = chunking_signature

    # ------------------------------------------------------------- public ---

    def ingest_file(self, path: Path | str, *, doc_type: DocumentType) -> IngestionResult:
        """Ingest a file from disk.

        Raises:
            UnsupportedFileTypeError: No loader handles this file type.
            InvalidDocumentError: The file could not be parsed.
            EmptyDocumentError: No usable text was extracted.
        """
        timings = StageTimings()
        with Stopwatch("load", timings, filename=Path(path).name):
            pages = self._loaders.load(str(path))
        return self._ingest(pages, doc_type=doc_type, timings=timings)

    def ingest_text(
        self, text: str, *, filename: str, doc_type: DocumentType
    ) -> IngestionResult:
        """Ingest an in-memory string.

        Takes the identical path as :meth:`ingest_file` from cleaning onward, so
        a pasted job description is indexed exactly like an uploaded one and no
        behaviour can diverge between the two entry points.
        """
        timings = StageTimings()
        with Stopwatch("load", timings, filename=filename):
            pages = RawTextLoader(text, filename=filename).load()
        return self._ingest(pages, doc_type=doc_type, timings=timings)

    # ------------------------------------------------------------ internal ---

    def _ingest(
        self,
        pages: list[SourceDocument],
        *,
        doc_type: DocumentType,
        timings: StageTimings,
    ) -> IngestionResult:
        """Run the pipeline from cleaning onward."""
        filename = pages[0].filename if pages else "unknown"

        with Stopwatch("clean", timings, filename=filename):
            cleaned = self._cleaner.clean_documents(pages)

        if not cleaned:
            raise EmptyDocumentError(
                f"No usable text remained in '{filename}' after cleaning. The document "
                f"appears to contain no readable content.",
                details={"filename": filename, "pages": len(pages)},
            )

        document_id = self._fingerprint(cleaned)

        cached = self._load_if_indexed(document_id, filename, doc_type, cleaned, timings)
        if cached is not None:
            return cached

        with Stopwatch("split", timings, document_id=document_id):
            chunks = self._splitter.split(cleaned, document_id=document_id, doc_type=doc_type)

        if not chunks:
            raise EmptyDocumentError(
                f"'{filename}' produced no indexable passages. It may be too short "
                f"or contain only formatting.",
                details={"filename": filename},
            )

        with Stopwatch("embed", timings, chunks=len(chunks)):
            embeddings = self._embedder.embed_documents([chunk.text for chunk in chunks])

        with Stopwatch("store", timings, chunks=len(chunks)):
            self._store.add(chunks, embeddings)

        manifest = self._manifest(
            document_id=document_id,
            filename=filename,
            doc_type=doc_type,
            pages=cleaned,
            chunks=chunks,
            timings=timings,
            cached=False,
        )
        logger.info(
            "ingested document",
            extra={
                "document_id": document_id,
                "source_file": filename,
                "doc_type": doc_type.value,
                "chunks": len(chunks),
                **timings.as_reported(),
            },
        )
        return IngestionResult(manifest, chunks)

    def _fingerprint(self, pages: list[SourceDocument]) -> str:
        """Content-derived document id.

        Computed from cleaned text so cosmetic differences do not produce a new
        id, and salted with the chunking signature so a configuration change
        does (research.md R7).
        """
        combined = "\n".join(page.text for page in pages)
        return content_fingerprint(combined, salt=self._signature)

    def _load_if_indexed(
        self,
        document_id: str,
        filename: str,
        doc_type: DocumentType,
        pages: list[SourceDocument],
        timings: StageTimings,
    ) -> IngestionResult | None:
        """Return the existing index for this fingerprint, if there is one.

        This is the mechanism behind FR-008 and SC-006. The returned manifest has
        ``embed_ms is None``, which is how a caller — or a test — proves no
        embedding work ran, rather than having to trust a boolean flag.
        """
        existing = self._store.get_by_document(document_id)
        if not existing:
            return None

        logger.info(
            "document already indexed; skipping embedding",
            extra={"document_id": document_id, "source_file": filename, "chunks": len(existing)},
        )
        manifest = self._manifest(
            document_id=document_id,
            filename=filename,
            doc_type=doc_type,
            pages=pages,
            chunks=existing,
            timings=timings,
            cached=True,
        )
        return IngestionResult(manifest, existing)

    @staticmethod
    def _manifest(
        *,
        document_id: str,
        filename: str,
        doc_type: DocumentType,
        pages: list[SourceDocument],
        chunks: list[Chunk],
        timings: StageTimings,
        cached: bool,
    ) -> DocumentManifest:
        """Assemble the manifest describing an ingested document."""
        return DocumentManifest(
            document_id=document_id,
            filename=filename,
            doc_type=doc_type,
            page_count=len({page.page for page in pages}),
            chunk_count=len(chunks),
            char_count=sum(len(page.text) for page in pages),
            cached=cached,
            timings=timings,
        )
