"""Chunking stage.

Wraps LangChain's ``RecursiveCharacterTextSplitter`` and attaches full provenance
to every chunk. The splitter is chosen for a specific property: it tries a
descending list of separators — paragraph, line, sentence, word — and only falls
back to a hard character cut when none apply. For resumes that means it breaks at
blank lines between roles first, keeping a bullet's subject and its metrics in one
chunk. A fixed-width splitter would routinely cut ``"reduced inference latency
by"`` from ``"40%"``, and no amount of overlap reliably recovers that.

Chunk size 1000 / overlap 200 are the configured defaults. The 20% overlap is
what makes a claim that straddles a boundary retrievable from either side.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.schemas.rag import Chunk, ChunkMetadata, DocumentType, SourceDocument
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Descending granularity. The blank-line separator is first because it is the
# strongest structural boundary in a resume; the empty string is the last-resort
# hard cut that guarantees the size bound is respected.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", "")


class DocumentSplitter:
    """Splits documents into overlapping, fully attributed chunks.

    Args:
        chunk_size: Target characters per chunk.
        chunk_overlap: Characters shared between neighbours. Must be smaller
            than ``chunk_size`` — enforced in ``Settings`` at startup.
        min_chunk_chars: Chunks shorter than this are discarded. A 12-character
            chunk ("References") carries no retrievable signal but does occupy a
            top-k slot, so dropping it strictly improves retrieval.
        separators: Split points in descending priority.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_chars: int = 50,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_overlap >= chunk_size:
            msg = f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
            raise ValueError(msg)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars
        self._separators = list(separators)
        self._splitter = self._build_splitter()

    def _build_splitter(self):  # type: ignore[no-untyped-def] # third-party return type
        """Construct the underlying LangChain splitter.

        Imported lazily so that constructing this object in a unit test does not
        pay for the LangChain import tree.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415

        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self._separators,
            length_function=len,
            keep_separator=True,
        )

    def split(
        self,
        documents: Sequence[SourceDocument],
        *,
        document_id: str,
        doc_type: DocumentType,
    ) -> list[Chunk]:
        """Split pages into chunks, carrying provenance into each one.

        ``chunk_index`` is continuous across the whole document rather than
        restarting per page. That makes it a genuine ordering key: chunk 7 always
        follows chunk 6, whether or not a page boundary sits between them, which
        is what lets the prompt builder present retrieved passages in document
        order instead of an arbitrary one.

        Args:
            documents: Cleaned, page-attributed source text.
            document_id: Content fingerprint of the owning document.
            doc_type: Role used as the primary retrieval filter.

        Returns:
            Chunks in document order. May be empty if every candidate fell below
            ``min_chunk_chars`` — the caller is responsible for treating that as
            an empty document.
        """
        chunks: list[Chunk] = []
        index = 0
        discarded = 0

        for document in documents:
            for piece in self._splitter.split_text(document.text):
                text = piece.strip()
                if len(text) < self.min_chunk_chars:
                    discarded += 1
                    continue

                chunks.append(
                    Chunk(
                        text=text,
                        metadata=ChunkMetadata(
                            document_id=document_id,
                            filename=document.filename,
                            doc_type=doc_type,
                            page=document.page,
                            chunk_index=index,
                            char_count=len(text),
                        ),
                    )
                )
                index += 1

        logger.info(
            "split document into %d chunk(s)",
            len(chunks),
            extra={
                "stage": "split",
                "document_id": document_id,
                "chunks": len(chunks),
                "discarded": discarded,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        )
        return chunks
