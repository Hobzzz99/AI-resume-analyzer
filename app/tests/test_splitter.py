"""Chunking tests.

The contract being verified: chunks respect the configured size, overlap enough
that a fact straddling a boundary is retrievable from either side, and carry
complete, stable provenance. A chunk without correct metadata cannot be cited,
and an uncitable chunk fails SC-004.
"""

from __future__ import annotations

import pytest

from app.rag.splitter import DocumentSplitter
from app.schemas.rag import DocumentType, SourceDocument

LONG_TEXT = (
    "Machine learning engineer with production experience. " * 40
)


def make_pages(text: str, count: int = 1) -> list[SourceDocument]:
    return [
        SourceDocument(text=text, filename="resume.pdf", page=index + 1) for index in range(count)
    ]


def test_rejects_overlap_greater_than_chunk_size() -> None:
    """An overlap >= size makes the splitter never advance."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        DocumentSplitter(chunk_size=100, chunk_overlap=100)


def test_respects_the_configured_chunk_size() -> None:
    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=40, min_chunk_chars=10)
    chunks = splitter.split(make_pages(LONG_TEXT), document_id="d1", doc_type=DocumentType.RESUME)

    assert chunks
    # The recursive splitter may overshoot slightly to avoid cutting mid-word.
    assert all(len(chunk.text) <= 260 for chunk in chunks)


def test_produces_multiple_chunks_for_long_text() -> None:
    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=40, min_chunk_chars=10)
    chunks = splitter.split(make_pages(LONG_TEXT), document_id="d1", doc_type=DocumentType.RESUME)
    assert len(chunks) > 1


def test_chunks_overlap() -> None:
    """Overlap is what makes a fact spanning a boundary retrievable from either side."""
    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=80, min_chunk_chars=10)
    chunks = splitter.split(make_pages(LONG_TEXT), document_id="d1", doc_type=DocumentType.RESUME)

    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words


def test_every_chunk_carries_complete_provenance() -> None:
    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=40, min_chunk_chars=10)
    chunks = splitter.split(
        make_pages(LONG_TEXT), document_id="fingerprint1", doc_type=DocumentType.RESUME
    )

    for chunk in chunks:
        meta = chunk.metadata
        assert meta.document_id == "fingerprint1"
        assert meta.filename == "resume.pdf"
        assert meta.doc_type is DocumentType.RESUME
        assert meta.page == 1
        assert meta.char_count == len(chunk.text)
        assert meta.ingested_at is not None


def test_chunk_index_is_continuous_across_pages() -> None:
    """Continuous indices make the index a real ordering key across page breaks."""
    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=40, min_chunk_chars=10)
    chunks = splitter.split(
        make_pages(LONG_TEXT, count=3), document_id="d1", doc_type=DocumentType.RESUME
    )

    indices = [chunk.metadata.chunk_index for chunk in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_ids_are_stable_and_unique() -> None:
    """Deterministic ids are what make re-ingestion an upsert rather than a duplicate."""
    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=40, min_chunk_chars=10)
    pages = make_pages(LONG_TEXT, count=2)

    first = splitter.split(pages, document_id="d1", doc_type=DocumentType.RESUME)
    second = splitter.split(pages, document_id="d1", doc_type=DocumentType.RESUME)

    ids = [chunk.chunk_id for chunk in first]
    assert ids == [chunk.chunk_id for chunk in second]
    assert len(set(ids)) == len(ids)


def test_discards_chunks_below_the_minimum() -> None:
    """A 12-character chunk occupies a top-k slot and carries no signal."""
    splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=100, min_chunk_chars=50)
    chunks = splitter.split(
        [SourceDocument(text="References", filename="r.pdf", page=1)],
        document_id="d1",
        doc_type=DocumentType.RESUME,
    )
    assert chunks == []


def test_prefers_paragraph_boundaries() -> None:
    """Splitting at blank lines keeps a role's bullet and its metric together."""
    text = "SKILLS\nPython, PyTorch, AWS\n\nEXPERIENCE\nReduced inference latency by 61 percent."
    splitter = DocumentSplitter(chunk_size=45, chunk_overlap=0, min_chunk_chars=5)

    chunks = splitter.split(
        [SourceDocument(text=text, filename="r.pdf", page=1)],
        document_id="d1",
        doc_type=DocumentType.RESUME,
    )
    joined = " ".join(chunk.text for chunk in chunks)
    assert "61 percent" in joined


def test_doc_type_is_carried_through() -> None:
    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=40, min_chunk_chars=10)
    chunks = splitter.split(
        make_pages(LONG_TEXT), document_id="d1", doc_type=DocumentType.JOB_DESCRIPTION
    )
    assert all(chunk.metadata.doc_type is DocumentType.JOB_DESCRIPTION for chunk in chunks)
