"""Ingestion tests.

The headline assertion is SC-006: re-ingesting identical content performs zero
embedding work. It is verified through ``embed_ms is None`` rather than through
the ``cached`` flag, because a timing of ``None`` proves the stage did not run,
whereas a boolean only proves someone set a boolean.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.ingestion import IngestionPipeline
from app.rag.vector_store import InMemoryVectorStore
from app.schemas.rag import DocumentType
from app.services.ingestion_service import IngestionService
from app.utils.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)

RESUME_TEXT = (
    "Senior machine learning engineer with production experience building retrieval "
    "systems, deploying models on AWS, and leading small engineering teams. "
) * 6


class TestIngestionPipeline:
    def test_indexes_a_document(
        self, ingestion_pipeline: IngestionPipeline, store: InMemoryVectorStore, tmp_path: Path
    ) -> None:
        path = tmp_path / "resume.txt"
        path.write_text(RESUME_TEXT, encoding="utf-8")

        result = ingestion_pipeline.ingest_file(path, doc_type=DocumentType.RESUME)

        assert result.manifest.chunk_count > 0
        assert store.count() == result.manifest.chunk_count
        assert not result.manifest.cached

    def test_records_stage_timings(
        self, ingestion_pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        path = tmp_path / "resume.txt"
        path.write_text(RESUME_TEXT, encoding="utf-8")

        result = ingestion_pipeline.ingest_file(path, doc_type=DocumentType.RESUME)
        timings = result.manifest.timings
        assert timings.load_ms is not None
        assert timings.embed_ms is not None
        assert timings.store_ms is not None

    def test_identical_content_yields_the_same_id(
        self, ingestion_pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """Identity is content, not filename — two 'resume.pdf' files must not collide."""
        first = tmp_path / "a.txt"
        second = tmp_path / "b.txt"
        first.write_text(RESUME_TEXT, encoding="utf-8")
        second.write_text(RESUME_TEXT, encoding="utf-8")

        id_a = ingestion_pipeline.ingest_file(first, doc_type=DocumentType.RESUME)
        id_b = ingestion_pipeline.ingest_file(second, doc_type=DocumentType.RESUME)

        assert id_a.manifest.document_id == id_b.manifest.document_id

    def test_cosmetic_whitespace_does_not_change_identity(
        self, ingestion_pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """The same resume re-exported has different bytes but identical text."""
        first = tmp_path / "a.txt"
        second = tmp_path / "b.txt"
        first.write_text(RESUME_TEXT, encoding="utf-8")
        second.write_text(RESUME_TEXT.replace(" ", "  ") + "\n\n\n", encoding="utf-8")

        assert (
            ingestion_pipeline.ingest_file(first, doc_type=DocumentType.RESUME).manifest.document_id
            == ingestion_pipeline.ingest_file(
                second, doc_type=DocumentType.RESUME
            ).manifest.document_id
        )

    def test_different_content_yields_different_ids(
        self, ingestion_pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        first = tmp_path / "a.txt"
        second = tmp_path / "b.txt"
        first.write_text(RESUME_TEXT, encoding="utf-8")
        second.write_text(RESUME_TEXT.replace("machine learning", "pastry"), encoding="utf-8")

        assert (
            ingestion_pipeline.ingest_file(first, doc_type=DocumentType.RESUME).manifest.document_id
            != ingestion_pipeline.ingest_file(
                second, doc_type=DocumentType.RESUME
            ).manifest.document_id
        )

    def test_reingestion_performs_no_embedding_work(
        self, ingestion_pipeline: IngestionPipeline, store: InMemoryVectorStore, tmp_path: Path
    ) -> None:
        """SC-006, proven by the absence of a timing rather than by a flag."""
        path = tmp_path / "resume.txt"
        path.write_text(RESUME_TEXT, encoding="utf-8")

        first = ingestion_pipeline.ingest_file(path, doc_type=DocumentType.RESUME)
        count_after_first = store.count()

        second = ingestion_pipeline.ingest_file(path, doc_type=DocumentType.RESUME)

        assert second.manifest.cached
        assert second.manifest.timings.embed_ms is None
        assert store.count() == count_after_first
        assert second.manifest.chunk_count == first.manifest.chunk_count

    def test_changing_chunk_settings_invalidates_the_cache(
        self,
        store: InMemoryVectorStore,
        embedder,
        tmp_path: Path,
    ) -> None:
        """Otherwise one collection ends up holding two chunk geometries."""
        from app.rag.cleaner import TextCleaner
        from app.rag.loaders import LoaderRegistry
        from app.rag.splitter import DocumentSplitter

        path = tmp_path / "resume.txt"
        path.write_text(RESUME_TEXT, encoding="utf-8")

        def pipeline_with(signature: str, chunk_size: int) -> IngestionPipeline:
            return IngestionPipeline(
                loaders=LoaderRegistry(),
                cleaner=TextCleaner(),
                splitter=DocumentSplitter(
                    chunk_size=chunk_size, chunk_overlap=40, min_chunk_chars=20
                ),
                embedder=embedder,
                store=store,
                chunking_signature=signature,
            )

        first = pipeline_with("cs=300", 300).ingest_file(path, doc_type=DocumentType.RESUME)
        second = pipeline_with("cs=600", 600).ingest_file(path, doc_type=DocumentType.RESUME)

        assert first.manifest.document_id != second.manifest.document_id
        assert not second.manifest.cached

    def test_pasted_text_takes_the_same_path(
        self, ingestion_pipeline: IngestionPipeline
    ) -> None:
        result = ingestion_pipeline.ingest_text(
            RESUME_TEXT, filename="pasted.txt", doc_type=DocumentType.JOB_DESCRIPTION
        )
        assert result.manifest.chunk_count > 0
        assert result.chunks[0].metadata.doc_type is DocumentType.JOB_DESCRIPTION

    def test_rejects_a_document_with_no_usable_text(
        self, ingestion_pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        path = tmp_path / "scan.txt"
        path.write_text("... 1 | 2 ... 3 4 5 -- 6 7 8 9 10 11 12 13 14", encoding="utf-8")

        with pytest.raises(EmptyDocumentError):
            ingestion_pipeline.ingest_file(path, doc_type=DocumentType.RESUME)

    def test_leaves_no_partial_index_after_rejection(
        self, ingestion_pipeline: IngestionPipeline, store: InMemoryVectorStore, tmp_path: Path
    ) -> None:
        path = tmp_path / "scan.txt"
        path.write_text("... 1 | 2 ... 3 4 5 -- 6 7 8 9 10 11 12 13 14", encoding="utf-8")

        with pytest.raises(EmptyDocumentError):
            ingestion_pipeline.ingest_file(path, doc_type=DocumentType.RESUME)
        assert store.count() == 0

    def test_carries_doc_type_into_metadata(
        self, ingestion_pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        path = tmp_path / "job.txt"
        path.write_text(RESUME_TEXT, encoding="utf-8")

        result = ingestion_pipeline.ingest_file(path, doc_type=DocumentType.JOB_DESCRIPTION)
        assert all(
            chunk.metadata.doc_type is DocumentType.JOB_DESCRIPTION for chunk in result.chunks
        )


class TestIngestionService:
    def test_ingests_an_upload_and_writes_a_manifest(
        self, ingestion_service: IngestionService
    ) -> None:
        manifest = ingestion_service.ingest_upload(
            stream=iter([RESUME_TEXT.encode("utf-8")]),
            filename="jane_doe_resume.txt",
            doc_type=DocumentType.RESUME,
        )

        assert manifest.filename == "jane_doe_resume.txt"
        assert manifest.chunk_count > 0
        assert ingestion_service.get_manifest(manifest.document_id).document_id == (
            manifest.document_id
        )

    def test_rejects_an_unsupported_extension(
        self, ingestion_service: IngestionService
    ) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            ingestion_service.ingest_upload(
                stream=iter([b"data"]), filename="resume.docx", doc_type=DocumentType.RESUME
            )

    def test_rejects_a_file_with_no_extension(
        self, ingestion_service: IngestionService
    ) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            ingestion_service.ingest_upload(
                stream=iter([b"data"]), filename="resume", doc_type=DocumentType.RESUME
            )

    def test_aborts_an_oversize_upload_mid_stream(
        self, ingestion_pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """Rejecting after buffering the whole file is a DoS vector, not a size limit."""
        service = IngestionService(
            pipeline=ingestion_pipeline,
            upload_dir=tmp_path / "uploads",
            manifest_dir=tmp_path / "manifests",
            max_bytes=100,
            allowed_extensions=frozenset({".txt"}),
        )
        chunks_read = 0

        def endless():  # type: ignore[no-untyped-def]
            nonlocal chunks_read
            while True:
                chunks_read += 1
                yield b"x" * 50

        with pytest.raises(FileTooLargeError):
            service.ingest_upload(
                stream=endless(), filename="huge.txt", doc_type=DocumentType.RESUME
            )
        assert chunks_read < 10

    def test_rejects_an_empty_upload(self, ingestion_service: IngestionService) -> None:
        with pytest.raises(EmptyDocumentError):
            ingestion_service.ingest_upload(
                stream=iter([]), filename="empty.txt", doc_type=DocumentType.RESUME
            )

    def test_sanitises_a_traversal_filename(self) -> None:
        assert IngestionService.safe_filename("../../etc/passwd") == "passwd"
        assert IngestionService.safe_filename("C:\\Windows\\evil.txt") == "evil.txt"

    def test_retains_the_original_under_its_fingerprint(
        self, ingestion_service: IngestionService, settings
    ) -> None:
        """Named by id so two files called resume.pdf coexist."""
        manifest = ingestion_service.ingest_upload(
            stream=iter([RESUME_TEXT.encode("utf-8")]),
            filename="resume.txt",
            doc_type=DocumentType.RESUME,
        )
        assert (settings.upload_dir / f"{manifest.document_id}.txt").is_file()

    def test_lists_manifests_newest_first(self, ingestion_service: IngestionService) -> None:
        ingestion_service.ingest_upload(
            stream=iter([RESUME_TEXT.encode("utf-8")]),
            filename="a.txt",
            doc_type=DocumentType.RESUME,
        )
        ingestion_service.ingest_text(
            text=RESUME_TEXT.replace("machine", "software"),
            filename="b.txt",
            doc_type=DocumentType.JOB_DESCRIPTION,
        )
        manifests = ingestion_service.list_manifests()

        assert len(manifests) == 2
        assert manifests[0].ingested_at >= manifests[1].ingested_at

    def test_unknown_document_raises_not_found(
        self, ingestion_service: IngestionService
    ) -> None:
        with pytest.raises(DocumentNotFoundError):
            ingestion_service.get_manifest("does-not-exist")

    def test_delete_purges_manifest_and_chunks(
        self, ingestion_service: IngestionService, store: InMemoryVectorStore
    ) -> None:
        manifest = ingestion_service.ingest_upload(
            stream=iter([RESUME_TEXT.encode("utf-8")]),
            filename="resume.txt",
            doc_type=DocumentType.RESUME,
        )

        removed = ingestion_service.delete_document(manifest.document_id, store=store)

        assert removed == manifest.chunk_count
        assert store.count() == 0
        with pytest.raises(DocumentNotFoundError):
            ingestion_service.get_manifest(manifest.document_id)

    def test_an_unreadable_manifest_does_not_break_the_listing(
        self, ingestion_service: IngestionService, settings
    ) -> None:
        """One corrupt file must not make the whole document list unavailable."""
        ingestion_service.ingest_upload(
            stream=iter([RESUME_TEXT.encode("utf-8")]),
            filename="good.txt",
            doc_type=DocumentType.RESUME,
        )
        (settings.manifest_dir / "corrupt.json").write_text("{ not json", encoding="utf-8")

        assert len(ingestion_service.list_manifests()) == 1
