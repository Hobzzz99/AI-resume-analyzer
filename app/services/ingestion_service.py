"""Upload orchestration.

Sits between the HTTP layer and the engine's ingestion pipeline, owning the three
concerns the engine deliberately does not:

* **Validation of untrusted input** — extension, size, and filename safety. These
  are web concerns; an engine embedded in a batch job has no uploads to police.
* **Byte persistence** — the original file is retained so a citation can be traced
  back to the document a user actually submitted.
* **Manifests** — the index of what has been ingested, which Chroma cannot answer
  cheaply.

Size is enforced by streaming the upload in bounded chunks and aborting the moment
the ceiling is crossed. Reading the file into memory and then checking ``len`` —
the obvious implementation — means a 2 GB upload has already been absorbed before
it is rejected, which is a denial-of-service vector rather than a size limit.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from app.rag.ingestion import IngestionPipeline, IngestionResult
from app.schemas.rag import DocumentManifest, DocumentType
from app.utils.exceptions import (
    DocumentNotFoundError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

_UPLOAD_CHUNK = 1024 * 1024  # 1 MiB


class IngestionService:
    """Validates, stores, and ingests submitted documents.

    Args:
        pipeline: The engine ingestion pipeline.
        upload_dir: Where original files are retained.
        manifest_dir: Where document manifests are written.
        max_bytes: Upload size ceiling.
        allowed_extensions: Permitted file extensions, lowercase with a dot.
    """

    def __init__(
        self,
        *,
        pipeline: IngestionPipeline,
        upload_dir: Path,
        manifest_dir: Path,
        max_bytes: int,
        allowed_extensions: frozenset[str],
    ) -> None:
        self._pipeline = pipeline
        self._upload_dir = Path(upload_dir)
        self._manifest_dir = Path(manifest_dir)
        self._max_bytes = max_bytes
        self._allowed = allowed_extensions

    # ------------------------------------------------------------ validate ---

    def validate_filename(self, filename: str) -> str:
        """Check the extension and return it.

        Raises:
            UnsupportedFileTypeError: The extension is missing or not allowed.
        """
        suffix = Path(filename).suffix.lower()
        if not suffix:
            raise UnsupportedFileTypeError(
                f"'{filename}' has no file extension, so its type cannot be determined.",
                details={"filename": filename, "allowed": sorted(self._allowed)},
            )
        if suffix not in self._allowed:
            raise UnsupportedFileTypeError(
                f"Files of type '{suffix}' are not supported. "
                f"Allowed types: {', '.join(sorted(self._allowed))}.",
                details={"filename": filename, "extension": suffix,
                         "allowed": sorted(self._allowed)},
            )
        return suffix

    @staticmethod
    def safe_filename(filename: str) -> str:
        """Reduce an untrusted filename to a safe basename.

        ``Path(...).name`` discards any directory component, which defeats
        ``../../etc/passwd`` style traversal. The remaining character filter
        keeps Windows-illegal characters out of the retained copy.
        """
        base = Path(filename).name
        cleaned = "".join(char for char in base if char.isalnum() or char in "._- ()")
        return cleaned.strip() or "document"

    # -------------------------------------------------------------- ingest ---

    def ingest_upload(
        self,
        *,
        stream: Iterable[bytes],
        filename: str,
        doc_type: DocumentType,
    ) -> DocumentManifest:
        """Persist an uploaded file and ingest it.

        Args:
            stream: Chunks of file content.
            filename: Client-supplied filename, treated as untrusted.
            doc_type: Role of the document.

        Returns:
            The manifest for the ingested document.

        Raises:
            UnsupportedFileTypeError: Disallowed extension.
            FileTooLargeError: Content exceeds the ceiling.
            InvalidDocumentError: The file could not be parsed.
            EmptyDocumentError: No usable text was extracted.
        """
        suffix = self.validate_filename(filename)
        safe_name = self.safe_filename(filename)

        self._upload_dir.mkdir(parents=True, exist_ok=True)
        # Written under a temporary name first: if the size check trips mid-write
        # there is no partially written file sitting in the uploads directory
        # pretending to be a valid document.
        staging = self._upload_dir / f".incoming-{datetime.now(UTC).timestamp()}{suffix}"
        written = 0

        try:
            with staging.open("wb") as handle:
                for chunk in stream:
                    written += len(chunk)
                    if written > self._max_bytes:
                        raise FileTooLargeError(
                            f"'{safe_name}' exceeds the {self._max_bytes // (1024 * 1024)} MB "
                            f"upload limit.",
                            details={"filename": safe_name, "max_bytes": self._max_bytes},
                        )
                    handle.write(chunk)

            if written == 0:
                from app.utils.exceptions import EmptyDocumentError  # noqa: PLC0415

                raise EmptyDocumentError(
                    f"'{safe_name}' is empty.", details={"filename": safe_name}
                )

            result = self._pipeline.ingest_file(staging, doc_type=doc_type)
            self._retain(staging, result, suffix)
        finally:
            staging.unlink(missing_ok=True)

        return self._persist_manifest(result, filename=safe_name)

    def ingest_text(
        self, *, text: str, filename: str, doc_type: DocumentType
    ) -> DocumentManifest:
        """Ingest pasted text.

        Raises:
            EmptyDocumentError: The text is empty or unusable.
        """
        safe_name = self.safe_filename(filename)
        result = self._pipeline.ingest_text(text, filename=safe_name, doc_type=doc_type)
        return self._persist_manifest(result, filename=safe_name)

    def _retain(self, staging: Path, result: IngestionResult, suffix: str) -> None:
        """Keep the original bytes under the document's fingerprint.

        Named by ``document_id`` rather than by the original filename, so two
        different files both called ``resume.pdf`` coexist and re-uploading the
        same content overwrites itself instead of accumulating copies.
        """
        target = self._upload_dir / f"{result.manifest.document_id}{suffix}"
        try:
            staging.replace(target)
        except OSError:
            logger.warning("could not retain uploaded file", extra={"target": str(target)})

    # ----------------------------------------------------------- manifests ---

    def _persist_manifest(self, result: IngestionResult, *, filename: str) -> DocumentManifest:
        """Write the manifest to disk and return it.

        The manifest carries the *submitted* filename rather than the staging
        name, because the filename is what a user recognises in a citation.
        """
        manifest = result.manifest.model_copy(update={"filename": filename})
        self._manifest_dir.mkdir(parents=True, exist_ok=True)
        path = self._manifest_dir / f"{manifest.document_id}.json"
        try:
            path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        except OSError:
            # A manifest is an index, not the source of truth — the chunks are
            # already durably stored, so failing the request here would discard
            # successful work over a bookkeeping problem.
            logger.warning("could not persist manifest", extra={"path": str(path)})
        return manifest

    def get_manifest(self, document_id: str) -> DocumentManifest:
        """Load one manifest.

        Raises:
            DocumentNotFoundError: No manifest for that id.
        """
        path = self._manifest_dir / f"{document_id}.json"
        if not path.is_file():
            raise DocumentNotFoundError(
                f"No document found with id '{document_id}'. It may not have been uploaded yet.",
                details={"document_id": document_id},
            )
        try:
            return DocumentManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise DocumentNotFoundError(
                f"The manifest for '{document_id}' could not be read.",
                details={"document_id": document_id, "reason": str(exc)},
            ) from exc

    def list_manifests(self) -> list[DocumentManifest]:
        """Every manifest on disk, newest first.

        Unreadable manifests are skipped rather than raised: one corrupt file
        must not make the whole document list unavailable.
        """
        manifests: list[DocumentManifest] = []
        if not self._manifest_dir.is_dir():
            return manifests
        for path in self._manifest_dir.glob("*.json"):
            try:
                manifests.append(
                    DocumentManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
                )
            except (OSError, ValueError):
                logger.warning("skipping unreadable manifest", extra={"path": str(path)})
        manifests.sort(key=lambda manifest: manifest.ingested_at, reverse=True)
        return manifests

    def delete_document(self, document_id: str, *, store: object) -> int:
        """Purge a document's chunks, manifest, and retained file.

        Returns:
            The number of chunks removed.

        Raises:
            DocumentNotFoundError: No such document.
        """
        manifest = self.get_manifest(document_id)
        removed = store.delete_document(document_id)  # type: ignore[attr-defined]

        (self._manifest_dir / f"{document_id}.json").unlink(missing_ok=True)
        for retained in self._upload_dir.glob(f"{document_id}.*"):
            retained.unlink(missing_ok=True)

        logger.info(
            "deleted document",
            extra={
                "document_id": document_id,
                "source_file": manifest.filename,
                "chunks": removed,
            },
        )
        return removed
