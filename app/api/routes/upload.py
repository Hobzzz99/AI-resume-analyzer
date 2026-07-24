"""Document upload endpoints.

Both routes are thin: validate the shape of the request, delegate to
:class:`~app.services.ingestion_service.IngestionService`, return the manifest.
No business logic lives here, so the ingestion path is identical whether it was
reached over HTTP, from a script, or from a test.

Uploads are read in bounded chunks rather than with ``await file.read()``. The
one-liner materialises the whole upload in memory *before* the size check can
reject it, which turns a size limit into a denial-of-service vector.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter, File, Form, UploadFile, status

from app.api.dependencies import IngestionServiceDep
from app.schemas.api import JobDescriptionRequest
from app.schemas.rag import DocumentManifest, DocumentType
from app.utils.exceptions import InvalidDocumentError
from app.utils.logging import get_logger

router = APIRouter(prefix="/upload", tags=["upload"])
logger = get_logger(__name__)

_READ_CHUNK = 1024 * 1024  # 1 MiB


def _stream(file: UploadFile) -> Iterator[bytes]:
    """Yield the upload in bounded chunks.

    Synchronous reads on ``UploadFile.file`` are used deliberately: the
    downstream ingestion pipeline is CPU-bound and synchronous, so the route is a
    ``def`` and FastAPI runs it in a threadpool. Mixing an async read into a sync
    handler would require an event loop that this call path does not have.
    """
    handle = file.file
    handle.seek(0)
    while True:
        chunk = handle.read(_READ_CHUNK)
        if not chunk:
            break
        yield chunk


@router.post(
    "/resume",
    response_model=DocumentManifest,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a resume",
)
def upload_resume(
    service: IngestionServiceDep,
    file: UploadFile = File(..., description="Resume as PDF, TXT, or MD."),
) -> DocumentManifest:
    """Ingest a resume.

    Returns the document manifest. A repeat upload of identical content returns
    the existing manifest with ``cached: true`` and ``embed_ms: null`` — no
    embedding work is repeated (FR-008, SC-006).
    """
    return service.ingest_upload(
        stream=_stream(file),
        filename=file.filename or "resume",
        doc_type=DocumentType.RESUME,
    )


@router.post(
    "/job",
    response_model=DocumentManifest,
    status_code=status.HTTP_201_CREATED,
    summary="Upload or paste a job description",
)
def upload_job_file(
    service: IngestionServiceDep,
    file: UploadFile = File(..., description="Job description as PDF, TXT, or MD."),
    title: str = Form(default="", description="Role title."),
) -> DocumentManifest:
    """Ingest a job description supplied as a file."""
    filename = file.filename or (f"{title}.txt" if title else "job_description")
    return service.ingest_upload(
        stream=_stream(file),
        filename=filename,
        doc_type=DocumentType.JOB_DESCRIPTION,
    )


@router.post(
    "/job/text",
    response_model=DocumentManifest,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a job description as text",
)
def upload_job_text(
    service: IngestionServiceDep,
    payload: JobDescriptionRequest,
) -> DocumentManifest:
    """Ingest a pasted job description.

    A separate path from the file upload because the two carry different media
    types, and FastAPI cannot express "multipart or JSON" on one operation
    without losing schema generation for both. The *ingestion* path is shared
    from cleaning onward, so behaviour cannot diverge between them.

    Raises:
        InvalidDocumentError: The text is whitespace only.
    """
    if not payload.text.strip():
        raise InvalidDocumentError("The submitted job description is empty.")

    safe_title = payload.title.strip() or "job_description"
    return service.ingest_text(
        text=payload.text,
        filename=f"{safe_title}.txt",
        doc_type=DocumentType.JOB_DESCRIPTION,
    )
