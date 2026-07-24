"""Document management endpoints.

Exists because a content-fingerprinted index needs a way to answer "what is in
here?" and "remove this". Without deletion, a demo accumulates every document
ever uploaded in one collection — and since retrieval filters on
``document_id``, an operator has no way to tell whether the index is 5 documents
or 500.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.dependencies import IngestionServiceDep, VectorStoreDep
from app.schemas.api import DeleteResponse
from app.schemas.rag import DocumentManifest

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentManifest], summary="List ingested documents")
def list_documents(service: IngestionServiceDep) -> list[DocumentManifest]:
    """Return every document manifest, newest first."""
    return service.list_manifests()


@router.get("/{document_id}", response_model=DocumentManifest, summary="Get one document")
def get_document(document_id: str, service: IngestionServiceDep) -> DocumentManifest:
    """Return one manifest.

    Raises:
        DocumentNotFoundError: No document with that id.
    """
    return service.get_manifest(document_id)


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a document and purge its passages",
)
def delete_document(
    document_id: str, service: IngestionServiceDep, store: VectorStoreDep
) -> DeleteResponse:
    """Remove a document's manifest, retained file, and indexed passages.

    Returns the chunk count rather than a bare 204 so the caller can confirm the
    index was actually purged — a manifest deleted without its chunks would leave
    orphaned passages that still surface in retrieval.

    Raises:
        DocumentNotFoundError: No document with that id.
    """
    removed = service.delete_document(document_id, store=store)
    return DeleteResponse(document_id=document_id, chunks_removed=removed)
