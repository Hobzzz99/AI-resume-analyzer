"""Follow-up question endpoint.

A second product on the same engine (US5). The route is trivial precisely
*because* the engine is generic — all that differs from ``/analyze`` is the
schema, the template, and the plan.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import ChatServiceDep, IngestionServiceDep
from app.schemas.api import ChatRequest, ChatResponseBody

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponseBody, summary="Ask a grounded follow-up question")
def chat(
    request: ChatRequest,
    service: ChatServiceDep,
    ingestion_service: IngestionServiceDep,
) -> ChatResponseBody:
    """Answer a question using only passages from the named documents.

    Manifests are resolved before retrieval so an unknown id returns a clear
    404. Skipping the check would instead produce ``INSUFFICIENT_CONTEXT``, which
    tells the user their documents were unhelpful when the real problem is that
    they passed a bad id.

    Raises:
        DocumentNotFoundError: One of the document ids is unknown.
        InsufficientContextError: Nothing relevant was retrieved.
        LLMError: The generation provider failed.
    """
    for document_id in request.document_ids:
        ingestion_service.get_manifest(document_id)

    response = service.ask(
        session_id=request.session_id,
        message=request.message,
        document_ids=request.document_ids,
        top_k=request.top_k,
    )
    return ChatResponseBody(
        session_id=response.session_id,
        answer=response.answer.answer,
        citations=response.answer.citations,
        confidence=response.answer.confidence,
        retrieval=response.trace.model_dump(),
        timings=response.timings.as_reported(),
    )
