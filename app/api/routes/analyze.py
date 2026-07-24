"""Analysis endpoints.

``/analyze`` is synchronous and returns the validated envelope. ``/analyze/stream``
returns the same envelope but emits stage events while the work happens.

The streaming design is deliberately *stage* streaming rather than token
streaming. An analysis is only meaningful once it has been parsed and validated
(Principle III), so streaming raw JSON tokens to a browser would invite a client
to render an unvalidated half-object as a result. What a user actually wants to
know during the 5-20 seconds of generation is "what is it doing now" — so the
stream carries retrieval, prompt, generation, and validation events, and exactly
one terminal ``result`` event carrying the validated analysis.

The analysis itself is CPU-bound and synchronous, so it runs in a worker thread
and pushes events onto an asyncio queue via ``call_soon_threadsafe``. Running it
inline would block the event loop and stall every other request, including the
stream it is meant to be feeding.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.dependencies import AnalysisServiceDep, IngestionServiceDep
from app.schemas.analysis import AnalysisResult
from app.schemas.api import AnalyzeRequest
from app.services.analysis_service import AnalysisService
from app.services.ingestion_service import IngestionService
from app.utils.exceptions import AppError
from app.utils.logging import get_logger, get_request_id, set_request_id

router = APIRouter(tags=["analysis"])
logger = get_logger(__name__)

_SENTINEL = object()


def _run_analysis(
    request: AnalyzeRequest,
    analysis_service: AnalysisService,
    ingestion_service: IngestionService,
    on_event: Any = None,
) -> AnalysisResult:
    """Resolve manifests and run the analysis.

    Shared by both routes so the synchronous and streaming paths cannot drift —
    a fix applied to one is applied to both by construction.

    Raises:
        DocumentNotFoundError: Either document id is unknown.
    """
    resume = ingestion_service.get_manifest(request.resume_document_id)
    job = ingestion_service.get_manifest(request.job_document_id)
    return analysis_service.analyze(
        resume=resume,
        job=job,
        job_title=request.job_title,
        template_name=request.prompt_template,
        top_k=request.top_k,
        strategy=request.strategy.value if request.strategy else None,
        use_cache=request.use_cache,
        on_event=on_event,
    )


@router.post("/analyze", response_model=AnalysisResult, summary="Analyze a resume against a job")
def analyze(
    request: AnalyzeRequest,
    analysis_service: AnalysisServiceDep,
    ingestion_service: IngestionServiceDep,
) -> AnalysisResult:
    """Produce a validated, evidence-grounded analysis.

    Declared ``def`` rather than ``async def`` on purpose: the work is CPU-bound
    and synchronous, so FastAPI runs it in a threadpool. Declaring it ``async``
    would block the event loop for the duration of the whole analysis.
    """
    return _run_analysis(request, analysis_service, ingestion_service)


def _sse(event: str, payload: dict[str, Any]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


@router.post("/analyze/stream", summary="Analyze with stage-progress streaming")
async def analyze_stream(
    request: AnalyzeRequest,
    analysis_service: AnalysisServiceDep,
    ingestion_service: IngestionServiceDep,
) -> StreamingResponse:
    """Stream stage progress, then the validated analysis.

    Events: ``stage`` (repeated), then exactly one of ``result`` or ``error``,
    then ``done``. Clients must treat only ``result`` as the analysis.
    """
    request_id = get_request_id() or set_request_id()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()

    def emit(event: str, payload: dict[str, Any]) -> None:
        """Forward a stage event from the worker thread to the event loop."""
        loop.call_soon_threadsafe(queue.put_nowait, ("stage", {"stage": event, **payload}))

    async def produce() -> None:
        """Run the analysis off the event loop and queue its outcome."""
        # The request id is context-local, and a worker thread starts with a
        # fresh context — rebinding it keeps the stage logs correlated with the
        # request that triggered them.
        def work() -> AnalysisResult:
            set_request_id(request_id)
            return _run_analysis(request, analysis_service, ingestion_service, on_event=emit)

        try:
            result = await asyncio.to_thread(work)
            await queue.put(("result", result.model_dump(mode="json")))
        except AppError as exc:
            logger.warning("streamed analysis failed", extra={"code": exc.code})
            await queue.put(("error", exc.to_dict(request_id)))
        except Exception as exc:
            logger.exception("unexpected failure during streamed analysis")
            await queue.put(
                (
                    "error",
                    {
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": f"Unexpected failure: {exc}",
                            "details": {},
                            "request_id": request_id,
                        }
                    },
                )
            )
        finally:
            await queue.put(_SENTINEL)

    async def stream() -> AsyncIterator[str]:
        """Drain the queue as SSE frames."""
        task = asyncio.create_task(produce())
        try:
            yield _sse("stage", {"stage": "accepted", "request_id": request_id})
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                event, payload = item
                yield _sse(event, payload)
            yield _sse("done", {"request_id": request_id})
        finally:
            # A disconnected client cancels this generator; without cancelling the
            # producer the analysis would run to completion writing into a queue
            # nobody reads.
            if not task.done():
                task.cancel()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disables proxy buffering, without which nginx holds every event
            # until the response completes and the stream arrives all at once.
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )
