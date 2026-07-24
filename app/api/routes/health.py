"""Health and configuration reporting.

Two endpoints with genuinely different jobs, which is why they are not one:

* ``/health/live`` touches nothing. A liveness probe must answer "is this process
  running", and a probe that queries the vector store will restart a healthy
  container because a dependency blipped.
* ``/health`` inspects every dependency and reports the *resolved* configuration.
  It returns 200 even when degraded, because the endpoint itself is working —
  the ``status`` field carries the verdict. Returning 503 here would take the
  service out of a load balancer for a condition an operator needs to read.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.dependencies import (
    AnalysisCacheDep,
    EmbedderDep,
    PromptRegistryDep,
    SettingsDep,
    VectorStoreDep,
    uptime_seconds,
)
from app.schemas.api import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health/live", summary="Liveness probe")
def live() -> dict[str, str]:
    """Report that the process is running. Touches no dependency."""
    return {"status": "alive"}


@router.get("/health", response_model=HealthResponse, summary="Service health and configuration")
def health(
    settings: SettingsDep,
    embedder: EmbedderDep,
    store: VectorStoreDep,
    registry: PromptRegistryDep,
    cache: AnalysisCacheDep,
) -> HealthResponse:
    """Report component status and the configuration actually in effect.

    This is the answer to "which model produced this analysis?" in production.
    Because ``GROQ_MODEL`` is configuration with no in-code default (FR-021),
    reading the source cannot tell you — only the running process can.
    """
    store_reachable = store.health()
    chunk_count = store.count() if store_reachable else 0

    vector_store: dict[str, Any] = {
        "backend": type(store).__name__,
        "collection": getattr(store, "collection_name", "unknown"),
        "chunk_count": chunk_count,
        "reachable": store_reachable,
        "distance_metric": settings.distance_metric.value,
    }

    llm: dict[str, Any] = {
        "provider": settings.llm_provider.value,
        "model": settings.resolved_llm_model,
        "configured": settings.llm_configured,
        "temperature": settings.llm_temperature,
        "max_retries": settings.llm_max_retries,
        "timeout_seconds": settings.llm_timeout_seconds,
    }

    embeddings: dict[str, Any] = {
        "model": embedder.model_name,
        "dimension": embedder.dimension,
        "loaded": getattr(embedder, "is_loaded", True),
        "device": settings.embedding_device,
    }

    retrieval: dict[str, Any] = {
        "strategy": settings.retrieval_strategy.value,
        "top_k": settings.top_k,
        "fetch_k": settings.fetch_k,
        "reranking": settings.use_reranker,
        "reranker_model": settings.reranker_model if settings.use_reranker else None,
        "max_context_chunks": settings.max_context_chunks,
        "max_context_chars": settings.max_context_chars,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }

    prompts: dict[str, Any] = {
        "directory": str(settings.prompt_template_dir),
        "available": registry.names(),
        "analysis_template": settings.default_analysis_template,
        "chat_template": settings.default_chat_template,
    }

    degraded = not store_reachable or not settings.llm_configured
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=settings.app_version,
        environment=settings.environment.value,
        uptime_seconds=uptime_seconds(),
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store,
        retrieval=retrieval,
        prompts=prompts,
        cache=cache.stats,
    )
