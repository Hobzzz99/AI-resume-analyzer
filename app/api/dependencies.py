"""Composition root.

Constitution Principle V in one file: this is the *only* place in the application
where concrete classes are chosen and wired together. Every other module receives
its collaborators and never constructs them, which is what makes the object graph
swappable and the test suite offline-capable — a test overrides one provider here
and the entire stack below it becomes a fake.

Each provider is ``@lru_cache``'d, so the embedding model, the Chroma client, and
the prompt registry are process singletons. Rebuilding an embedder per request
would reload 90 MB of weights per request.

Nothing here loads a model or opens a connection at import time. The providers
construct objects whose expensive work is itself lazy, so importing this module
is free and startup stays fast.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.config.settings import LLMProvider, Settings, get_settings
from app.llm.gemini_client import GeminiClient
from app.llm.groq_client import GroqClient
from app.llm.structured import StructuredGenerator
from app.prompts.registry import PromptRegistry
from app.rag.base import Embedder, LLMClient, VectorStore
from app.rag.cleaner import TextCleaner
from app.rag.embeddings import SentenceTransformerEmbedder
from app.rag.ingestion import IngestionPipeline
from app.rag.loaders import LoaderRegistry
from app.rag.pipeline import RAGPipeline
from app.rag.prompt_builder import ContextBudget, PromptBuilder
from app.rag.retriever import CrossEncoderReranker, RetrieverFactory
from app.rag.splitter import DocumentSplitter
from app.rag.vector_store import ChromaVectorStore
from app.services.analysis_service import AnalysisService
from app.services.cache import AnalysisCache, SemanticQueryCache
from app.services.chat_service import ChatService, ConversationMemory
from app.services.ingestion_service import IngestionService
from app.utils.logging import get_logger

logger = get_logger(__name__)

START_TIME = time.monotonic()


def uptime_seconds() -> float:
    """Seconds since the process started."""
    return round(time.monotonic() - START_TIME, 2)


# --------------------------------------------------------------- engine ---


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """The embedding model. One instance per process; weights load on first use."""
    settings = get_settings()
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        normalize=settings.embedding_normalize,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """The persistent vector store."""
    settings = get_settings()
    return ChromaVectorStore(
        settings.chroma_dir,
        collection_name=settings.collection_name,
        distance_metric=settings.distance_metric.value,
    )


@lru_cache(maxsize=1)
def get_prompt_registry() -> PromptRegistry:
    """The prompt template registry, eagerly loaded.

    Eager so a malformed template is a startup failure rather than a failure in
    the middle of a user's analysis.
    """
    return PromptRegistry(get_settings().prompt_template_dir, eager=True)


@lru_cache(maxsize=1)
def get_retriever_factory() -> RetrieverFactory:
    """Factory producing retrievers for a named strategy."""
    settings = get_settings()
    reranker = (
        CrossEncoderReranker(settings.reranker_model, device=settings.embedding_device)
        if settings.use_reranker
        else None
    )
    return RetrieverFactory(
        get_vector_store(),
        get_embedder(),
        mmr_lambda=settings.mmr_lambda,
        fetch_k=settings.fetch_k,
        rrf_k=settings.rrf_k,
        reranker=reranker,
        rerank_multiplier=settings.rerank_multiplier,
    )


@lru_cache(maxsize=1)
def get_query_cache() -> SemanticQueryCache:
    """Semantic cache in front of sub-query retrieval."""
    settings = get_settings()
    return SemanticQueryCache(
        get_embedder(),
        max_size=settings.query_cache_size,
        threshold=settings.query_cache_threshold,
        enabled=settings.enable_query_cache,
    )


@lru_cache(maxsize=1)
def get_analysis_cache() -> AnalysisCache:
    """Exact-match cache for completed analyses."""
    settings = get_settings()
    return AnalysisCache(
        max_size=settings.analysis_cache_size, enabled=settings.enable_analysis_cache
    )


# ------------------------------------------------------------------ llm ---


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """The generation provider client for the configured provider.

    The selection lives here, in the composition root, and nowhere else — the
    structured generator, pipeline, and API layer receive an ``LLMClient`` and
    never learn which provider is behind it. Switching providers is a one-line
    ``.env`` change (``LLM_PROVIDER``), which is Principle V paying off.

    Raises:
        ConfigurationError: The selected provider's key/model is unset. Raised
            lazily, on first use, so the service still starts and ``/health`` can
            report the misconfiguration rather than the process dying at import.
    """
    settings = get_settings()
    if settings.llm_provider is LLMProvider.GEMINI:
        return GeminiClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    return GroqClient(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )


@lru_cache(maxsize=1)
def get_generator() -> StructuredGenerator:
    """Structured generation with the repair-retry loop."""
    settings = get_settings()
    return StructuredGenerator(
        client=get_llm_client(),
        registry=get_prompt_registry(),
        max_retries=settings.llm_max_retries,
        backoff_seconds=settings.llm_retry_backoff_seconds,
        repair_template=settings.repair_template,
        rate_limit_retries=settings.llm_rate_limit_retries,
        rate_limit_wait_seconds=settings.llm_rate_limit_wait_seconds,
    )


# -------------------------------------------------------------- pipeline ---


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """The generic RAG pipeline, shared by every domain service.

    One instance serves both the analyzer and the chat service. That sharing is
    the practical demonstration of the reuse thesis: two products, different
    schemas, different prompts, identical engine object.
    """
    settings = get_settings()
    return RAGPipeline(
        retriever_factory=get_retriever_factory(),
        prompt_builder=PromptBuilder(
            ContextBudget(
                max_chunks=settings.max_context_chunks, max_chars=settings.max_context_chars
            )
        ),
        generator=get_generator(),
        default_strategy=settings.retrieval_strategy.value,
        default_top_k=settings.top_k,
        min_chunks=settings.min_retrieved_chunks,
        query_cache=get_query_cache(),
    )


@lru_cache(maxsize=1)
def get_ingestion_pipeline() -> IngestionPipeline:
    """The document ingestion pipeline."""
    settings = get_settings()
    return IngestionPipeline(
        loaders=LoaderRegistry(),
        cleaner=TextCleaner(),
        splitter=DocumentSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_chars=settings.min_chunk_chars,
        ),
        embedder=get_embedder(),
        store=get_vector_store(),
        chunking_signature=settings.chunking_signature(),
    )


# -------------------------------------------------------------- services ---


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    """Upload orchestration."""
    settings = get_settings()
    return IngestionService(
        pipeline=get_ingestion_pipeline(),
        upload_dir=settings.upload_dir,
        manifest_dir=settings.manifest_dir,
        max_bytes=settings.max_upload_bytes,
        allowed_extensions=settings.allowed_extension_set,
    )


@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    """Resume analysis."""
    settings = get_settings()
    return AnalysisService(
        pipeline=get_rag_pipeline(),
        registry=get_prompt_registry(),
        cache=get_analysis_cache(),
        template_name=settings.default_analysis_template,
    )


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """Grounded follow-up Q&A."""
    settings = get_settings()
    return ChatService(
        pipeline=get_rag_pipeline(),
        registry=get_prompt_registry(),
        memory=ConversationMemory(
            max_turns=settings.chat_memory_turns, max_sessions=settings.chat_session_limit
        ),
        template_name=settings.default_chat_template,
    )


def reset_dependencies() -> None:
    """Clear every cached provider.

    Used by tests between cases, and after a configuration change in development,
    so a stale singleton built from previous settings cannot survive into the
    next scenario.
    """
    for provider in (
        get_embedder, get_vector_store, get_prompt_registry, get_retriever_factory,
        get_query_cache, get_analysis_cache, get_llm_client, get_generator,
        get_rag_pipeline, get_ingestion_pipeline, get_ingestion_service,
        get_analysis_service, get_chat_service, get_settings,
    ):
        provider.cache_clear()


# FastAPI annotations. Routes depend on these aliases rather than calling the
# providers directly, which is what makes `app.dependency_overrides[...]` able to
# substitute a fake for any node in the graph.
SettingsDep = Annotated[Settings, Depends(get_settings)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
PromptRegistryDep = Annotated[PromptRegistry, Depends(get_prompt_registry)]
AnalysisCacheDep = Annotated[AnalysisCache, Depends(get_analysis_cache)]
