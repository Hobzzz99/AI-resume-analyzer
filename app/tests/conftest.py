"""Shared fixtures.

Every fixture is built from fakes and a ``tmp_path``, so a test run touches
nothing outside its own temporary directory and needs neither credentials nor a
network. That is SC-009, and it is the reason this suite is worth running on
every save rather than once before a commit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.settings import Settings
from app.prompts.registry import PromptRegistry
from app.rag.cleaner import TextCleaner
from app.rag.embeddings import HashingEmbedder
from app.rag.ingestion import IngestionPipeline
from app.rag.loaders import LoaderRegistry
from app.rag.pipeline import RAGPipeline
from app.rag.prompt_builder import ContextBudget, PromptBuilder
from app.rag.retriever import RetrieverFactory
from app.rag.splitter import DocumentSplitter
from app.rag.vector_store import InMemoryVectorStore
from app.schemas.rag import DocumentType
from app.services.ingestion_service import IngestionService
from app.tests.fakes import SAMPLE_JOB, SAMPLE_RESUME

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "templates"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a temporary data directory.

    ``_env_file=None`` is important: without it pydantic-settings reads the
    developer's real ``.env``, and the suite would pass or fail depending on
    whose machine it ran on.
    """
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        chroma_dir=tmp_path / "chroma",
        manifest_dir=tmp_path / "manifests",
        upload_dir=tmp_path / "uploads",
        prompt_template_dir=PROMPT_DIR,
        groq_api_key="test-key",
        groq_model="test-model",
        chunk_size=300,
        chunk_overlap=60,
        min_chunk_chars=20,
        enable_query_cache=False,
        enable_analysis_cache=True,
        log_json=False,
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    """Deterministic, dependency-free embedder."""
    return HashingEmbedder(dimension=64)


@pytest.fixture
def store() -> InMemoryVectorStore:
    """Exact-search in-memory vector store."""
    return InMemoryVectorStore()


@pytest.fixture
def registry() -> PromptRegistry:
    """The real prompt registry, reading the real templates.

    Deliberately not a fake: the templates are a deliverable, and loading them
    here means a malformed YAML file or a renamed variable fails the suite rather
    than surfacing at runtime.
    """
    return PromptRegistry(PROMPT_DIR)


@pytest.fixture
def splitter(settings: Settings) -> DocumentSplitter:
    """Splitter configured from the test settings."""
    return DocumentSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_chars=settings.min_chunk_chars,
    )


@pytest.fixture
def ingestion_pipeline(
    settings: Settings,
    embedder: HashingEmbedder,
    store: InMemoryVectorStore,
    splitter: DocumentSplitter,
) -> IngestionPipeline:
    """Full ingestion pipeline over fakes."""
    return IngestionPipeline(
        loaders=LoaderRegistry(),
        cleaner=TextCleaner(),
        splitter=splitter,
        embedder=embedder,
        store=store,
        chunking_signature=settings.chunking_signature(),
    )


@pytest.fixture
def ingestion_service(
    settings: Settings, ingestion_pipeline: IngestionPipeline
) -> IngestionService:
    """Ingestion service writing into the temporary directory."""
    return IngestionService(
        pipeline=ingestion_pipeline,
        upload_dir=settings.upload_dir,
        manifest_dir=settings.manifest_dir,
        max_bytes=settings.max_upload_bytes,
        allowed_extensions=settings.allowed_extension_set,
    )


@pytest.fixture
def retriever_factory(
    store: InMemoryVectorStore, embedder: HashingEmbedder
) -> RetrieverFactory:
    """Factory producing every retrieval strategy over the fake store."""
    return RetrieverFactory(store, embedder, mmr_lambda=0.5, fetch_k=10, rrf_k=60)


@pytest.fixture
def prompt_builder() -> PromptBuilder:
    """Prompt builder with a small budget, so budget tests are cheap to write."""
    return PromptBuilder(ContextBudget(max_chunks=12, max_chars=6000))


def build_pipeline(
    retriever_factory: RetrieverFactory,
    prompt_builder: PromptBuilder,
    generator: object,
    *,
    min_chunks: int = 1,
    top_k: int = 3,
    strategy: str = "hybrid",
) -> RAGPipeline:
    """Assemble a pipeline for a test.

    A helper rather than a fixture because most pipeline tests need to vary the
    generator, and a fixture would force every one of them through an override.
    """
    return RAGPipeline(
        retriever_factory=retriever_factory,
        prompt_builder=prompt_builder,
        generator=generator,  # type: ignore[arg-type]
        default_strategy=strategy,
        default_top_k=top_k,
        min_chunks=min_chunks,
    )


@pytest.fixture
def sample_resume_file(tmp_path: Path) -> Path:
    """A resume on disk as a text file."""
    path = tmp_path / "jane_doe_resume.txt"
    path.write_text(SAMPLE_RESUME, encoding="utf-8")
    return path


@pytest.fixture
def sample_job_file(tmp_path: Path) -> Path:
    """A job description on disk as a text file."""
    path = tmp_path / "senior_ai_engineer.txt"
    path.write_text(SAMPLE_JOB, encoding="utf-8")
    return path


@pytest.fixture
def indexed_documents(
    ingestion_pipeline: IngestionPipeline, sample_resume_file: Path, sample_job_file: Path
) -> tuple[str, str]:
    """Ingest the sample resume and job description; return their ids."""
    resume = ingestion_pipeline.ingest_file(sample_resume_file, doc_type=DocumentType.RESUME)
    job = ingestion_pipeline.ingest_file(sample_job_file, doc_type=DocumentType.JOB_DESCRIPTION)
    return resume.manifest.document_id, job.manifest.document_id
