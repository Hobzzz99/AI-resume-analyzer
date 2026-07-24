"""Typed application configuration.

Constitution Principle VII: every tunable value in this system is read from here,
and nowhere else. Modules receive a ``Settings`` instance by injection rather than
importing a module-level singleton, which is what allows tests to run against a
temporary data directory without touching the developer's real index.

The one deliberate exception is :func:`get_settings`, a process-level cached
accessor used by the FastAPI composition root and by scripts. It is a factory,
not a constant, and tests override it via ``app.dependency_overrides``.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class RetrievalStrategy(StrEnum):
    """Supported retrieval strategies.

    Defined here rather than in ``app/schemas`` because it is a configuration
    value first: the settings object must be able to validate it at startup,
    before any schema module is imported.
    """

    SIMILARITY = "similarity"
    MMR = "mmr"
    BM25 = "bm25"
    HYBRID = "hybrid"


class DistanceMetric(StrEnum):
    """Vector distance metrics supported by the store backend."""

    COSINE = "cosine"
    L2 = "l2"
    IP = "ip"


class LLMProvider(StrEnum):
    """Generation providers behind the ``LLMClient`` protocol.

    Both satisfy the same interface, so selecting one is a configuration change
    with no effect on the pipeline, the repair loop, or the API.
    """

    GROQ = "groq"
    GEMINI = "gemini"


class Settings(BaseSettings):
    """All runtime configuration, sourced from the environment and ``.env``.

    Grouped by concern to keep the surface navigable. Every field carries a
    description because these descriptions are what ``/health`` and the README
    are generated against — configuration that is not documented at the point of
    definition is configuration nobody sets correctly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------- app ---
    app_name: str = Field(default="AI Resume Analyzer", description="Service display name.")
    app_version: str = Field(default="1.0.0", description="Semantic version of the service.")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(
        default=True, description="Enables verbose errors and reload-friendly behaviour."
    )
    api_prefix: str = Field(default="/api/v1", description="Mount point for all routes.")
    cors_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated allowed origins. '*' permitted in development only.",
    )

    # ------------------------------------------------------------ logging ---
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=True, description="Emit structured JSON logs (Principle VI).")

    # -------------------------------------------------------------- paths ---
    data_dir: Path = Field(default=Path("./data"))
    chroma_dir: Path = Field(default=Path("./data/chroma"))
    manifest_dir: Path = Field(default=Path("./data/manifests"))
    upload_dir: Path = Field(default=Path("./data/uploads"))

    # ------------------------------------------------------------ uploads ---
    max_upload_mb: Annotated[int, Field(ge=1, le=100)] = 10
    allowed_extensions: str = Field(default=".pdf,.txt,.md")

    # ----------------------------------------------------------- chunking ---
    chunk_size: Annotated[int, Field(ge=100, le=8000)] = 1000
    chunk_overlap: Annotated[int, Field(ge=0, le=4000)] = 200
    min_chunk_chars: Annotated[int, Field(ge=0)] = 50

    # --------------------------------------------------------- embeddings ---
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu")
    embedding_batch_size: Annotated[int, Field(ge=1, le=512)] = 32
    embedding_normalize: bool = Field(
        default=True,
        description="Normalise vectors so cosine similarity reduces to a dot product.",
    )

    # -------------------------------------------------------- vector store ---
    collection_name: str = Field(default="resume_rag_chunks")
    distance_metric: DistanceMetric = Field(default=DistanceMetric.COSINE)

    # ---------------------------------------------------------- retrieval ---
    retrieval_strategy: RetrievalStrategy = Field(default=RetrievalStrategy.HYBRID)
    top_k: Annotated[int, Field(ge=1, le=50)] = 5
    fetch_k: Annotated[int, Field(ge=1, le=200)] = 20
    mmr_lambda: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    rrf_k: Annotated[int, Field(ge=1)] = 60
    min_retrieved_chunks: Annotated[int, Field(ge=0)] = 2
    max_context_chunks: Annotated[int, Field(ge=1, le=200)] = 24
    max_context_chars: Annotated[int, Field(ge=500)] = 24_000

    # ---------------------------------------------------------- reranking ---
    use_reranker: bool = Field(default=False)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_multiplier: Annotated[int, Field(ge=1, le=20)] = 4

    # ---------------------------------------------------------------- llm ---
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GROQ,
        description="Which generation provider to use. Both satisfy the same interface.",
    )
    groq_api_key: str = Field(default="", description="Groq API key. Absent in offline test runs.")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description=(
            "Groq model id. FR-021 forbids hardcoding this in application logic; the value "
            "lives here and is reported by /health so the running model is always knowable."
        ),
    )
    gemini_api_key: str = Field(
        default="", description="Google AI Studio API key. https://aistudio.google.com/apikey"
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description=(
            "Gemini model id. Never hardcoded in application logic (FR-021); reported by /health. "
            "gemini-2.0-flash has a large free tier and native JSON mode."
        ),
    )
    llm_temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.1
    llm_max_tokens: Annotated[int, Field(ge=256, le=32_000)] = 4096
    llm_timeout_seconds: Annotated[float, Field(ge=1.0)] = 60.0
    llm_max_retries: Annotated[int, Field(ge=0, le=5)] = 2
    llm_retry_backoff_seconds: Annotated[float, Field(ge=0.0)] = 1.0
    llm_rate_limit_retries: Annotated[int, Field(ge=0, le=10)] = 3
    llm_rate_limit_wait_seconds: Annotated[float, Field(ge=0.0)] = 20.0

    # ------------------------------------------------------------ prompts ---
    prompt_template_dir: Path = Field(default=Path("./app/prompts/templates"))
    default_analysis_template: str = Field(default="resume_analysis_v1")
    default_chat_template: str = Field(default="chat_qa_v1")
    repair_template: str = Field(default="repair_v1")

    # ------------------------------------------------------------ caching ---
    enable_analysis_cache: bool = Field(default=True)
    analysis_cache_size: Annotated[int, Field(ge=1)] = 128
    enable_query_cache: bool = Field(default=True)
    query_cache_size: Annotated[int, Field(ge=1)] = 512
    query_cache_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.97

    # --------------------------------------------------------------- chat ---
    chat_memory_turns: Annotated[int, Field(ge=0, le=50)] = 6
    chat_session_limit: Annotated[int, Field(ge=1)] = 64

    # -------------------------------------------------------- validation ----
    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_below_size(cls, value: int, info) -> int:  # type: ignore[no-untyped-def]
        """Reject an overlap that meets or exceeds the chunk size.

        An overlap >= chunk size makes the splitter emit chunks that never
        advance, producing either an infinite loop or a degenerate index. Failing
        at startup is far kinder than discovering it during ingestion.
        """
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            msg = f"chunk_overlap ({value}) must be smaller than chunk_size ({chunk_size})"
            raise ValueError(msg)
        return value

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            msg = f"log_level must be one of {sorted(allowed)}, got {value!r}"
            raise ValueError(msg)
        return upper

    # --------------------------------------------------------- accessors ----
    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list.

        Kept as a comma-separated string on the field because pydantic-settings
        parses ``list[str]`` fields as JSON, which makes the ``.env`` line
        unreadable and easy to get wrong.
        """
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowed_extension_set(self) -> frozenset[str]:
        """Permitted upload extensions, normalised to lowercase with a leading dot."""
        return frozenset(
            ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
            for ext in self.allowed_extensions.split(",")
            if ext.strip()
        )

    @property
    def max_upload_bytes(self) -> int:
        """Upload ceiling in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def llm_configured(self) -> bool:
        """Whether the *selected* provider has a credential present."""
        if self.llm_provider is LLMProvider.GEMINI:
            return bool(self.gemini_api_key.strip())
        return bool(self.groq_api_key.strip())

    @property
    def resolved_llm_model(self) -> str:
        """Model id of the selected provider, for reporting through /health."""
        if self.llm_provider is LLMProvider.GEMINI:
            return self.gemini_model
        return self.groq_model

    def chunking_signature(self) -> str:
        """Identity of the current chunking configuration.

        Folded into the document fingerprint so that changing ``chunk_size`` or
        ``chunk_overlap`` produces new document ids rather than silently mixing
        two chunk geometries in one collection (research.md R7).
        """
        return f"cs={self.chunk_size}|co={self.chunk_overlap}|mc={self.min_chunk_chars}"

    def ensure_directories(self) -> None:
        """Create the runtime directories. Called once at startup, never at import."""
        for directory in (self.data_dir, self.chroma_dir, self.manifest_dir, self.upload_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Cached because reading ``.env`` and validating on every request would be
    wasteful, and because a single instance makes the resolved configuration
    reportable through ``/health``.
    """
    return Settings()
