"""Application factory, middleware, and error handling.

Three things live here and nowhere else:

* **The app factory.** ``create_app()`` rather than a module-level ``app =
  FastAPI()`` so tests can build an isolated instance with overridden settings
  instead of mutating a global.
* **Request correlation.** One middleware binds a request id into the logging
  context and returns it as ``X-Request-ID``, so every log line from a request —
  including those emitted deep inside the engine — is joinable.
* **Error translation.** A single handler maps the ``AppError`` hierarchy onto
  HTTP status codes. Because each exception class carries its own ``code`` and
  ``http_status``, adding a failure mode never requires touching this file, and
  no route contains a ``try/except`` that converts an exception into a response.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.dependencies import get_prompt_registry, get_vector_store
from app.api.routes import analyze, chat, documents, health, upload
from app.config.settings import Settings, get_settings
from app.utils.exceptions import AppError
from app.utils.logging import configure_logging, get_logger, get_request_id, set_request_id

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001 - FastAPI's signature
    """Prepare the runtime and warm the components worth warming.

    Directories and prompt templates are handled eagerly, so a bad path or a
    malformed template fails at startup rather than inside a user's request.

    The embedding model is *not* warmed here by default. Loading 90 MB of weights
    would add seconds to every start, including the reload cycle in development,
    and the first request absorbs it anyway. Warming is opt-in via
    ``EMBEDDING_WARMUP`` for a production deployment where the first user's
    latency matters more than boot time.
    """
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    settings.ensure_directories()

    registry = get_prompt_registry()
    logger.info(
        "service starting",
        extra={
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment.value,
            "provider": settings.llm_provider.value,
            "model": settings.resolved_llm_model,
            "llm_configured": settings.llm_configured,
            "embedding_model": settings.embedding_model,
            "retrieval_strategy": settings.retrieval_strategy.value,
            "prompt_templates": registry.names(),
        },
    )
    if not settings.llm_configured:
        logger.warning(
            "%s API key is not set; analysis requests will fail with CONFIGURATION_ERROR",
            settings.llm_provider.value,
        )

    store = get_vector_store()
    if store.health():
        logger.info("vector store ready", extra={"chunks": store.count()})
    else:
        logger.error("vector store unreachable at startup")

    yield

    logger.info("service stopping")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Override configuration. Used by tests to point the app at a
            temporary data directory.
    """
    resolved = settings or get_settings()

    app = FastAPI(
        title=resolved.app_name,
        version=resolved.app_version,
        description=(
            "A modular Retrieval-Augmented Generation engine, demonstrated by an AI Resume "
            "Analyzer. Documents are chunked, embedded once, and retrieved by facet; the model "
            "only ever sees retrieved passages, and every response is schema-validated."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # The browser cannot read a custom header on a cross-origin response
        # unless it is exposed — without this the client loses the request id it
        # needs to report a problem.
        expose_headers=["X-Request-ID"],
    )

    _register_middleware(app)
    _register_error_handlers(app)

    for module in (health, upload, analyze, chat, documents):
        app.include_router(module.router, prefix=resolved.api_prefix)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        """Point a browser at the interesting endpoints."""
        return {
            "service": resolved.app_name,
            "version": resolved.app_version,
            "docs": "/docs",
            "health": f"{resolved.api_prefix}/health",
        }

    return app


def _register_middleware(app: FastAPI) -> None:
    """Install request correlation and access logging."""

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Any]]
    ) -> Any:
        """Bind a correlation id, time the request, and log the outcome.

        An inbound ``X-Request-ID`` is honoured so a trace can span a proxy or a
        calling service; otherwise one is generated.
        """
        request_id = set_request_id(request.headers.get("X-Request-ID"))
        started = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response


def _register_error_handlers(app: FastAPI) -> None:
    """Install the uniform error envelope for every failure class."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        """Render a known application failure.

        Client errors log at warning, server and upstream errors at error — so a
        user uploading a bad PDF does not page anyone, while a provider outage
        does.
        """
        log = logger.warning if exc.http_status < 500 else logger.error
        log(
            "request failed: %s",
            exc.code,
            extra={"code": exc.code, "status_code": exc.http_status, "path": request.url.path},
        )
        return JSONResponse(
            status_code=exc.http_status, content=exc.to_dict(get_request_id())
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,  # noqa: ARG001 - required by the handler signature
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Render FastAPI's request validation failures in the same envelope.

        Without this, a malformed request body returns FastAPI's ``{"detail":
        [...]}`` shape while every other failure returns ``{"error": {...}}``,
        forcing clients to implement two error paths.

        The errors are rebuilt field by field rather than passed through: a
        failing pydantic ``model_validator`` puts the raw ``ValueError`` into
        ``ctx``, which is not JSON-serialisable and makes the error handler
        itself raise — turning a clean 422 into an opaque 500.
        """
        errors = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())),
                "message": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "REQUEST_VALIDATION_ERROR",
                    "message": "The request body or parameters are invalid.",
                    "details": {"errors": errors},
                    "request_id": get_request_id(),
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(
        request: Request,
        exc: Exception,  # noqa: ARG001 - logged via logger.exception, never returned
    ) -> JSONResponse:
        """Catch-all for genuinely unexpected failures.

        The exception text is logged with a traceback but deliberately not
        returned: an unhandled exception's message can carry a filesystem path,
        a query, or a credential fragment. The client gets the request id, which
        is enough to correlate with the log entry that has the detail.
        """
        logger.exception("unhandled exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred. Please retry.",
                    "details": {},
                    "request_id": get_request_id(),
                }
            },
        )


app = create_app()
