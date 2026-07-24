"""Structured logging with request correlation.

Constitution Principle VI requires every pipeline stage to emit a record carrying
a request-scoped correlation id and a duration. The correlation id is held in a
:class:`~contextvars.ContextVar` rather than threaded through every function
signature — it is ambient request context, not a parameter, and passing it
explicitly through nine pipeline stages would pollute every interface in the
engine for the benefit of the logger alone.

``ContextVar`` is the right primitive here because it is both thread-safe and
task-safe: FastAPI serves concurrent requests on one event loop, and a plain
module global would interleave ids across them.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes present on every LogRecord. Anything outside this set was attached
# by a caller via `extra=` and belongs in the structured payload.
_RESERVED_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
        "pathname", "process", "processName", "relativeCreated", "stack_info",
        "thread", "threadName", "taskName",
    }
)


def safe_extra(fields: dict[str, Any]) -> dict[str, Any]:
    """Rename context keys that collide with built-in ``LogRecord`` attributes.

    ``logging.Logger.makeRecord`` raises ``KeyError`` if ``extra`` contains a key
    the record already defines — ``filename``, ``module``, ``name``, ``args`` and
    friends. That is a landmine for this application specifically, because
    ``filename`` is the single most natural thing to attach to an ingestion log
    line, and the crash only appears once INFO logging is switched on.

    Colliding keys are prefixed rather than dropped: losing the context silently
    would be worse than renaming it.
    """
    return {
        (f"ctx_{key}" if key in _RESERVED_ATTRS else key): value
        for key, value in fields.items()
    }


def new_request_id() -> str:
    """Generate a fresh correlation id."""
    return uuid.uuid4().hex


def set_request_id(request_id: str | None = None) -> str:
    """Bind a correlation id to the current context and return it."""
    resolved = request_id or new_request_id()
    _request_id.set(resolved)
    return resolved


def get_request_id() -> str | None:
    """Return the correlation id bound to the current context, if any."""
    return _request_id.get()


class JSONFormatter(logging.Formatter):
    """Render log records as single-line JSON.

    One line per record with a stable key set is what makes logs queryable in
    any aggregator. Extra fields passed by callers (``stage``, ``duration_ms``,
    ``document_id``) are promoted to top-level keys so they can be filtered on
    directly rather than grepped out of a message string.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render a record as a single JSON line."""
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id

        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Readable console format for local development.

    Same information as :class:`JSONFormatter`, arranged for a human reading a
    terminal rather than a machine reading a log stream.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render a record for a terminal."""
        request_id = get_request_id()
        prefix = f"[{request_id[:8]}] " if request_id else ""
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_ATTRS and not key.startswith("_")
        }
        suffix = f"  {extras}" if extras else ""
        base = (
            f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} "
            f"{prefix}{record.name}: {record.getMessage()}{suffix}"
        )
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Install the root log handler. Call once, at startup.

    Replaces existing handlers rather than adding to them so that repeated calls
    (uvicorn's reloader, test fixtures) do not duplicate every line.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if json_output else HumanFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These libraries log a request line per HTTP call at INFO, which drowns the
    # application's own stage records during an analysis.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Thin wrapper, kept so call sites import one module."""
    return logging.getLogger(name)
