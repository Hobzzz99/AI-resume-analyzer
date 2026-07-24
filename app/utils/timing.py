"""Stage timing instrumentation.

Constitution Principle VI requires a duration for every pipeline stage. A context
manager is used rather than a decorator because several stages are *parts* of a
function rather than whole functions — the embed and store phases of ingestion
live in one method, and decorating that method would report one number where two
are needed.

Uses :func:`time.perf_counter`, not :func:`time.time`: the latter is wall clock
and can move backwards across an NTP correction, producing negative durations in
exactly the logs you most want to trust.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Any

from app.schemas.rag import StageTimings
from app.utils.logging import get_logger, safe_extra

logger = get_logger(__name__)


class Stopwatch:
    """Times a block and records it against a :class:`StageTimings`.

    Records the duration on exit whether the block succeeded or raised. Timing a
    failure is not optional — "the LLM call failed after 60 seconds" and "the LLM
    call failed after 40 milliseconds" are different incidents with different
    causes, and a stopwatch that only reports success cannot tell them apart.

    Example:
        >>> timings = StageTimings()
        >>> with Stopwatch("retrieve", timings):
        ...     chunks = retriever.retrieve("query")
        >>> timings.retrieve_ms is not None
        True
    """

    __slots__ = ("_context", "_start", "_timings", "elapsed_ms", "stage")

    def __init__(
        self,
        stage: str,
        timings: StageTimings | None = None,
        **context: Any,
    ) -> None:
        self.stage = stage
        self._timings = timings
        self._context = context
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if self._timings is not None:
            self._timings.record(self.stage, self.elapsed_ms)

        level = logger.warning if exc_type is not None else logger.debug
        level(
            "stage %s %s",
            self.stage,
            "failed" if exc_type else "completed",
            # Sanitised because callers pass arbitrary context here, and a key
            # like `filename` would otherwise crash the request being timed.
            extra=safe_extra(
                {
                    "stage": self.stage,
                    "duration_ms": round(self.elapsed_ms, 3),
                    "failed": exc_type is not None,
                    **self._context,
                }
            ),
        )
        return False  # never suppress; timing is observation, not handling


@contextmanager
def timed(stage: str, timings: StageTimings | None = None, **context: Any) -> Iterator[Stopwatch]:
    """Function-style alias for :class:`Stopwatch`.

    Exists because ``with timed("llm", t) as watch:`` reads better at call sites
    that need the elapsed value afterwards.
    """
    watch = Stopwatch(stage, timings, **context)
    with watch:
        yield watch
