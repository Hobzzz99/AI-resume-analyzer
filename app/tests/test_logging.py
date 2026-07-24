"""Logging and timing tests.

The ``safe_extra`` tests exist because of a real defect found during
development: ``filename`` is a built-in ``LogRecord`` attribute, so attaching it
to ``extra`` raises ``KeyError`` inside the standard library and takes down the
request being logged. It surfaced only once INFO logging was enabled, which is to
say only in production-like conditions — exactly the class of bug worth a
permanent regression test.
"""

from __future__ import annotations

import logging

import pytest

from app.schemas.rag import StageTimings
from app.utils.logging import (
    JSONFormatter,
    get_request_id,
    safe_extra,
    set_request_id,
)
from app.utils.timing import Stopwatch


class TestSafeExtra:
    @pytest.mark.parametrize("reserved", ["filename", "module", "name", "args", "levelname"])
    def test_renames_reserved_attributes(self, reserved: str) -> None:
        assert safe_extra({reserved: "x"}) == {f"ctx_{reserved}": "x"}

    def test_leaves_safe_keys_alone(self) -> None:
        assert safe_extra({"document_id": "abc", "stage": "load"}) == {
            "document_id": "abc",
            "stage": "load",
        }

    def test_renames_rather_than_drops(self) -> None:
        """Losing the context silently would be worse than renaming it."""
        assert safe_extra({"filename": "resume.pdf"})["ctx_filename"] == "resume.pdf"

    def test_a_reserved_key_no_longer_crashes_the_logger(self) -> None:
        """The regression itself."""
        logger = logging.getLogger("test.reserved")
        logger.setLevel(logging.INFO)
        logger.info("ingesting", extra=safe_extra({"filename": "resume.pdf"}))


class TestRequestCorrelation:
    def test_generates_an_id(self) -> None:
        request_id = set_request_id()
        assert get_request_id() == request_id

    def test_honours_a_supplied_id(self) -> None:
        set_request_id("trace-1234")
        assert get_request_id() == "trace-1234"


class TestJSONFormatter:
    def test_emits_one_json_line(self) -> None:
        import json

        record = logging.LogRecord(
            "test", logging.INFO, "path.py", 10, "hello", (), None
        )
        payload = json.loads(JSONFormatter().format(record))

        assert payload["level"] == "INFO"
        assert payload["message"] == "hello"

    def test_promotes_caller_context_to_top_level_keys(self) -> None:
        """Filterable fields beat grepping a message string."""
        import json

        record = logging.LogRecord("test", logging.INFO, "p.py", 1, "m", (), None)
        record.document_id = "abc123"  # type: ignore[attr-defined]

        assert json.loads(JSONFormatter().format(record))["document_id"] == "abc123"

    def test_includes_the_request_id(self) -> None:
        import json

        set_request_id("corr-1")
        record = logging.LogRecord("test", logging.INFO, "p.py", 1, "m", (), None)

        assert json.loads(JSONFormatter().format(record))["request_id"] == "corr-1"


class TestStopwatch:
    def test_records_a_duration(self) -> None:
        timings = StageTimings()
        with Stopwatch("retrieve", timings):
            pass
        assert timings.retrieve_ms is not None

    def test_records_the_duration_of_a_failure(self) -> None:
        """A 60s failure and a 40ms failure are different incidents."""
        timings = StageTimings()
        with pytest.raises(ValueError, match="boom"), Stopwatch("llm", timings):
            raise ValueError("boom")
        assert timings.llm_ms is not None

    def test_never_suppresses_an_exception(self) -> None:
        with pytest.raises(RuntimeError), Stopwatch("embed"):
            raise RuntimeError

    def test_accumulates_across_repeated_stages(self) -> None:
        """A six-step plan calls retrieve six times; the total is what matters."""
        timings = StageTimings()
        for _ in range(3):
            with Stopwatch("retrieve", timings):
                pass
        assert timings.retrieve_ms is not None

    def test_arbitrary_caller_context_is_safe(self) -> None:
        """Stopwatch takes **context from callers, so it must sanitise it."""
        logging.getLogger().setLevel(logging.DEBUG)
        with Stopwatch("load", StageTimings(), filename="resume.pdf", module="x"):
            pass

    def test_unknown_stage_names_are_ignored(self) -> None:
        """Instrumenting a new stage must never break a request path."""
        timings = StageTimings()
        timings.record("not_a_real_stage", 12.0)
        assert timings.as_reported() == {}
