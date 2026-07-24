"""Output parsing and repair tests.

Constitution Principle III in practice. The extraction cases are drawn from what
models actually emit under load — fenced blocks, prose preambles, trailing
commentary, braces inside string values — because a parser tested only against
clean JSON is a parser tested against the case that never fails.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from app.llm.structured import StructuredGenerator
from app.parsers.json_extract import (
    extract_json,
    find_json_object,
    repair_common_defects,
    strip_code_fences,
)
from app.parsers.structured_parser import ParseFailure, StructuredOutputParser
from app.schemas.analysis import ResumeAnalysis
from app.tests.fakes import ScriptedLLMClient, valid_analysis_json
from app.utils.exceptions import LLMError, OutputValidationError


class Simple(BaseModel):
    """Small schema for parser tests that are not about the analysis shape."""

    name: str
    score: int = Field(ge=0, le=100)


class TestFindJsonObject:
    def test_finds_a_plain_object(self) -> None:
        assert find_json_object('{"a": 1}') == '{"a": 1}'

    def test_handles_nesting(self) -> None:
        text = '{"outer": {"inner": {"deep": 1}}}'
        assert find_json_object(text) == text

    def test_ignores_braces_inside_strings(self) -> None:
        """The exact case a regex-based extractor truncates."""
        text = '{"quote": "use {} for a dict", "n": 1}'
        assert find_json_object(text) == text

    def test_handles_escaped_quotes(self) -> None:
        text = '{"quote": "she said \\"hello\\"", "n": 1}'
        assert find_json_object(text) == text

    def test_stops_at_the_first_complete_object(self) -> None:
        assert find_json_object('{"a": 1} trailing {"b": 2}') == '{"a": 1}'

    def test_returns_none_when_absent(self) -> None:
        assert find_json_object("no braces here") is None

    def test_returns_none_when_unbalanced(self) -> None:
        assert find_json_object('{"a": 1') is None


class TestStripCodeFences:
    def test_extracts_a_fenced_json_block(self) -> None:
        assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_extracts_an_unlabelled_block(self) -> None:
        assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_leaves_unfenced_text_alone(self) -> None:
        assert strip_code_fences('{"a": 1}') == '{"a": 1}'


class TestRepairCommonDefects:
    def test_removes_trailing_commas(self) -> None:
        assert json.loads(repair_common_defects('{"a": 1,}')) == {"a": 1}

    def test_converts_python_literals(self) -> None:
        repaired = json.loads(repair_common_defects('{"a": None, "b": True, "c": False}'))
        assert repaired == {"a": None, "b": True, "c": False}

    def test_leaves_those_words_inside_strings_alone(self) -> None:
        assert '"None found"' in repair_common_defects('{"note": "None found"}')


class TestExtractJson:
    def test_parses_clean_json(self) -> None:
        assert extract_json('{"name": "x", "score": 5}') == {"name": "x", "score": 5}

    def test_parses_fenced_json(self) -> None:
        assert extract_json('```json\n{"name": "x"}\n```') == {"name": "x"}

    def test_parses_json_wrapped_in_prose(self) -> None:
        text = 'Here is the analysis you requested:\n\n{"name": "x"}\n\nHope that helps.'
        assert extract_json(text) == {"name": "x"}

    def test_parses_json_with_a_trailing_comma(self) -> None:
        assert extract_json('{"name": "x", "score": 5,}') == {"name": "x", "score": 5}

    def test_unwraps_a_single_item_list(self) -> None:
        assert extract_json('[{"name": "x"}]') == {"name": "x"}

    def test_rejects_empty_output(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            extract_json("")

    def test_rejects_output_with_no_json(self) -> None:
        with pytest.raises(ValueError, match="No valid JSON"):
            extract_json("I am unable to help with that request.")


class TestStructuredOutputParser:
    def test_validates_a_correct_payload(self) -> None:
        parsed = StructuredOutputParser(Simple).parse('{"name": "x", "score": 50}')
        assert parsed.name == "x"

    def test_rejects_a_range_violation(self) -> None:
        """The failure JSON mode cannot catch — syntactically perfect, semantically wrong."""
        with pytest.raises(ParseFailure) as caught:
            StructuredOutputParser(Simple).parse('{"name": "x", "score": 140}')
        assert "score" in caught.value.errors

    def test_rejects_a_missing_field(self) -> None:
        with pytest.raises(ParseFailure) as caught:
            StructuredOutputParser(Simple).parse('{"name": "x"}')
        assert "score" in caught.value.errors

    def test_failure_carries_the_raw_output_for_repair(self) -> None:
        raw = '{"name": "x", "score": 999}'
        with pytest.raises(ParseFailure) as caught:
            StructuredOutputParser(Simple).parse(raw)
        assert caught.value.raw == raw

    def test_errors_lead_with_the_field_path(self) -> None:
        """The field path is the only part that tells the model where to fix."""
        with pytest.raises(ParseFailure) as caught:
            StructuredOutputParser(Simple).parse('{"name": "x", "score": 140}')
        assert caught.value.errors.startswith("- Field 'score'")

    def test_error_text_is_capped(self) -> None:
        """An unbounded error list pushes the instructions out of the retry's context."""
        parser = StructuredOutputParser(ResumeAnalysis, max_error_chars=100)
        with pytest.raises(ParseFailure) as caught:
            parser.parse("{}")
        assert len(caught.value.errors) <= 160

    def test_try_parse_returns_the_failure_instead_of_raising(self) -> None:
        value, failure = StructuredOutputParser(Simple).try_parse('{"bad": 1}')
        assert value is None
        assert failure is not None


class TestStructuredGeneratorRepairLoop:
    def test_returns_immediately_on_valid_output(self, registry) -> None:
        client = ScriptedLLMClient([valid_analysis_json()])
        generator = StructuredGenerator(
            client=client, registry=registry, max_retries=2, backoff_seconds=0
        )

        result = generator.generate("prompt", ResumeAnalysis)
        assert result.retry_count == 0
        assert client.call_count == 1

    def test_repairs_after_a_validation_failure(self, registry) -> None:
        """The behaviour FR-025 is about: a corrective retry, not a blind resample."""
        client = ScriptedLLMClient(
            [valid_analysis_json(overall_score=250), valid_analysis_json()]
        )
        generator = StructuredGenerator(
            client=client, registry=registry, max_retries=2, backoff_seconds=0
        )

        result = generator.generate("prompt", ResumeAnalysis)
        assert result.retry_count == 1
        assert client.call_count == 2

    def test_the_repair_prompt_contains_the_validation_errors(self, registry) -> None:
        """Feeding back the concrete error is what makes the retry corrective."""
        client = ScriptedLLMClient(
            [valid_analysis_json(overall_score=250), valid_analysis_json()]
        )
        generator = StructuredGenerator(
            client=client, registry=registry, max_retries=2, backoff_seconds=0
        )
        generator.generate("original prompt", ResumeAnalysis)

        repair_prompt = client.prompts[1]
        assert "VALIDATION ERRORS" in repair_prompt
        assert "overall_score" in repair_prompt
        assert "250" in repair_prompt

    def test_recovers_from_unparseable_output(self, registry) -> None:
        client = ScriptedLLMClient(["I cannot do that.", valid_analysis_json()])
        generator = StructuredGenerator(
            client=client, registry=registry, max_retries=2, backoff_seconds=0
        )
        assert generator.generate("prompt", ResumeAnalysis).retry_count == 1

    def test_raises_when_the_budget_is_exhausted(self, registry) -> None:
        client = ScriptedLLMClient([valid_analysis_json(overall_score=250)])
        generator = StructuredGenerator(
            client=client, registry=registry, max_retries=1, backoff_seconds=0
        )

        with pytest.raises(OutputValidationError) as caught:
            generator.generate("prompt", ResumeAnalysis)
        assert caught.value.details["attempts"] == 2
        assert "validation_errors" in caught.value.details

    def test_zero_retries_disables_repair(self, registry) -> None:
        client = ScriptedLLMClient([valid_analysis_json(overall_score=250)])
        generator = StructuredGenerator(
            client=client, registry=registry, max_retries=0, backoff_seconds=0
        )

        with pytest.raises(OutputValidationError):
            generator.generate("prompt", ResumeAnalysis)
        assert client.call_count == 1

    def test_provider_failures_are_not_retried(self, registry) -> None:
        """A repair prompt cannot fix an outage; retrying would waste the budget."""
        from app.tests.fakes import FailingLLMClient

        generator = StructuredGenerator(
            client=FailingLLMClient(LLMError("provider down")),
            registry=registry,
            max_retries=2,
            backoff_seconds=0,
        )
        with pytest.raises(LLMError, match="provider down"):
            generator.generate("prompt", ResumeAnalysis)
