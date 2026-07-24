"""Schema validation of model output, with repair instructions on failure.

Constitution Principle III: nothing crosses a module boundary until it has been
validated against a Pydantic model. This parser is where that happens, and it is
deliberately split from the retry *orchestration* in :mod:`app.llm.structured`:

* the parser answers "is this valid, and if not, what exactly is wrong?" — a pure
  function of text and schema, trivially testable without a provider;
* the generator decides what to do about it — retries, backoff, budget.

Keeping them apart is what lets the entire validation surface be tested offline
(SC-009), and it means a different retry policy is a swap of the orchestrator
rather than an edit inside the validator.
"""

from __future__ import annotations

from typing import Any, Generic

from pydantic import BaseModel, ValidationError

from app.parsers.json_extract import extract_json
from app.rag.base import SchemaT
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ParseFailure(Exception):  # noqa: N818 - not an Error: it is a recoverable loop signal
    """A parse or validation failure, carrying what a repair prompt needs.

    Not an :class:`~app.utils.exceptions.AppError`: this is an *expected*,
    recoverable step in the repair loop, not a request-terminating condition. It
    becomes an ``OutputValidationError`` only when the budget is exhausted.
    """

    def __init__(self, message: str, *, raw: str, errors: str) -> None:
        super().__init__(message)
        self.message = message
        self.raw = raw
        self.errors = errors


class StructuredOutputParser(Generic[SchemaT]):
    """Parses and validates model output against a Pydantic model.

    Args:
        schema: The model every response must satisfy.
        max_error_chars: Ceiling on the rendered error text fed back to the
            model. An unbounded error list from a 30-item array can be longer
            than the response itself, which pushes the actual instructions out of
            the model's attention on the retry — the opposite of the intent.
    """

    def __init__(self, schema: type[SchemaT], *, max_error_chars: int = 2000) -> None:
        self._schema = schema
        self._max_error_chars = max_error_chars

    @property
    def schema(self) -> type[SchemaT]:
        """The target model."""
        return self._schema

    def parse(self, text: str) -> SchemaT:
        """Extract JSON from ``text`` and validate it.

        Raises:
            ParseFailure: The text held no JSON object, or the object did not
                satisfy the schema. Carries the raw output and a rendered error
                description for the repair prompt.
        """
        try:
            payload = extract_json(text)
        except ValueError as exc:
            raise ParseFailure(
                str(exc),
                raw=text,
                errors=(
                    f"The response was not valid JSON: {exc}\n"
                    f"Return a single JSON object with no surrounding text or markdown."
                ),
            ) from exc

        try:
            return self._schema.model_validate(payload)
        except ValidationError as exc:
            rendered = self.format_errors(exc)
            logger.warning(
                "structured output failed validation",
                extra={"schema": self._schema.__name__, "error_count": exc.error_count()},
            )
            raise ParseFailure(
                f"Response did not satisfy {self._schema.__name__}: {exc.error_count()} error(s).",
                raw=text,
                errors=rendered,
            ) from exc

    def try_parse(self, text: str) -> tuple[SchemaT | None, ParseFailure | None]:
        """Non-raising variant, for callers driving their own loop."""
        try:
            return self.parse(text), None
        except ParseFailure as failure:
            return None, failure

    def format_errors(self, error: ValidationError) -> str:
        """Render validation errors as instructions the model can act on.

        Pydantic's default rendering leads with the type name and the input value
        and buries the field path. Reordering it to
        ``Field 'x': message (received: ...)`` matters more than it looks: the
        field path is the only part that tells the model *where* to make a change,
        and putting it first measurably raises the success rate of the repair.
        """
        lines: list[str] = []
        for detail in error.errors():
            location = ".".join(str(part) for part in detail["loc"]) or "(root)"
            message = detail["msg"]
            received = detail.get("input")
            received_text = ""
            if received is not None and not isinstance(received, (dict, list)):
                received_text = f" (received: {received!r})"
            lines.append(f"- Field '{location}': {message}{received_text}")

        rendered = "\n".join(lines)
        if len(rendered) > self._max_error_chars:
            kept = rendered[: self._max_error_chars]
            remaining = len(lines) - kept.count("\n") - 1
            rendered = f"{kept}\n- ... and {max(0, remaining)} further error(s)."
        return rendered


def schema_json(model: type[BaseModel]) -> dict[str, Any]:
    """Return a model's JSON schema. Thin wrapper, kept for a single import site."""
    return model.model_json_schema()
