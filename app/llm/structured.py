"""Structured generation with a bounded repair loop.

This is the orchestrator behind FR-025. On a validation failure it does **not**
resample the same prompt — that would only reroll the dice on a model that has
already demonstrated it misread the instructions. Instead it sends a repair
prompt containing the previous output *and the exact validation errors*, which
turns the retry into a correction.

In practice the failures this recovers from are mundane and repeatable: a score
of 140, ``confidence: "high"`` where a float was required, evidence returned as a
list of strings, or a missing required field. Each one is a single well-targeted
error message away from being fixed, and blind resampling fixes none of them
reliably.

The budget is bounded (``LLM_MAX_RETRIES``) because the alternative on a free
tier is a loop that turns one bad response into a rate-limit ban.
"""

from __future__ import annotations

import time
from typing import Any

from app.parsers.structured_parser import ParseFailure, StructuredOutputParser
from app.prompts.registry import PromptRegistry, schema_instructions
from app.rag.base import LLMClient, SchemaT, StructuredResult
from app.utils.exceptions import LLMRateLimitError, OutputValidationError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class StructuredGenerator:
    """Generates validated instances of a caller-supplied schema.

    Args:
        client: The generation provider.
        registry: Supplies the repair template.
        max_retries: Repair attempts after the initial one. ``0`` disables repair.
        backoff_seconds: Base delay between attempts, doubled each round. Exists
            because the most common cause of a *second* failure is transient
            provider pressure, and hammering it immediately makes that worse.
        repair_template: Name of the repair template in the registry.
        json_mode: Request provider-level JSON decoding.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        registry: PromptRegistry,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
        repair_template: str = "repair_v1",
        json_mode: bool = True,
        rate_limit_retries: int = 3,
        rate_limit_wait_seconds: float = 20.0,
    ) -> None:
        self._client = client
        self._registry = registry
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._repair_template = repair_template
        self._json_mode = json_mode
        self._rate_limit_retries = rate_limit_retries
        self._rate_limit_wait = rate_limit_wait_seconds

    def _generate_raw(self, prompt: str, system: str | None) -> str:
        """Call the provider, waiting out a per-minute rate limit rather than failing.

        Free-tier providers enforce a tokens-per-minute window that a single large
        request can momentarily exhaust. The design elsewhere treats a rate limit
        as terminal — correct for an outage — but for a self-hosted single-user
        deployment the friendlier behaviour is to wait for the window to refill
        (it resets within a minute) and try again, bounded so a genuine sustained
        limit still surfaces rather than hanging forever.

        Raises:
            LLMRateLimitError: The window did not clear within the retry budget.
            LLMError: Any other provider failure, propagated immediately.
        """
        for attempt in range(self._rate_limit_retries + 1):
            try:
                return self._client.generate(
                    prompt, system=system, json_mode=self._json_mode
                )
            except LLMRateLimitError:
                if attempt >= self._rate_limit_retries:
                    raise
                logger.warning(
                    "rate limited; waiting for the provider window to refill",
                    extra={
                        "attempt": attempt + 1,
                        "wait_s": self._rate_limit_wait,
                        "remaining_retries": self._rate_limit_retries - attempt,
                    },
                )
                time.sleep(self._rate_limit_wait)
        # Unreachable: the loop either returns or raises.
        raise LLMRateLimitError("Rate limit retry loop exhausted.")

    @property
    def model_name(self) -> str:
        """Resolved model identifier."""
        return self._client.model_name

    def generate(
        self, prompt: str, schema: type[SchemaT], *, system: str | None = None
    ) -> StructuredResult[SchemaT]:
        """Generate and validate, repairing on failure.

        Args:
            prompt: The fully rendered prompt.
            schema: Pydantic model the response must satisfy.
            system: Optional system message.

        Returns:
            The validated instance, the raw text it came from, and how many
            repair rounds were needed.

        Raises:
            OutputValidationError: The repair budget was exhausted.
            LLMError: The provider failed. Propagated unchanged — a provider
                outage is not something a repair prompt can fix, so retrying it
                here would waste the budget on the wrong problem.
        """
        parser: StructuredOutputParser[SchemaT] = StructuredOutputParser(schema)
        attempts: list[str] = []
        failure: ParseFailure | None = None
        current_prompt = prompt
        current_system = system

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                delay = self._backoff * (2 ** (attempt - 1))
                if delay > 0:
                    time.sleep(delay)
                logger.info(
                    "retrying with repair prompt",
                    extra={"attempt": attempt, "schema": schema.__name__, "delay_s": delay},
                )

            raw = self._generate_raw(current_prompt, current_system)
            attempts.append(raw)

            try:
                value = parser.parse(raw)
            except ParseFailure as parse_failure:
                failure = parse_failure
                if attempt >= self._max_retries:
                    break
                current_prompt, current_system = self._repair_prompt(schema, parse_failure)
                continue

            if attempt > 0:
                logger.info(
                    "validation succeeded after repair",
                    extra={"schema": schema.__name__, "retry_count": attempt},
                )
            return StructuredResult(value=value, raw=raw, retry_count=attempt, attempts=attempts)

        assert failure is not None  # the loop only breaks after a failure
        logger.error(
            "output validation exhausted",
            extra={"schema": schema.__name__, "attempts": len(attempts)},
        )
        raise OutputValidationError(
            f"The model could not produce a valid {schema.__name__} after "
            f"{len(attempts)} attempt(s). This usually means the retrieved context was too "
            f"thin to support the required fields.",
            details={
                "schema": schema.__name__,
                "attempts": len(attempts),
                "validation_errors": failure.errors,
                # Truncated: the raw response can be kilobytes, and the error
                # envelope is returned to a client.
                "last_response_preview": failure.raw[:500],
            },
        )

    def _repair_prompt(self, schema: type[Any], failure: ParseFailure) -> tuple[str, str | None]:
        """Build the repair prompt and system message for the next attempt."""
        spec = self._registry.get(self._repair_template)
        rendered = spec.compile().format(
            output_schema=schema_instructions(schema),
            errors=failure.errors,
            # The previous response is capped for the same reason as above: a
            # runaway response must not push the instructions out of context on
            # the very attempt meant to correct it.
            raw_output=failure.raw[:4000],
        )
        return rendered, spec.system or None


class ScriptedGenerator:
    """Returns pre-scripted responses. For tests only.

    Lives beside the real implementation rather than in the test package because
    it satisfies the same :class:`~app.rag.base.StructuredGenerator` protocol, and
    keeping them adjacent makes a divergence between them obvious at review time.

    Raising a ``ParseFailure``-shaped script entry is how the repair loop itself
    is tested without a provider: the first script entry is invalid, the second
    valid, and the test asserts ``retry_count == 1``.
    """

    def __init__(
        self,
        responses: list[str],
        *,
        model_name: str = "scripted-model",
        registry: PromptRegistry | None = None,
        max_retries: int = 2,
    ) -> None:
        self._responses = list(responses)
        self._model_name = model_name
        self._registry = registry
        self._max_retries = max_retries
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        """Identifier reported in place of a real model name."""
        return self._model_name

    def generate(
        self,
        prompt: str,
        schema: type[SchemaT],
        *,
        system: str | None = None,  # noqa: ARG002 - required by the protocol
    ) -> StructuredResult[SchemaT]:
        """Walk the script until one entry validates.

        Raises:
            OutputValidationError: The script was exhausted without a valid entry.
        """
        parser: StructuredOutputParser[SchemaT] = StructuredOutputParser(schema)
        attempts: list[str] = []
        failure: ParseFailure | None = None

        for index in range(min(len(self._responses), self._max_retries + 1)):
            self.prompts.append(prompt)
            raw = self._responses[index]
            attempts.append(raw)
            try:
                value = parser.parse(raw)
            except ParseFailure as parse_failure:
                failure = parse_failure
                continue
            return StructuredResult(value=value, raw=raw, retry_count=index, attempts=attempts)

        raise OutputValidationError(
            f"Scripted generator exhausted after {len(attempts)} attempt(s).",
            details={
                "schema": schema.__name__,
                "attempts": len(attempts),
                "validation_errors": failure.errors if failure else "no responses scripted",
            },
        )
