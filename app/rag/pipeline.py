"""The generic RAG pipeline.

This is the class the whole project exists to justify. It is generic over the
answer schema and knows nothing about resumes — it executes a retrieval plan,
budgets the results, renders a prompt, generates, and returns a validated object
of whatever type the caller asked for:

    >>> pipeline = RAGPipeline(retriever_factory=..., prompt_builder=..., generator=...)
    >>> analysis = pipeline.run(plan=resume_plan, template=resume_tpl, schema=ResumeAnalysis)
    >>> contract = pipeline.run(plan=clause_plan, template=clause_tpl, schema=ClauseRisk)

Both calls exercise identical engine code. That is SC-008, and
``tests/test_pipeline.py::test_same_pipeline_drives_a_different_schema`` asserts
it rather than asserting a promise about it.

Note the imports: ``app.rag.*``, ``app.schemas.rag``, ``app.utils.*``. Nothing
from ``app.services``, ``app.prompts``, or ``app.llm`` — enforced by
``tests/test_architecture.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from app.rag.base import SchemaT, StructuredGenerator
from app.rag.prompt_builder import PromptBuilder
from app.rag.retriever import deduplicate
from app.schemas.rag import (
    RetrievalPlan,
    RetrievalStepTrace,
    RetrievalTrace,
    RetrievedChunk,
    StageTimings,
)
from app.utils.exceptions import InsufficientContextError
from app.utils.logging import get_logger
from app.utils.timing import Stopwatch

logger = get_logger(__name__)

ProgressCallback = Callable[[str, dict[str, Any]], None]
"""Called at each stage boundary with ``(event_name, payload)``.

Exists so a transport can report progress without the pipeline knowing what a
Server-Sent Event is. The SSE route supplies a callback that forwards to a queue;
every other caller passes ``None`` and pays nothing.
"""


def _noop_event(event: str, payload: dict[str, Any]) -> None:
    """Default progress callback. Avoids an ``if on_event`` guard at five call sites."""


class PipelineResult:
    """A validated answer with everything needed to audit it.

    The answer is never returned bare. Provenance — which passages were used, how
    long each stage took, how many repairs were needed — travels with it, because
    an unauditable analysis is indistinguishable from a plausible guess.
    """

    __slots__ = ("chunks", "prompt", "retry_count", "timings", "trace", "value")

    def __init__(
        self,
        value: Any,
        *,
        chunks: list[RetrievedChunk],
        trace: RetrievalTrace,
        timings: StageTimings,
        prompt: str,
        retry_count: int,
    ) -> None:
        self.value = value
        self.chunks = chunks
        self.trace = trace
        self.timings = timings
        self.prompt = prompt
        self.retry_count = retry_count


class RAGPipeline:
    """Retrieval-augmented generation, parameterised by schema and plan.

    Args:
        retriever_factory: Builds a retriever for a named strategy. A factory
            rather than a retriever, so a single plan can mix strategies —
            lexical for a certifications lookup, dense for a narrative facet.
        prompt_builder: Assembles and budgets the prompt context.
        generator: Turns a prompt into a validated schema instance.
        default_strategy: Strategy for plan steps that do not name one.
        default_top_k: Top-k for plan steps that do not name one.
        min_chunks: Below this many retrieved chunks, refuse to generate.
        query_cache: Optional cache in front of sub-query retrieval.
    """

    def __init__(
        self,
        *,
        retriever_factory: Any,
        prompt_builder: PromptBuilder,
        generator: StructuredGenerator,
        default_strategy: str = "hybrid",
        default_top_k: int = 5,
        min_chunks: int = 2,
        query_cache: Any | None = None,
    ) -> None:
        self._factory = retriever_factory
        self._builder = prompt_builder
        self._generator = generator
        self._default_strategy = default_strategy
        self._default_top_k = default_top_k
        self._min_chunks = min_chunks
        self._query_cache = query_cache

    # ---------------------------------------------------------- retrieval ---

    def execute_plan(
        self, plan: RetrievalPlan, *, timings: StageTimings | None = None
    ) -> tuple[dict[str, list[RetrievedChunk]], RetrievalTrace]:
        """Run every step of a retrieval plan.

        Results stay grouped by facet name rather than being merged immediately,
        because the prompt presents them as titled sections — telling the model
        *why* each passage was retrieved measurably improves how it uses them,
        and an undifferentiated wall of text loses that signal.

        Deduplication is applied within each group and, separately, tracked
        across groups: a chunk relevant to two facets is legitimately shown under
        both, but is counted once against the trace's unique total.
        """
        trace = RetrievalTrace()
        groups: dict[str, list[RetrievedChunk]] = {}
        seen: set[str] = set()

        for step in plan.steps:
            strategy = (step.strategy.value if step.strategy else None) or self._default_strategy
            top_k = step.top_k or self._default_top_k
            filters = step.filters()

            with Stopwatch("retrieve", timings, step=step.name) as watch:
                hits, cached = self._retrieve_step(step.query, strategy, top_k, filters)

            hits = deduplicate(hits)
            groups[step.name] = hits
            seen.update(hit.chunk_id for hit in hits)

            trace.add(
                RetrievalStepTrace(
                    name=step.name,
                    query=step.query,
                    strategy=strategy,
                    candidates=len(hits),
                    returned=len(hits),
                    duration_ms=round(watch.elapsed_ms, 3),
                    cached=cached,
                    top_score=hits[0].score if hits else None,
                )
            )

        trace.unique_chunks = len(seen)
        trace.deduplicated = max(0, trace.total_chunks - trace.unique_chunks)
        logger.info(
            "retrieval plan complete",
            extra={
                "stage": "retrieve",
                "steps": len(plan),
                "total_chunks": trace.total_chunks,
                "unique_chunks": trace.unique_chunks,
            },
        )
        return groups, trace

    def _retrieve_step(
        self, query: str, strategy: str, top_k: int, filters: dict[str, Any]
    ) -> tuple[list[RetrievedChunk], bool]:
        """Retrieve one step, consulting the query cache when configured.

        The cache key includes the strategy, top-k, and filters — not just the
        query text. Two facets can share wording but target different documents,
        and a cache keyed on text alone would serve the resume's passages for a
        job-description query. That failure would be silent and would corrupt
        every downstream conclusion.
        """
        if self._query_cache is None:
            return self._retriever(strategy).retrieve(query, top_k=top_k, filters=filters), False

        key = self._query_cache.build_key(query, strategy, top_k, filters)
        cached = self._query_cache.get(key)
        if cached is not None:
            return list(cached), True

        hits = self._retriever(strategy).retrieve(query, top_k=top_k, filters=filters)
        self._query_cache.set(key, hits)
        return hits, False

    def _retriever(self, strategy: str) -> Any:
        """Resolve a retriever for ``strategy``."""
        return self._factory.create(strategy)

    # ------------------------------------------------------------ generate ---

    def run(
        self,
        *,
        plan: RetrievalPlan,
        template: Any,
        schema: type[SchemaT],
        system: str | None = None,
        timings: StageTimings | None = None,
        context_variable: str = "context",
        on_event: ProgressCallback | None = None,
        **variables: Any,
    ) -> PipelineResult:
        """Execute the full pipeline and return a validated answer.

        Args:
            plan: Retrieval plan to execute.
            template: Prompt template exposing ``.format(**kwargs)``.
            schema: Pydantic model the response must satisfy.
            system: Optional system message.
            timings: Stage timings to accumulate into.
            context_variable: Template variable receiving the rendered context.
            on_event: Optional stage-progress callback.
            **variables: Remaining template variables.

        Raises:
            InsufficientContextError: Too few passages were retrieved to ground
                an answer. Raised *before* generation — prompting a model with a
                near-empty context is the condition under which it fabricates
                most freely, so the correct move is to refuse.
            OutputValidationError: The repair budget was exhausted.
            LLMError: The provider failed.
        """
        timings = timings if timings is not None else StageTimings()
        emit = on_event or _noop_event

        emit("retrieval", {"status": "started", "steps": len(plan)})
        groups, trace = self.execute_plan(plan, timings=timings)
        self._require_context(groups, trace)
        emit(
            "retrieval",
            {
                "status": "completed",
                "unique_chunks": trace.unique_chunks,
                "duration_ms": timings.retrieve_ms,
            },
        )

        with Stopwatch("prompt", timings):
            prompt, chunks, truncated = self._builder.build(
                template, groups, context_variable=context_variable, **variables
            )
        trace.budget_truncated = truncated
        emit("prompt", {"status": "completed", "chunks": len(chunks), "chars": len(prompt)})

        emit("generation", {"status": "started", "model": self._generator.model_name})
        with Stopwatch("llm", timings):
            result = self._generator.generate(prompt, schema, system=system)
        emit(
            "generation",
            {"status": "completed", "retry_count": result.retry_count,
             "duration_ms": timings.llm_ms},
        )
        emit("validation", {"status": "completed", "schema": schema.__name__})

        logger.info(
            "pipeline complete",
            extra={
                "schema": schema.__name__,
                "chunks_used": len(chunks),
                "retry_count": result.retry_count,
                **timings.as_reported(),
            },
        )
        return PipelineResult(
            result.value,
            chunks=chunks,
            trace=trace,
            timings=timings,
            prompt=prompt,
            retry_count=result.retry_count,
        )

    def _require_context(
        self, groups: Mapping[str, Sequence[RetrievedChunk]], trace: RetrievalTrace
    ) -> None:
        """Refuse to generate without enough grounding.

        Raises:
            InsufficientContextError: Fewer than ``min_chunks`` unique passages.
        """
        if trace.unique_chunks >= self._min_chunks:
            return
        raise InsufficientContextError(
            "Not enough relevant content was retrieved to produce a grounded answer. "
            "The documents may be too short, or unrelated to the query.",
            details={
                "unique_chunks": trace.unique_chunks,
                "required": self._min_chunks,
                "empty_facets": [name for name, hits in groups.items() if not hits],
            },
        )
