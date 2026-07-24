"""Pipeline tests.

The centrepiece is ``TestSchemaIndependence``: the same pipeline instance driving
two unrelated answer schemas with zero engine changes. That is SC-008 and the
architectural claim of the whole project, so it is asserted rather than asserted
*about* in a README.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from app.rag.embeddings import HashingEmbedder
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import InMemoryVectorStore
from app.schemas.rag import (
    DocumentType,
    RetrievalPlan,
    RetrievalPlanStep,
    RetrievalStrategy,
    StageTimings,
)
from app.tests.conftest import build_pipeline
from app.tests.fakes import StaticGenerator, make_chunk
from app.utils.exceptions import InsufficientContextError

CORPUS = [
    "Python PyTorch and TensorFlow for deep learning model development",
    "AWS SageMaker Lambda and S3 for cloud deployment of services",
    "Kubernetes and Terraform for container orchestration",
    "Led a team of four engineers on a classification pipeline",
    "MSc Computer Science University of Edinburgh 2019",
]


class FormatTemplate:
    """Minimal stand-in for a LangChain PromptTemplate."""

    def __init__(self, template: str = "{context}") -> None:
        self._template = template

    def format(self, **kwargs: object) -> str:
        return self._template.format(**kwargs)


class Answer(BaseModel):
    """A trivial schema for pipeline tests."""

    summary: str = "ok"


class ClauseRisk(BaseModel):
    """A schema from an entirely different domain, used to prove reuse."""

    clause_type: str
    risk_level: str = Field(pattern="^(low|medium|high)$")
    rationale: str


@pytest.fixture
def populated(store: InMemoryVectorStore, embedder: HashingEmbedder) -> InMemoryVectorStore:
    chunks = [
        make_chunk(text, document_id="r1", chunk_index=index)
        for index, text in enumerate(CORPUS)
    ]
    chunks.append(
        make_chunk(
            "Required: Kubernetes, Terraform, and production RAG experience",
            document_id="j1",
            filename="job.pdf",
            doc_type=DocumentType.JOB_DESCRIPTION,
            chunk_index=0,
        )
    )
    store.add(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    return store


@pytest.fixture
def factory(populated: InMemoryVectorStore, embedder: HashingEmbedder):  # type: ignore[no-untyped-def]
    from app.rag.retriever import RetrieverFactory

    return RetrieverFactory(populated, embedder, fetch_k=10)


def two_step_plan() -> RetrievalPlan:
    return RetrievalPlan(
        steps=(
            RetrievalPlanStep(
                name="skills", query="Python PyTorch deep learning", document_id="r1", top_k=2
            ),
            RetrievalPlanStep(
                name="requirements", query="Kubernetes Terraform", document_id="j1", top_k=2
            ),
        )
    )


class TestExecutePlan:
    def test_runs_every_step(self, factory, prompt_builder) -> None:
        pipeline = build_pipeline(factory, prompt_builder, StaticGenerator(Answer()))
        groups, trace = pipeline.execute_plan(two_step_plan())

        assert set(groups) == {"skills", "requirements"}
        assert len(trace.steps) == 2

    def test_keeps_results_grouped_by_facet(self, factory, prompt_builder) -> None:
        """The prompt presents facets as titled sections; merging loses that signal."""
        pipeline = build_pipeline(factory, prompt_builder, StaticGenerator(Answer()))
        groups, _ = pipeline.execute_plan(two_step_plan())

        assert all(hit.chunk.metadata.document_id == "r1" for hit in groups["skills"])
        assert all(hit.chunk.metadata.document_id == "j1" for hit in groups["requirements"])

    def test_traces_each_step(self, factory, prompt_builder) -> None:
        pipeline = build_pipeline(factory, prompt_builder, StaticGenerator(Answer()))
        _, trace = pipeline.execute_plan(two_step_plan())

        step = trace.steps[0]
        assert step.name == "skills"
        assert step.query
        assert step.strategy
        assert step.duration_ms >= 0

    def test_counts_unique_chunks_across_facets(self, factory, prompt_builder) -> None:
        pipeline = build_pipeline(factory, prompt_builder, StaticGenerator(Answer()))
        plan = RetrievalPlan(
            steps=(
                RetrievalPlanStep(name="a", query="Python PyTorch", document_id="r1", top_k=3),
                RetrievalPlanStep(name="b", query="Python PyTorch", document_id="r1", top_k=3),
            )
        )
        _, trace = pipeline.execute_plan(plan)

        assert trace.unique_chunks <= trace.total_chunks
        assert trace.deduplicated == trace.total_chunks - trace.unique_chunks

    def test_records_retrieval_timings(self, factory, prompt_builder) -> None:
        pipeline = build_pipeline(factory, prompt_builder, StaticGenerator(Answer()))
        timings = StageTimings()
        pipeline.execute_plan(two_step_plan(), timings=timings)

        assert timings.retrieve_ms is not None

    def test_honours_a_per_step_strategy_override(self, factory, prompt_builder) -> None:
        """A plan can mix strategies: lexical for a lookup, dense for a narrative."""
        pipeline = build_pipeline(factory, prompt_builder, StaticGenerator(Answer()))
        plan = RetrievalPlan(
            steps=(
                RetrievalPlanStep(
                    name="lexical",
                    query="Terraform",
                    document_id="j1",
                    strategy=RetrievalStrategy.BM25,
                ),
            )
        )
        _, trace = pipeline.execute_plan(plan)
        assert trace.steps[0].strategy == "bm25"


class TestRun:
    def test_produces_a_validated_answer_with_provenance(self, factory, prompt_builder) -> None:
        pipeline = build_pipeline(factory, prompt_builder, StaticGenerator(Answer(summary="done")))
        result = pipeline.run(plan=two_step_plan(), template=FormatTemplate(), schema=Answer)

        assert result.value.summary == "done"
        assert result.chunks
        assert result.trace.unique_chunks > 0
        assert result.prompt

    def test_the_prompt_contains_only_retrieved_passages(self, factory, prompt_builder) -> None:
        """Constitution Principle II, verified behaviourally."""
        generator = StaticGenerator(Answer())
        pipeline = build_pipeline(factory, prompt_builder, generator)
        result = pipeline.run(plan=two_step_plan(), template=FormatTemplate(), schema=Answer)

        prompt = generator.prompts[0]
        for chunk in result.chunks:
            assert chunk.text in prompt
        assert len(prompt) < sum(len(text) for text in CORPUS) * 3

    def test_every_included_chunk_is_cited_in_the_prompt(self, factory, prompt_builder) -> None:
        generator = StaticGenerator(Answer())
        pipeline = build_pipeline(factory, prompt_builder, generator)
        result = pipeline.run(plan=two_step_plan(), template=FormatTemplate(), schema=Answer)

        for chunk in result.chunks:
            assert chunk.citation in generator.prompts[0]

    def test_passes_the_system_message_through(self, factory, prompt_builder) -> None:
        generator = StaticGenerator(Answer())
        pipeline = build_pipeline(factory, prompt_builder, generator)
        pipeline.run(
            plan=two_step_plan(),
            template=FormatTemplate(),
            schema=Answer,
            system="You are careful.",
        )
        assert generator.systems[0] == "You are careful."

    def test_records_timings_for_every_stage(self, factory, prompt_builder) -> None:
        """FR-035: the response carries a stage-level breakdown."""
        pipeline = build_pipeline(factory, prompt_builder, StaticGenerator(Answer()))
        result = pipeline.run(plan=two_step_plan(), template=FormatTemplate(), schema=Answer)

        assert result.timings.retrieve_ms is not None
        assert result.timings.prompt_ms is not None
        assert result.timings.llm_ms is not None

    def test_refuses_to_generate_without_enough_grounding(
        self, factory, prompt_builder
    ) -> None:
        """Prompting with a near-empty context is when models fabricate most freely."""
        pipeline = build_pipeline(factory, prompt_builder, StaticGenerator(Answer()), min_chunks=5)
        plan = RetrievalPlan(
            steps=(
                RetrievalPlanStep(
                    name="nothing",
                    query="xylophone bassoon harpsichord",
                    document_id="does-not-exist",
                    top_k=2,
                    strategy=RetrievalStrategy.BM25,
                ),
            )
        )

        with pytest.raises(InsufficientContextError) as caught:
            pipeline.run(plan=plan, template=FormatTemplate(), schema=Answer)
        assert "nothing" in caught.value.details["empty_facets"]

    def test_emits_progress_events(self, factory, prompt_builder) -> None:
        """The SSE route depends on these; the pipeline stays transport-agnostic."""
        events: list[tuple[str, dict]] = []
        pipeline = build_pipeline(factory, prompt_builder, StaticGenerator(Answer()))
        pipeline.run(
            plan=two_step_plan(),
            template=FormatTemplate(),
            schema=Answer,
            on_event=lambda name, payload: events.append((name, payload)),
        )

        names = [name for name, _ in events]
        assert "retrieval" in names
        assert "generation" in names
        assert "validation" in names


class TestSchemaIndependence:
    """SC-008: one engine, many domains."""

    def test_the_same_pipeline_drives_a_different_schema(self, factory, prompt_builder) -> None:
        """No engine file changes between these two calls. That is the whole thesis."""
        resume_pipeline = build_pipeline(
            factory, prompt_builder, StaticGenerator(Answer(summary="a"))
        )
        resume_result = resume_pipeline.run(
            plan=two_step_plan(), template=FormatTemplate(), schema=Answer
        )
        assert isinstance(resume_result.value, Answer)

        legal_pipeline = build_pipeline(
            factory,
            prompt_builder,
            StaticGenerator(
                ClauseRisk(clause_type="indemnity", risk_level="high", rationale="uncapped")
            ),
        )
        legal_plan = RetrievalPlan(
            steps=(
                RetrievalPlanStep(
                    name="indemnity_clauses", query="Kubernetes Terraform", document_id="j1"
                ),
            )
        )
        legal_result = legal_pipeline.run(
            plan=legal_plan,
            template=FormatTemplate("Assess this contract:\n{context}"),
            schema=ClauseRisk,
        )

        assert isinstance(legal_result.value, ClauseRisk)
        assert legal_result.value.risk_level == "high"

    def test_a_retrieval_plan_is_plain_data(self) -> None:
        """Plans are values the engine executes, not code inside the engine."""
        plan = RetrievalPlan(
            steps=(
                RetrievalPlanStep(
                    name="clauses", query="liability", doc_type=DocumentType.GENERIC
                ),
            )
        )
        assert plan.steps[0].filters() == {"doc_type": "generic"}
        assert len(plan) == 1

    def test_the_pipeline_signature_is_generic_over_its_schema(self) -> None:
        """The engine receives its answer type; it does not know one.

        Domain-freedom of the *code* is enforced by the AST gate in
        ``test_architecture.py``, which correctly distinguishes an identifier
        from an explanatory comment. What is checked here is the complementary
        property that gate cannot see: the public signature accepts an arbitrary
        schema type rather than hardcoding one.
        """
        import inspect

        signature = inspect.signature(RAGPipeline.run)
        assert "schema" in signature.parameters
        assert signature.parameters["schema"].kind is inspect.Parameter.KEYWORD_ONLY
