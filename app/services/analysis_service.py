"""Resume analysis — the domain layer over the generic engine.

Everything resume-specific in this system lives here, in the prompt templates, and
in :mod:`app.schemas.analysis`. The engine is handed three values — a retrieval
plan, a template, and a schema — and returns a validated object. Swapping this
module for ``ContractAnalysisService`` would change no engine code, which is the
claim SC-008 tests.

The most important design decision in this file is the **retrieval plan**. A
single query ("analyze this resume") retrieves generic prose about the candidate
and reliably misses low-frequency facets: certifications, leadership, and
publications never win a top-k slot against the densely-worded skills section.
FR-027 enumerates twelve assessment dimensions, so the plan issues one query per
dimension and merges the results. That is the difference between "five chunks
about nothing in particular" and coverage of every field the schema asks for.
"""

from __future__ import annotations

from app.prompts.registry import PromptRegistry, schema_instructions
from app.rag.pipeline import ProgressCallback, RAGPipeline
from app.schemas.analysis import AnalysisResult, ResumeAnalysis
from app.schemas.rag import (
    DocumentManifest,
    DocumentType,
    RetrievalPlan,
    RetrievalPlanStep,
    StageTimings,
)
from app.services.cache import AnalysisCache
from app.utils.logging import get_logger, get_request_id
from app.utils.timing import Stopwatch

logger = get_logger(__name__)

RESUME_FACETS: tuple[tuple[str, str], ...] = (
    (
        "programming_languages",
        "programming languages, software engineering, coding proficiency, "
        "Python Java JavaScript TypeScript C++ Go Rust SQL",
    ),
    (
        "frameworks_libraries",
        "frameworks and libraries, React FastAPI Django Spring Node PyTorch TensorFlow "
        "scikit-learn pandas LangChain",
    ),
    (
        "cloud_infrastructure",
        "cloud platforms and infrastructure, AWS Azure GCP Docker Kubernetes Terraform "
        "CI/CD DevOps deployment",
    ),
    (
        "machine_learning_ai",
        "machine learning, deep learning, artificial intelligence, LLMs, NLP, computer "
        "vision, model training, MLOps, RAG",
    ),
    (
        "projects",
        "personal and professional projects, systems built, technical accomplishments, "
        "open source contributions",
    ),
    (
        "work_experience",
        "work experience, employment history, job titles, responsibilities, tenure, "
        "measurable impact and outcomes",
    ),
    (
        "education",
        "education, university degree, major, GPA, academic coursework, thesis",
    ),
    (
        "certifications",
        "certifications, professional licences, accredited training, online course "
        "credentials",
    ),
    (
        "leadership",
        "leadership, mentoring, team lead, ownership, initiative, cross-functional "
        "collaboration",
    ),
    (
        "soft_skills",
        "communication, collaboration, problem solving, stakeholder management, "
        "presentation, writing",
    ),
    (
        "achievements",
        "quantified achievements, metrics, percentage improvements, scale, awards, "
        "recognition, publications",
    ),
    (
        "profile_summary",
        "professional summary, career objective, headline, seniority, specialisation",
    ),
)
"""Facets queried against the resume.

Each query is written as a *bag of related terms* rather than a natural question.
This is targeting a hybrid retriever: the dense half matches the semantic field
regardless of phrasing, and the lexical half matches the literal tokens — which is
what surfaces a chunk containing only the word ``Kubernetes`` with no surrounding
prose, a shape extremely common in a resume's skills block.
"""

JOB_FACETS: tuple[tuple[str, str], ...] = (
    (
        "required_skills",
        "required skills, must have, minimum qualifications, essential technical "
        "requirements, proficiency in",
    ),
    (
        "preferred_skills",
        "preferred qualifications, nice to have, bonus, desirable experience, plus",
    ),
    (
        "responsibilities",
        "responsibilities, what you will do, day to day duties, role scope, ownership",
    ),
    (
        "experience_requirements",
        "years of experience required, seniority level, education requirements, degree",
    ),
)
"""Facets queried against the job description."""


def build_resume_analysis_plan(
    *,
    resume_document_id: str,
    job_document_id: str,
    resume_top_k: int | None = None,
    job_top_k: int | None = None,
) -> RetrievalPlan:
    """Compose the retrieval plan for a resume-to-job analysis.

    A module-level function rather than a method: the plan is a pure value derived
    from two ids, so it is directly testable and directly reusable by the
    evaluation harness without constructing a service.

    Steps are filtered by ``document_id`` as well as ``doc_type``. Filtering on
    type alone would let a *previously uploaded* resume's passages leak into this
    analysis, since all documents share one collection — a silent correctness bug
    that would only surface as inexplicably good scores.
    """
    steps: list[RetrievalPlanStep] = [
        RetrievalPlanStep(
            name=name,
            query=query,
            doc_type=DocumentType.RESUME,
            document_id=resume_document_id,
            top_k=resume_top_k,
        )
        for name, query in RESUME_FACETS
    ]
    steps.extend(
        RetrievalPlanStep(
            name=name,
            query=query,
            doc_type=DocumentType.JOB_DESCRIPTION,
            document_id=job_document_id,
            top_k=job_top_k,
        )
        for name, query in JOB_FACETS
    )
    return RetrievalPlan(steps=tuple(steps))


class AnalysisService:
    """Produces a validated, audited resume analysis.

    Args:
        pipeline: The generic RAG pipeline.
        registry: Prompt template source.
        cache: Exact-match analysis cache.
        template_name: Default analysis template.
        resume_top_k: Passages per resume facet. Small by design — twelve facets
            at three passages each already approaches the context budget, and the
            budget exists to keep Principle II enforceable.
        job_top_k: Passages per job-description facet. Larger, because there are
            only four job facets and the requirements list is what the entire
            analysis is scored against.
    """

    def __init__(
        self,
        *,
        pipeline: RAGPipeline,
        registry: PromptRegistry,
        cache: AnalysisCache | None = None,
        template_name: str = "resume_analysis_v1",
        resume_top_k: int = 3,
        job_top_k: int = 4,
    ) -> None:
        self._pipeline = pipeline
        self._registry = registry
        self._cache = cache
        self._template_name = template_name
        self._resume_top_k = resume_top_k
        self._job_top_k = job_top_k

    def analyze(
        self,
        *,
        resume: DocumentManifest,
        job: DocumentManifest,
        job_title: str = "",
        template_name: str | None = None,
        top_k: int | None = None,
        strategy: str | None = None,
        use_cache: bool = True,
        on_event: ProgressCallback | None = None,
    ) -> AnalysisResult:
        """Analyze a resume against a job description.

        Args:
            resume: Manifest of the ingested resume.
            job: Manifest of the ingested job description.
            job_title: Role title for the prompt. Falls back to the filename.
            template_name: Override the analysis template.
            top_k: Override passages retrieved per facet.
            strategy: Override the retrieval strategy.
            use_cache: Consult and populate the analysis cache.
            on_event: Optional stage-progress callback, used by the SSE route.

        Returns:
            The validated analysis with full provenance.

        Raises:
            InsufficientContextError: Too little was retrieved to ground an answer.
            OutputValidationError: The repair budget was exhausted.
            LLMError: The generation provider failed.
        """
        timings = StageTimings()
        template = template_name or self._template_name
        resolved_strategy = strategy or "default"

        with Stopwatch("total", timings):
            cache_key = AnalysisCache.build_key(
                resume_id=resume.document_id,
                job_id=job.document_id,
                template=template,
                model=self._pipeline_model(),
                strategy=resolved_strategy,
                top_k=top_k or self._resume_top_k,
            )

            if use_cache and self._cache is not None:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    if on_event is not None:
                        on_event("cache", {"status": "hit"})
                    return cached.model_copy(
                        update={"cached": True, "request_id": get_request_id() or ""}
                    )

            plan = build_resume_analysis_plan(
                resume_document_id=resume.document_id,
                job_document_id=job.document_id,
                resume_top_k=top_k or self._resume_top_k,
                job_top_k=top_k or self._job_top_k,
            )

            spec = self._registry.get(template)
            result = self._pipeline.run(
                plan=plan,
                template=spec.compile(),
                schema=ResumeAnalysis,
                system=spec.system or None,
                timings=timings,
                on_event=on_event,
                output_schema=schema_instructions(ResumeAnalysis),
                job_title=job_title or job.filename,
            )

        analysis: ResumeAnalysis = result.value
        warnings = self._collect_warnings(analysis, result)

        envelope = AnalysisResult(
            request_id=get_request_id() or "",
            analysis=analysis,
            resume=resume,
            job=job,
            retrieval=result.trace,
            timings=timings,
            model=self._pipeline_model(),
            prompt_template=template,
            retry_count=result.retry_count,
            cached=False,
            warnings=warnings,
        )

        if use_cache and self._cache is not None:
            self._cache.set(cache_key, envelope)

        logger.info(
            "analysis complete",
            extra={
                "resume_id": resume.document_id,
                "job_id": job.document_id,
                "overall_score": analysis.overall_score,
                "confidence": analysis.confidence,
                "evidence_items": len(analysis.evidence),
                "retry_count": result.retry_count,
                "warnings": len(warnings),
                **timings.as_reported(),
            },
        )
        return envelope

    def _pipeline_model(self) -> str:
        """Resolved generation model id, for the cache key and the response."""
        return self._pipeline._generator.model_name

    @staticmethod
    def _collect_warnings(analysis: ResumeAnalysis, result: object) -> list[str]:
        """Assemble caller-visible warnings about the quality of this analysis.

        Citation verification happens here, and it is the sharpest signal
        available: a citation the model produced that was never in the prompt is
        a source it invented. The analysis is still returned — it may be largely
        correct — but the caller is told, which is strictly better than either
        silently trusting it or discarding useful work.
        """
        warnings = list(analysis.grounding_warnings)

        chunks = getattr(result, "chunks", [])
        valid_citations = {chunk.citation for chunk in chunks}
        unsupported = analysis.unsupported_citations(valid_citations)
        if unsupported:
            warnings.append(
                f"{len(unsupported)} evidence citation(s) do not match any retrieved passage: "
                f"{', '.join(unsupported[:3])}. Treat those claims with caution."
            )

        trace = getattr(result, "trace", None)
        if trace is not None:
            empty = [step.name for step in trace.steps if step.returned == 0]
            if empty:
                warnings.append(
                    f"No passages were retrieved for: {', '.join(empty)}. "
                    f"Conclusions about those areas are unsupported."
                )
            if trace.budget_truncated:
                warnings.append(
                    "Retrieved context exceeded the prompt budget and was truncated; "
                    "some passages were not shown to the model."
                )

        if getattr(result, "retry_count", 0) > 0:
            warnings.append(
                f"The model required {result.retry_count} repair attempt(s) to produce "  # type: ignore[attr-defined]
                f"valid output."
            )
        return warnings
