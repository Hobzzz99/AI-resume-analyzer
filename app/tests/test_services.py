"""Service-layer tests.

Covers the resume retrieval plan, cache semantics, warning propagation, and the
chat service. The cache tests get particular attention: an exact-match analysis
cache that accidentally became fuzzy would serve one candidate's analysis to
another, which is the single worst failure this product could have.
"""

from __future__ import annotations

import pytest

from app.rag.embeddings import HashingEmbedder
from app.schemas.analysis import ChatTurn, ResumeAnalysis
from app.schemas.rag import DocumentManifest, DocumentType
from app.services.analysis_service import (
    JOB_FACETS,
    RESUME_FACETS,
    AnalysisService,
    build_resume_analysis_plan,
)
from app.services.cache import AnalysisCache, LRUCache, SemanticQueryCache
from app.services.chat_service import ConversationMemory
from app.tests.conftest import build_pipeline
from app.tests.fakes import StaticGenerator


def manifest(document_id: str, doc_type: DocumentType, filename: str) -> DocumentManifest:
    return DocumentManifest(
        document_id=document_id, filename=filename, doc_type=doc_type, chunk_count=5
    )


def valid_analysis(**overrides: object) -> ResumeAnalysis:
    payload: dict[str, object] = {
        "overall_score": 75,
        "technical_score": 80,
        "experience_score": 70,
        "education_score": 85,
        "ats_score": 65,
        "matched_skills": ["Python"],
        "missing_skills": ["Kubernetes"],
        "recruiter_summary": "A strong applied ML candidate with relevant experience.",
        "confidence": 0.7,
        "evidence": [
            {"claim": "Python", "quote": "Python PyTorch", "citation": "[r.pdf p.1 #0]",
             "source": "resume"}
        ],
    }
    payload.update(overrides)
    return ResumeAnalysis.model_validate(payload)


class TestRetrievalPlan:
    def test_covers_every_assessment_facet(self) -> None:
        """FR-027 enumerates twelve dimensions; one query each is the honest implementation."""
        plan = build_resume_analysis_plan(resume_document_id="r1", job_document_id="j1")
        assert len(plan) == len(RESUME_FACETS) + len(JOB_FACETS)

    def test_filters_by_document_as_well_as_type(self) -> None:
        """Type alone would let a previously uploaded resume leak into the analysis."""
        plan = build_resume_analysis_plan(resume_document_id="r1", job_document_id="j1")

        for step in plan.steps:
            filters = step.filters()
            assert "document_id" in filters
            assert filters["document_id"] in {"r1", "j1"}

    def test_resume_and_job_steps_target_their_own_document(self) -> None:
        plan = build_resume_analysis_plan(resume_document_id="r1", job_document_id="j1")
        resume_steps = [s for s in plan.steps if s.doc_type is DocumentType.RESUME]
        job_steps = [s for s in plan.steps if s.doc_type is DocumentType.JOB_DESCRIPTION]

        assert all(step.document_id == "r1" for step in resume_steps)
        assert all(step.document_id == "j1" for step in job_steps)

    def test_step_names_are_unique(self) -> None:
        """Names become prompt section headers; duplicates would collide."""
        plan = build_resume_analysis_plan(resume_document_id="r1", job_document_id="j1")
        names = [step.name for step in plan.steps]
        assert len(set(names)) == len(names)

    def test_top_k_overrides_apply(self) -> None:
        plan = build_resume_analysis_plan(
            resume_document_id="r1", job_document_id="j1", resume_top_k=2, job_top_k=6
        )
        resume_steps = [s for s in plan.steps if s.doc_type is DocumentType.RESUME]
        assert all(step.top_k == 2 for step in resume_steps)


class TestLRUCache:
    def test_stores_and_retrieves(self) -> None:
        cache: LRUCache[str] = LRUCache(max_size=2)
        cache.set("a", "value")
        assert cache.get("a") == "value"

    def test_evicts_the_least_recently_used(self) -> None:
        cache: LRUCache[str] = LRUCache(max_size=2)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.get("a")  # refreshes 'a'
        cache.set("c", "3")

        assert cache.get("a") == "1"
        assert cache.get("b") is None

    def test_tracks_hit_rate(self) -> None:
        cache: LRUCache[str] = LRUCache()
        cache.set("a", "1")
        cache.get("a")
        cache.get("missing")

        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1
        assert cache.stats["hit_rate"] == 0.5


class TestAnalysisCache:
    def test_round_trips_a_value(self) -> None:
        cache = AnalysisCache()
        key = AnalysisCache.build_key(
            resume_id="r1", job_id="j1", template="t", model="m", strategy="hybrid", top_k=3
        )
        cache.set(key, "analysis")
        assert cache.get(key) == "analysis"

    def test_different_documents_produce_different_keys(self) -> None:
        """The failure this guards against is serving one candidate's analysis to another."""
        first = AnalysisCache.build_key(
            resume_id="r1", job_id="j1", template="t", model="m", strategy="hybrid", top_k=3
        )
        second = AnalysisCache.build_key(
            resume_id="r2", job_id="j1", template="t", model="m", strategy="hybrid", top_k=3
        )
        assert first != second

    @pytest.mark.parametrize(
        "field", ["template", "model", "strategy", "top_k"]
    )
    def test_every_output_affecting_input_is_in_the_key(self, field: str) -> None:
        """Omitting one would silently return a result from a different configuration."""
        base = {
            "resume_id": "r1", "job_id": "j1", "template": "t",
            "model": "m", "strategy": "hybrid", "top_k": 3,
        }
        changed = {**base, field: 9 if field == "top_k" else "other"}
        assert AnalysisCache.build_key(**base) != AnalysisCache.build_key(**changed)

    def test_disabled_cache_never_hits(self) -> None:
        cache = AnalysisCache(enabled=False)
        cache.set("k", "v")
        assert cache.get("k") is None


class TestSemanticQueryCache:
    def test_exact_key_hits(self, embedder: HashingEmbedder) -> None:
        cache = SemanticQueryCache(embedder)
        key = SemanticQueryCache.build_key("python skills", "hybrid", 3, {"document_id": "r1"})
        cache.set(key, ["result"], query="python skills")
        assert cache.get(key, query="python skills") == ["result"]

    def test_filters_participate_in_the_key(self, embedder: HashingEmbedder) -> None:
        """Same wording, different document — a text-only key would corrupt retrieval."""
        first = SemanticQueryCache.build_key("skills", "hybrid", 3, {"document_id": "r1"})
        second = SemanticQueryCache.build_key("skills", "hybrid", 3, {"document_id": "j1"})
        assert first != second

    def test_disabled_without_an_embedder(self) -> None:
        cache = SemanticQueryCache(None)
        cache.set("k", ["v"])
        assert cache.get("k") is None


class TestConversationMemory:
    def test_records_turns(self) -> None:
        memory = ConversationMemory()
        memory.append("s1", ChatTurn(role="user", content="hello"))
        assert len(memory.history("s1")) == 1

    def test_sessions_are_isolated(self) -> None:
        memory = ConversationMemory()
        memory.append("s1", ChatTurn(role="user", content="a"))
        memory.append("s2", ChatTurn(role="user", content="b"))
        assert memory.history("s1")[0].content == "a"

    def test_trims_to_the_window(self) -> None:
        memory = ConversationMemory(max_turns=2)
        for index in range(10):
            memory.append("s1", ChatTurn(role="user", content=f"m{index}"))
        assert len(memory.history("s1")) == 4  # max_turns * 2

    def test_evicts_the_least_recently_used_session(self) -> None:
        """Without this bound the service leaks memory for as long as it stays up."""
        memory = ConversationMemory(max_sessions=2)
        for index in range(3):
            memory.append(f"s{index}", ChatTurn(role="user", content="x"))
        assert memory.history("s0") == []
        assert memory.history("s2")

    def test_renders_history_for_a_prompt(self) -> None:
        memory = ConversationMemory()
        memory.append("s1", ChatTurn(role="user", content="Does she know Go?"))
        memory.append("s1", ChatTurn(role="assistant", content="Yes."))

        rendered = ConversationMemory.render(memory.history("s1"))
        assert "USER: Does she know Go?" in rendered
        assert "ASSISTANT: Yes." in rendered

    def test_renders_an_empty_history_explicitly(self) -> None:
        assert ConversationMemory.render([]) == "(no previous messages)"

    def test_clear_removes_a_session(self) -> None:
        memory = ConversationMemory()
        memory.append("s1", ChatTurn(role="user", content="x"))
        memory.clear("s1")
        assert memory.history("s1") == []


class TestAnalysisService:
    @pytest.fixture
    def service(self, indexed_documents, retriever_factory, prompt_builder, registry):  # type: ignore[no-untyped-def]
        pipeline = build_pipeline(
            retriever_factory, prompt_builder, StaticGenerator(valid_analysis()), min_chunks=1
        )
        return AnalysisService(
            pipeline=pipeline, registry=registry, cache=AnalysisCache(), resume_top_k=2, job_top_k=2
        )

    def test_produces_an_analysis_with_full_provenance(
        self, service: AnalysisService, indexed_documents
    ) -> None:
        resume_id, job_id = indexed_documents
        result = service.analyze(
            resume=manifest(resume_id, DocumentType.RESUME, "resume.txt"),
            job=manifest(job_id, DocumentType.JOB_DESCRIPTION, "job.txt"),
            job_title="Senior AI Engineer",
        )

        assert result.analysis.overall_score == 75
        assert result.model == "static-model"
        assert result.prompt_template == "resume_analysis_v1"
        assert result.retrieval.unique_chunks > 0
        assert result.timings.total_ms is not None
        assert not result.cached

    def test_a_repeat_request_is_served_from_cache(
        self, service: AnalysisService, indexed_documents
    ) -> None:
        resume_id, job_id = indexed_documents
        args = {
            "resume": manifest(resume_id, DocumentType.RESUME, "resume.txt"),
            "job": manifest(job_id, DocumentType.JOB_DESCRIPTION, "job.txt"),
        }
        service.analyze(**args)
        second = service.analyze(**args)

        assert second.cached

    def test_cache_can_be_bypassed(
        self, service: AnalysisService, indexed_documents
    ) -> None:
        resume_id, job_id = indexed_documents
        args = {
            "resume": manifest(resume_id, DocumentType.RESUME, "resume.txt"),
            "job": manifest(job_id, DocumentType.JOB_DESCRIPTION, "job.txt"),
        }
        service.analyze(**args)
        assert not service.analyze(**args, use_cache=False).cached

    def test_warns_about_a_citation_the_model_invented(
        self, indexed_documents, retriever_factory, prompt_builder, registry
    ) -> None:
        """Fabricated provenance is reported rather than silently trusted."""
        resume_id, job_id = indexed_documents
        fabricated = valid_analysis(
            evidence=[
                {
                    "claim": "Invented",
                    "quote": "Something never retrieved",
                    "citation": "[nonexistent.pdf p.99 #99]",
                    "source": "resume",
                }
            ]
        )
        pipeline = build_pipeline(
            retriever_factory, prompt_builder, StaticGenerator(fabricated), min_chunks=1
        )
        service = AnalysisService(pipeline=pipeline, registry=registry, cache=None)

        result = service.analyze(
            resume=manifest(resume_id, DocumentType.RESUME, "resume.txt"),
            job=manifest(job_id, DocumentType.JOB_DESCRIPTION, "job.txt"),
        )
        assert any("do not match any retrieved passage" in w for w in result.warnings)

    def test_propagates_grounding_warnings(
        self, indexed_documents, retriever_factory, prompt_builder, registry
    ) -> None:
        resume_id, job_id = indexed_documents
        thin = valid_analysis(confidence=0.95)
        pipeline = build_pipeline(
            retriever_factory, prompt_builder, StaticGenerator(thin), min_chunks=1
        )
        service = AnalysisService(pipeline=pipeline, registry=registry, cache=None)

        result = service.analyze(
            resume=manifest(resume_id, DocumentType.RESUME, "resume.txt"),
            job=manifest(job_id, DocumentType.JOB_DESCRIPTION, "job.txt"),
        )
        assert any("High confidence" in warning for warning in result.warnings)

    def test_emits_progress_events(
        self, service: AnalysisService, indexed_documents
    ) -> None:
        resume_id, job_id = indexed_documents
        events: list[str] = []

        service.analyze(
            resume=manifest(resume_id, DocumentType.RESUME, "resume.txt"),
            job=manifest(job_id, DocumentType.JOB_DESCRIPTION, "job.txt"),
            on_event=lambda name, payload: events.append(name),
        )
        assert "retrieval" in events
        assert "generation" in events
