"""API tests.

Built with ``app.dependency_overrides``, which is the payoff for the composition
root: every route is exercised end to end against fakes, with no network, no API
key, and no model download.

Coverage is deliberately weighted towards failure paths. FR-031 promises a
distinct, meaningful response for each failure mode, and that promise is only
worth anything if each one is actually verified.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies as deps
from app.config.settings import Settings
from app.main import create_app
from app.schemas.analysis import ResumeAnalysis
from app.services.analysis_service import AnalysisService
from app.services.cache import AnalysisCache
from app.tests.conftest import build_pipeline
from app.tests.fakes import SAMPLE_JOB, SAMPLE_RESUME, StaticGenerator
from app.utils.exceptions import LLMRateLimitError, OutputValidationError

ANALYSIS = ResumeAnalysis.model_validate(
    {
        "overall_score": 78,
        "technical_score": 82,
        "experience_score": 74,
        "education_score": 85,
        "ats_score": 70,
        "matched_skills": ["Python", "PyTorch", "AWS"],
        "missing_skills": ["Kubernetes"],
        "strengths": ["Six years of production ML experience"],
        "weaknesses": ["No infrastructure-as-code experience evidenced"],
        "recommendations": ["Add a Kubernetes deployment to the projects section"],
        "recruiter_summary": "Strong applied ML candidate with directly relevant RAG experience.",
        "confidence": 0.78,
        "evidence": [
            {"claim": "PyTorch", "quote": "Frameworks: PyTorch", "citation": "[r.txt p.0 #0]",
             "source": "resume"}
        ],
    }
)


@pytest.fixture
def client(
    settings: Settings,
    ingestion_service,
    retriever_factory,
    prompt_builder,
    registry,
    store,
    embedder,
) -> TestClient:
    """An app wired entirely to fakes."""
    application = create_app(settings)

    pipeline = build_pipeline(
        retriever_factory, prompt_builder, StaticGenerator(ANALYSIS), min_chunks=1, top_k=2
    )
    analysis_service = AnalysisService(
        pipeline=pipeline, registry=registry, cache=AnalysisCache(), resume_top_k=2, job_top_k=2
    )

    application.dependency_overrides.update(
        {
            deps.get_settings: lambda: settings,
            deps.get_ingestion_service: lambda: ingestion_service,
            deps.get_analysis_service: lambda: analysis_service,
            deps.get_vector_store: lambda: store,
            deps.get_embedder: lambda: embedder,
            deps.get_prompt_registry: lambda: registry,
            deps.get_analysis_cache: lambda: AnalysisCache(),
        }
    )
    with TestClient(application) as test_client:
        yield test_client


def upload_documents(client: TestClient) -> tuple[str, str]:
    """Upload a resume and a job description; return their ids."""
    resume = client.post(
        "/api/v1/upload/resume",
        files={"file": ("resume.txt", SAMPLE_RESUME.encode(), "text/plain")},
    )
    job = client.post(
        "/api/v1/upload/job/text",
        json={"text": SAMPLE_JOB, "title": "Senior AI Engineer"},
    )
    return resume.json()["document_id"], job.json()["document_id"]


class TestHealth:
    def test_liveness_touches_nothing(self, client: TestClient) -> None:
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    def test_reports_the_resolved_configuration(self, client: TestClient) -> None:
        """FR-021 forbids a hardcoded model, so this is how it becomes knowable."""
        body = client.get("/api/v1/health").json()

        assert body["status"] in {"ok", "degraded"}
        assert body["llm"]["model"] == "test-model"
        assert body["embeddings"]["dimension"] == 64
        assert "resume_analysis_v1" in body["prompts"]["available"]
        assert body["retrieval"]["chunk_size"] == 300

    def test_stays_200_when_degraded(self, client: TestClient) -> None:
        """A degraded dependency must not remove the service from a load balancer."""
        assert client.get("/api/v1/health").status_code == 200

    def test_root_points_at_the_docs(self, client: TestClient) -> None:
        assert client.get("/").json()["docs"] == "/docs"


class TestUpload:
    def test_indexes_a_resume(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/upload/resume",
            files={"file": ("resume.txt", SAMPLE_RESUME.encode(), "text/plain")},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["chunk_count"] > 0
        assert body["doc_type"] == "resume"
        assert body["cached"] is False

    def test_a_repeat_upload_skips_embedding(self, client: TestClient) -> None:
        """SC-006, observable from outside the process."""
        files = {"file": ("resume.txt", SAMPLE_RESUME.encode(), "text/plain")}
        client.post("/api/v1/upload/resume", files=files)
        second = client.post("/api/v1/upload/resume", files=files).json()

        assert second["cached"] is True
        assert second["timings"]["embed_ms"] is None

    def test_accepts_a_pasted_job_description(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/upload/job/text", json={"text": SAMPLE_JOB, "title": "Senior AI Engineer"}
        )

        assert response.status_code == 201
        assert response.json()["doc_type"] == "job_description"
        assert response.json()["filename"] == "Senior AI Engineer.txt"

    def test_accepts_a_job_description_file(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/upload/job",
            files={"file": ("job.txt", SAMPLE_JOB.encode(), "text/plain")},
        )
        assert response.status_code == 201

    def test_rejects_an_unsupported_type(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/upload/resume",
            files={"file": ("resume.docx", b"binary", "application/octet-stream")},
        )

        assert response.status_code == 415
        assert response.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"

    def test_rejects_an_unreadable_document(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/upload/resume",
            files={"file": ("scan.txt", b"... 1 | 2 ... 3 4 5 -- 6 7 8 9", "text/plain")},
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "EMPTY_DOCUMENT"

    def test_rejects_empty_pasted_text(self, client: TestClient) -> None:
        response = client.post("/api/v1/upload/job/text", json={"text": "   "})
        assert response.status_code == 422


class TestAnalyze:
    def test_returns_a_validated_analysis(self, client: TestClient) -> None:
        resume_id, job_id = upload_documents(client)

        response = client.post(
            "/api/v1/analyze",
            json={"resume_document_id": resume_id, "job_document_id": job_id},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["analysis"]["overall_score"] == 78
        assert body["analysis"]["matched_skills"] == ["Python", "PyTorch", "AWS"]
        assert body["model"] == "static-model"

    def test_response_carries_stage_timings(self, client: TestClient) -> None:
        """FR-035."""
        resume_id, job_id = upload_documents(client)
        body = client.post(
            "/api/v1/analyze",
            json={"resume_document_id": resume_id, "job_document_id": job_id},
        ).json()

        assert body["timings"]["retrieve_ms"] is not None
        assert body["timings"]["total_ms"] is not None

    def test_response_carries_a_retrieval_trace(self, client: TestClient) -> None:
        resume_id, job_id = upload_documents(client)
        body = client.post(
            "/api/v1/analyze",
            json={"resume_document_id": resume_id, "job_document_id": job_id},
        ).json()

        assert body["retrieval"]["unique_chunks"] > 0
        assert len(body["retrieval"]["steps"]) == 16

    def test_evidence_is_exposed_in_both_shapes(self, client: TestClient) -> None:
        resume_id, job_id = upload_documents(client)
        analysis = client.post(
            "/api/v1/analyze",
            json={"resume_document_id": resume_id, "job_document_id": job_id},
        ).json()["analysis"]

        assert analysis["evidence"][0]["citation"]
        assert analysis["evidence_strings"]

    def test_unknown_document_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={"resume_document_id": "nope", "job_document_id": "also-nope"},
        )

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    def test_rejects_analysing_a_document_against_itself(self, client: TestClient) -> None:
        """Same fingerprint means the same document, and would score a near-perfect match."""
        response = client.post(
            "/api/v1/analyze",
            json={"resume_document_id": "same", "job_document_id": "same"},
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "REQUEST_VALIDATION_ERROR"

    def test_a_malformed_request_uses_the_same_error_envelope(
        self, client: TestClient
    ) -> None:
        """Two error shapes would force clients to write two error handlers."""
        body = client.post("/api/v1/analyze", json={"resume_document_id": "only-one"}).json()
        assert "error" in body
        assert body["error"]["code"] == "REQUEST_VALIDATION_ERROR"

    def test_provider_rate_limits_surface_as_429(
        self, settings, ingestion_service, registry, store, embedder
    ) -> None:
        """Never a partial analysis; always an explicit upstream failure."""

        class RateLimited:
            model_name = "rate-limited"

            def generate(self, prompt, schema, *, system=None):  # type: ignore[no-untyped-def]
                raise LLMRateLimitError("slow down", details={"retry_after": 12})

        application = create_app(settings)
        service = AnalysisService(
            pipeline=build_pipeline(
                __import__("app.rag.retriever", fromlist=["RetrieverFactory"]).RetrieverFactory(
                    store, embedder
                ),
                __import__("app.rag.prompt_builder", fromlist=["PromptBuilder"]).PromptBuilder(),
                RateLimited(),
                min_chunks=0,
            ),
            registry=registry,
            cache=None,
        )
        application.dependency_overrides.update(
            {
                deps.get_settings: lambda: settings,
                deps.get_ingestion_service: lambda: ingestion_service,
                deps.get_analysis_service: lambda: service,
                deps.get_vector_store: lambda: store,
                deps.get_embedder: lambda: embedder,
                deps.get_prompt_registry: lambda: registry,
            }
        )

        with TestClient(application) as test_client:
            resume_id, job_id = upload_documents(test_client)
            response = test_client.post(
                "/api/v1/analyze",
                json={"resume_document_id": resume_id, "job_document_id": job_id},
            )

        assert response.status_code == 429
        assert response.json()["error"]["code"] == "LLM_RATE_LIMITED"

    def test_validation_exhaustion_surfaces_as_422(
        self, settings, ingestion_service, registry, store, embedder
    ) -> None:
        class AlwaysInvalid:
            model_name = "invalid"

            def generate(self, prompt, schema, *, system=None):  # type: ignore[no-untyped-def]
                raise OutputValidationError(
                    "could not produce valid output", details={"attempts": 3}
                )

        from app.rag.prompt_builder import PromptBuilder
        from app.rag.retriever import RetrieverFactory

        application = create_app(settings)
        service = AnalysisService(
            pipeline=build_pipeline(
                RetrieverFactory(store, embedder), PromptBuilder(), AlwaysInvalid(), min_chunks=0
            ),
            registry=registry,
            cache=None,
        )
        application.dependency_overrides.update(
            {
                deps.get_settings: lambda: settings,
                deps.get_ingestion_service: lambda: ingestion_service,
                deps.get_analysis_service: lambda: service,
                deps.get_vector_store: lambda: store,
                deps.get_embedder: lambda: embedder,
                deps.get_prompt_registry: lambda: registry,
            }
        )

        with TestClient(application) as test_client:
            resume_id, job_id = upload_documents(test_client)
            response = test_client.post(
                "/api/v1/analyze",
                json={"resume_document_id": resume_id, "job_document_id": job_id},
            )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "OUTPUT_VALIDATION_FAILED"


class TestAnalyzeStream:
    def test_streams_stage_events_then_a_result(self, client: TestClient) -> None:
        resume_id, job_id = upload_documents(client)

        with client.stream(
            "POST",
            "/api/v1/analyze/stream",
            json={"resume_document_id": resume_id, "job_document_id": job_id},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

        assert "event: stage" in body
        assert "event: result" in body
        assert "event: done" in body
        assert body.index("event: stage") < body.index("event: result")

    def test_streams_an_error_event_on_failure(self, client: TestClient) -> None:
        with client.stream(
            "POST",
            "/api/v1/analyze/stream",
            json={"resume_document_id": "missing", "job_document_id": "other-missing"},
        ) as response:
            body = "".join(response.iter_text())

        assert "event: error" in body
        assert "DOCUMENT_NOT_FOUND" in body


class TestDocuments:
    def test_lists_uploaded_documents(self, client: TestClient) -> None:
        upload_documents(client)
        body = client.get("/api/v1/documents").json()
        assert len(body) == 2

    def test_fetches_one_document(self, client: TestClient) -> None:
        resume_id, _ = upload_documents(client)
        assert client.get(f"/api/v1/documents/{resume_id}").json()["document_id"] == resume_id

    def test_unknown_document_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/documents/nope")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    def test_delete_purges_the_index(self, client: TestClient) -> None:
        resume_id, _ = upload_documents(client)

        response = client.delete(f"/api/v1/documents/{resume_id}")

        assert response.status_code == 200
        assert response.json()["chunks_removed"] > 0
        assert client.get(f"/api/v1/documents/{resume_id}").status_code == 404


class TestRequestCorrelation:
    def test_every_response_carries_a_request_id(self, client: TestClient) -> None:
        assert client.get("/api/v1/health").headers["X-Request-ID"]

    def test_an_inbound_request_id_is_honoured(self, client: TestClient) -> None:
        """Lets a trace span a proxy or a calling service."""
        response = client.get("/api/v1/health", headers={"X-Request-ID": "trace-me"})
        assert response.headers["X-Request-ID"] == "trace-me"

    def test_errors_carry_the_request_id_too(self, client: TestClient) -> None:
        body = client.get("/api/v1/documents/nope").json()
        assert body["error"]["request_id"]
