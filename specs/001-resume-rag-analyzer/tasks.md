---
description: "Task list for 001-resume-rag-analyzer"
---

# Tasks: AI Resume Analyzer on a Reusable Retrieval Engine

**Input**: Design documents from `specs/001-resume-rag-analyzer/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md),
[data-model.md](./data-model.md), [contracts/api.md](./contracts/api.md)

**Tests**: Included — the spec mandates tests (FR-033 observability verification, SC-009 offline
suite) and the constitution makes them a quality gate.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: parallelisable (different files, no dependency)
- **[Story]**: owning user story (US1..US5)

---

## Phase 1: Setup

- [X] T001 Create the package tree from plan.md (`app/{api,config,rag,llm,parsers,prompts,schemas,services,utils,tests}`, `frontend/`, `scripts/`, `data/`) with `__init__.py` files
- [X] T002 [P] Author `requirements.txt` (fastapi, uvicorn, langchain, langchain-community, langchain-groq, langchain-huggingface, pydantic, pydantic-settings, chromadb, sentence-transformers, rank-bm25, pypdf, python-dotenv, streamlit, httpx, python-multipart, reportlab)
- [X] T003 [P] Author `requirements-dev.txt` (pytest, pytest-asyncio, pytest-cov, ruff, mypy)
- [X] T004 [P] Author `pyproject.toml` with ruff/mypy/pytest config, registering the `integration` marker and deselecting it by default
- [X] T005 [P] Author `.env.example` documenting every setting, and `.gitignore` covering `.env`, `data/`, `.venv`

---

## Phase 2: Foundational (blocks all stories)

- [X] T006 Implement `app/config/settings.py` — typed `Settings` (pydantic-settings) covering app, upload, chunking, embedding, vector store, retrieval, reranking, LLM, cache, chat, and logging groups; `GROQ_MODEL` required with no in-code default (FR-021, Principle VII)
- [X] T007 [P] Implement `app/utils/exceptions.py` — `AppError` base carrying `code`/`http_status`/`details`, plus the full taxonomy in contracts/api.md (FR-031)
- [X] T008 [P] Implement `app/utils/logging.py` — JSON formatter, `request_id` `ContextVar`, `configure_logging()`, `get_logger()` (FR-033, FR-034)
- [X] T009 [P] Implement `app/utils/timing.py` — `Stopwatch` context manager accumulating into `StageTimings` (FR-035)
- [X] T010 [P] Implement `app/utils/hashing.py` — content fingerprint over normalised text + chunk parameters (FR-008, R7)
- [X] T011 [P] Implement `app/utils/text.py` — whitespace collapse, hyphen/line-break repair, ligature and control-character normalisation (FR-003)
- [X] T012 Implement `app/schemas/rag.py` — `DocumentType`, `ChunkMetadata` (+ store round-trip), `Chunk`, `RetrievedChunk` (+ `citation`), `RetrievalStrategy`, `RetrievalPlanStep`, `RetrievalPlan`, `RetrievalTrace`, `DocumentManifest`, `StageTimings` (data-model.md)
- [X] T013 Implement `app/rag/base.py` — `Embedder`, `VectorStore`, `Retriever`, `LLMClient`, `AnalysisCache`, `PromptTemplateProvider` Protocols (Principle V)

**Checkpoint**: foundation ready.

---

## Phase 3: User Story 2 — Upload and prepare documents (P1)

**Goal**: a submitted document becomes indexed, retrievable passages exactly once.

**Independent Test**: upload → manifest with `chunk_count > 0`; re-upload identical content →
`cached: true`, no additional embedding.

### Tests

- [X] T014 [P] [US2] `app/tests/fakes.py` + `conftest.py` — `HashingEmbedder`, `InMemoryVectorStore`, `ScriptedLLMClient`, tmp-path settings fixture (R11, SC-009)
- [X] T015 [P] [US2] `app/tests/test_cleaner.py` — whitespace collapse, de-hyphenation, control chars, idempotence
- [X] T016 [P] [US2] `app/tests/test_loaders.py` — text loading, page attribution, unsupported type, empty document rejection
- [X] T017 [P] [US2] `app/tests/test_splitter.py` — chunk size/overlap honoured, metadata completeness, chunk-id stability
- [X] T018 [P] [US2] `app/tests/test_embeddings.py` — dimension, determinism, batch equals single, empty-input guard
- [X] T019 [P] [US2] `app/tests/test_vector_store.py` — add/query/filter/delete, metadata round-trip

### Implementation

- [X] T020 [P] [US2] `app/rag/cleaner.py` — `TextCleaner` (FR-003)
- [X] T021 [P] [US2] `app/rag/loaders.py` — `PDFLoader` (PyPDFLoader, page-attributed), `TextLoader`, `LoaderRegistry`, empty/corrupt detection (FR-001, FR-002, FR-004)
- [X] T022 [US2] `app/rag/splitter.py` — `DocumentSplitter` over `RecursiveCharacterTextSplitter`, emits `Chunk` with full `ChunkMetadata` (FR-006, FR-007)
- [X] T023 [P] [US2] `app/rag/embeddings.py` — `SentenceTransformerEmbedder` (lazy singleton) + `HashingEmbedder` (FR-008)
- [X] T024 [US2] `app/rag/vector_store.py` — `ChromaVectorStore` adapter + `InMemoryVectorStore` (FR-009, FR-013)
- [X] T025 [US2] `app/rag/ingestion.py` — `IngestionPipeline.ingest()`: load → clean → split → fingerprint → cache check → embed → store, stage-timed (FR-008, R7)
- [X] T026 [US2] `app/services/ingestion_service.py` — upload validation (type, size), manifest persistence, `DocumentManifest` assembly (FR-005)
- [X] T027 [US2] `app/tests/test_ingestion.py` — happy path, fingerprint cache hit, empty-document rejection, oversize rejection

**Checkpoint**: documents can be indexed and re-uploaded without re-embedding.

---

## Phase 4: User Story 1 — Analyze a resume against a job description (P1) 🎯 MVP

**Goal**: two indexed documents produce a validated, evidence-carrying analysis.

**Independent Test**: analyze two known documents; assert in-range scores, populated skill lists,
and evidence whose quotes appear in the retrieved passages.

### Tests

- [X] T028 [P] [US1] `app/tests/test_retriever.py` — similarity ordering, MMR diversity, BM25 lexical hit, RRF fusion, metadata filtering, top-k, score direction (FR-010..FR-015)
- [X] T029 [P] [US1] `app/tests/test_prompt_builder.py` — all required sections present, citations attached, context budget enforced, no full document path (FR-017..FR-020, SC-005)
- [X] T030 [P] [US1] `app/tests/test_parsers.py` — fenced-JSON extraction, prose-wrapped JSON, invalid JSON, range violation, repair-retry succeeds on second attempt, budget exhaustion raises (FR-023..FR-026)
- [X] T031 [P] [US1] `app/tests/test_analysis_schema.py` — range enforcement, `"Not Found"` normalisation, case-insensitive de-duplication, empty-evidence rejection, grounding warning
- [X] T032 [P] [US1] `app/tests/test_pipeline.py` — end-to-end with fakes, and the same pipeline driving a *different* schema with zero engine edits (SC-008)
- [X] T033 [P] [US1] `app/tests/test_architecture.py` — AST gate: no `app/rag/*` module imports `app.services`/`app.prompts`/`app.llm` or names a resume concept (Principle I, FR-036)

### Implementation

- [X] T034 [P] [US1] `app/rag/retriever.py` — `SimilarityRetriever`, `MMRRetriever`, `BM25Retriever`, `HybridRetriever` (RRF k=60), `RerankingRetriever` decorator, `RetrieverFactory` (R3, R4)
- [X] T035 [P] [US1] `app/schemas/analysis.py` — `EvidenceItem`, `ResumeAnalysis` with validators, `AnalysisResult` envelope (data-model.md)
- [X] T036 [P] [US1] `app/prompts/registry.py` + `templates/resume_analysis_v1.yaml`, `repair_v1.yaml` — versioned, file-backed templates (FR-018, FR-022)
- [X] T037 [US1] `app/rag/prompt_builder.py` — grouped, cited context blocks; budget guard raising `ContextBudgetExceeded` (FR-017, FR-019)
- [X] T038 [P] [US1] `app/parsers/json_extract.py` — balanced-brace extraction from noisy text
- [X] T039 [US1] `app/parsers/structured_parser.py` — `StructuredOutputParser[T]`: extract → validate → format repair instructions (FR-023, FR-025)
- [X] T040 [P] [US1] `app/llm/groq_client.py` — `GroqClient` implementing `LLMClient`, JSON mode, timeout, rate-limit/timeout error mapping (FR-021, R6)
- [X] T041 [US1] `app/llm/structured.py` — `StructuredGenerator[T]`: generate → parse → repair-retry with backoff, returns `(instance, retry_count)` (FR-025, R5)
- [X] T042 [US1] `app/rag/pipeline.py` — generic `RAGPipeline[T]`: execute plan → dedupe → budget → build prompt → generate → parse, emitting `RetrievalTrace` + `StageTimings` (FR-036)
- [X] T043 [US1] `app/services/cache.py` — fingerprint-keyed analysis cache + embedding-similarity query cache (R8)
- [X] T044 [US1] `app/services/analysis_service.py` — resume retrieval plan (12 assessment facets), pipeline invocation, `AnalysisResult` assembly (FR-027, FR-028, FR-029)
- [X] T045 [US1] `app/tests/test_services.py` — analysis service against fakes: cache hit, insufficient context, warning propagation

**Checkpoint**: MVP — the core analysis works end to end.

---

## Phase 5: User Story 4 — Operate the service (P2)

**Goal**: the service is inspectable and every request is traceable.

- [X] T046 [US4] `app/schemas/api.py` — request/response envelopes for upload, analyze, chat, health, errors (contracts/api.md)
- [X] T047 [US4] `app/api/dependencies.py` — composition root: cached singletons for settings, embedder, store, retriever, LLM, services (Principle V)
- [X] T048 [US4] `app/main.py` — app factory, lifespan warm-up, `RequestContextMiddleware` (request id + access log), unified `AppError` handler, CORS
- [X] T049 [P] [US4] `app/api/routes/health.py` — `/health` component report + `/health/live`
- [X] T050 [P] [US4] `app/api/routes/upload.py` — `/upload/resume`, `/upload/job` (file or pasted text, exactly one)
- [X] T051 [US4] `app/api/routes/analyze.py` — `/analyze` and SSE `/analyze/stream` (R10)
- [X] T052 [P] [US4] `app/api/routes/documents.py` — list / get / delete with index purge
- [X] T053 [US4] `app/tests/test_api.py` — every route with dependency overrides: happy paths and each error code

**Checkpoint**: service is operable and fully covered by route tests.

---

## Phase 6: User Story 3 — Inspect and trust the result (P2)

**Goal**: a dashboard that renders the analysis and its provenance.

- [X] T054a [US3] `frontend/` project scaffold — `package.json`, `tsconfig.json`, `next.config.mjs`, `tailwind.config.ts`, `postcss.config.mjs`, `.env.local.example`
- [X] T054b [US3] `frontend/lib/types.ts` — TypeScript mirrors of `DocumentManifest`, `ResumeAnalysis`, `AnalysisResult`, `RetrievalTrace`, `StageTimings`, error envelope
- [X] T054c [US3] `frontend/lib/api.ts` + `frontend/app/api/[...proxy]/route.ts` — typed client and server-side proxy that surfaces the error envelope as a typed `ApiError`
- [X] T055 [US3] `frontend/components/UploadPanel.tsx` — resume upload + job description (file or paste), with per-document manifest feedback
- [X] T056 [US3] `frontend/components/ScoreGauges.tsx`, `SkillLists.tsx`, `Findings.tsx` — score meters, matched/missing skills, strengths, weaknesses, recommendations
- [X] T057 [US3] `frontend/components/EvidenceTable.tsx` — evidence with citation, source, and quote
- [X] T058 [US3] `frontend/app/page.tsx`, `layout.tsx`, `globals.css`, `components/StageTrace.tsx` — flow (Upload → Analyze → Results), SSE progress, diagnostics panel (timings, retrieval trace, health)

**Checkpoint**: the full user journey is usable in a browser.

---

## Phase 7: User Story 5 — Reuse the engine (P3)

- [X] T059 [US5] `app/services/chat_service.py` + `app/api/routes/chat.py` + `templates/chat_qa_v1.yaml` — a second domain on the same engine, with windowed memory (R10)
- [X] T060 [US5] `scripts/evaluate.py` — retrieval evaluation harness: recall@k, MRR, per-strategy latency
- [X] T061 [US5] `app/tests/test_evaluation.py` — metric correctness on a fixed synthetic set

---

## Phase 8: Polish

- [X] T062 [P] `scripts/generate_sample_pdfs.py` — reportlab-generated sample resume and job description
- [X] T063 [P] `README.md` — architecture diagram, folder structure, RAG/LangChain/parser explanations, install, usage, design-decision rationale, future work
- [X] T064 [P] `data/eval/retrieval_cases.json` — evaluation fixtures
- [X] T065 Run the full suite offline and record the result

### T065 result (2026-07-24)

```
pytest app/tests            → 367 passed, 5 deselected (integration), 0 failed
ruff check app scripts      → All checks passed!
npx tsc --noEmit            → exit 0
npm run build               → compiled successfully, 4 routes
scripts/evaluate.py         → hybrid best: recall@5 0.867, MRR 0.833, MAP 0.786
scripts/generate_sample_pdfs.py → 2 PDFs + 2 TXT written to data/samples/
```

Run with `GROQ_API_KEY` unset and no network access, confirming SC-009.

Two real defects were found and fixed while writing the tests, both with permanent regression
coverage:

1. **`filename` in `extra=` crashes the logger.** It collides with a built-in `LogRecord`
   attribute, so `makeRecord` raises `KeyError` and takes down the request. Only appears once INFO
   logging is enabled. Fixed by `safe_extra()` plus renaming the key at every call site
   (`app/tests/test_logging.py`).
2. **The 422 handler was not JSON-serialisable.** A failing pydantic `model_validator` puts the
   raw `ValueError` into `ctx`, so returning `exc.errors()` verbatim turned a clean 422 into an
   opaque 500. Fixed by rebuilding the error list field by field
   (`app/tests/test_api.py::test_a_malformed_request_uses_the_same_error_envelope`).

---

## Dependencies & Execution Order

- **Setup (P1)** → **Foundational (P2)** blocks everything.
- **US2 (Phase 3)** must precede **US1 (Phase 4)** — analysis needs indexed documents. This is a
  real dependency, not a preference: US1's independent test presumes ingested content.
- **US4 (Phase 5)** depends on US1 + US2 services existing.
- **US3 (Phase 6)** depends on US4 (the client speaks HTTP only).
- **US5 (Phase 7)** depends on the engine (Phase 4) but not on US3/US4.
- **Polish (Phase 8)** last.

### Parallel opportunities

- T002–T005, T007–T011 — independent files.
- All test tasks marked [P] within a phase.
- T034, T035, T036, T038, T040 — independent modules in Phase 4.
- T049, T050, T052 — independent route modules.

## Implementation Strategy

MVP is Phase 1 → 2 → 3 → 4: a working, validated, evidence-grounded analysis reachable from
Python. Phase 5 makes it a service, Phase 6 makes it a product, Phase 7 proves the thesis.
