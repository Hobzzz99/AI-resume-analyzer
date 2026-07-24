# Implementation Plan: AI Resume Analyzer on a Reusable Retrieval Engine

**Branch**: `001-resume-rag-analyzer` | **Date**: 2026-07-24 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-resume-rag-analyzer/spec.md`

## Summary

Deliver an AI Resume Analyzer as a thin domain layer over a general-purpose, injectable RAG
engine. Documents are extracted, normalised, chunked, embedded once (content-fingerprint
de-duplicated), and persisted in a metadata-rich Chroma collection. Analysis runs a **retrieval
plan** — a set of facet sub-queries against the resume and requirement queries against the job
description — fuses dense and lexical results by Reciprocal Rank Fusion, optionally reranks with
a cross-encoder, assembles a citation-carrying prompt from the retrieved passages only, generates
with a configurable Groq model in JSON mode, and validates the response against a Pydantic v2
schema with a bounded repair-retry loop. The result is returned with full provenance: which
passages were used, how long each stage took, which model produced it.

The engine (`app/rag/`) is generic over the answer schema and knows nothing about resumes; the
resume domain supplies a schema, a prompt template, and a retrieval plan. A static test enforces
that separation.

## Technical Context

**Language/Version**: Python 3.12+ (developed on 3.13)

**Primary Dependencies**: FastAPI + Uvicorn (service), LangChain / langchain-community /
langchain-groq (loaders, splitter, prompt templates, chat model), Pydantic v2 +
pydantic-settings (schemas, config), ChromaDB (vector persistence),
sentence-transformers `all-MiniLM-L6-v2` (embeddings), `rank_bm25` (lexical retrieval),
pytest (tests). Frontend: Next.js 15 (App Router) + TypeScript + Tailwind CSS.

**Storage**: Local filesystem — Chroma persistent client at `data/chroma/`, JSON manifests at
`data/manifests/`, uploaded originals at `data/uploads/`

**Testing**: pytest with injected fakes (`HashingEmbedder`, `InMemoryVectorStore`,
`ScriptedLLMClient`); default run requires no network and no API key. Provider- and
model-download-dependent tests marked `@pytest.mark.integration` and deselected by default.

**Target Platform**: Linux/macOS/Windows, single node, CPU-only

**Project Type**: Web service (FastAPI backend) + separate Next.js client (Node 20+)

**Performance Goals**: Ingestion of a 2-page resume < 5 s warm (SC-002); end-to-end analysis
< 30 s (SC-001), of which retrieval < 500 ms and the remainder is provider generation

**Constraints**: Prompt context bounded by `MAX_CONTEXT_CHUNKS` × `CHUNK_SIZE` and never the full
document (SC-005); free-tier provider rate limits must surface as explicit errors, not retries to
exhaustion; unit suite must run offline (SC-009)

**Scale/Scope**: Single user, tens of documents, thousands of chunks — well inside Chroma's
in-process envelope. ~40 source modules, ~10 test modules.

## Constitution Check

*GATE: evaluated before Phase 0 and re-evaluated after Phase 1 design.*

| Principle | Gate | Design compliance |
|---|---|---|
| I. Engine/Application separation | No module in `app/rag/` imports a domain module or names a domain concept | Pipeline is generic over `T: BaseModel`; retrieval plans and prompt templates are injected data. `tests/test_architecture.py` walks the AST of `app/rag/*` and asserts zero imports from `app.services`/`app.prompts` and zero resume-domain identifiers. ✅ |
| II. Retrieval mandatory | Full documents never reach the LLM | `PromptBuilder` accepts `list[RetrievedChunk]` only — there is no code path from a `Document` to a prompt. Builder raises `ContextBudgetExceeded` above the configured budget. ✅ |
| III. Structured output contractual | Every model response validated before crossing a boundary | `StructuredOutputParser` returns `T` or raises; `RAGPipeline.run()` is typed `-> T`. Routers receive only validated instances. ✅ |
| IV. Grounding over fluency | Prompt forbids fabrication; evidence required | Template mandates `Not Found` and per-claim citations; `ResumeAnalysis` model validator rejects a non-zero score with empty evidence, which triggers repair. ✅ |
| V. Dependency injection | Protocols for every collaborator; no import-time clients | `app/rag/base.py` defines `Embedder`, `VectorStore`, `Retriever`, `LLMClient`, `AnalysisCache` Protocols. Construction happens in `app/api/dependencies.py` behind `@lru_cache`; nothing is built at import. ✅ |
| VI. Observability | Per-stage timing + correlation id in logs and responses | `Stopwatch` populates `StageTimings`; `RequestContextMiddleware` binds a request id into a `ContextVar` consumed by the JSON log formatter; `StageTimings` and `RetrievalTrace` are in every analysis response. ✅ |
| VII. Configuration over constants | No hardcoded model ids or tuning constants | Single `Settings` (pydantic-settings). `GROQ_MODEL` has no in-code default and is documented in `.env.example`; `/health` reports the resolved value. ✅ |

**Result**: PASS, no violations. Complexity Tracking table is empty.

## Project Structure

### Documentation (this feature)

```text
specs/001-resume-rag-analyzer/
├── plan.md              # This file
├── spec.md              # Phase 2 output
├── research.md          # Phase 0 output — R1..R12 decisions
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── api.md           # Phase 1 output — endpoints + error taxonomy
├── checklists/
│   └── requirements.md  # Spec quality gate
└── tasks.md             # Phase 5 output
```

### Source Code (repository root)

```text
app/
├── main.py                     # App factory, middleware, exception handlers, lifespan
├── api/
│   ├── dependencies.py         # Composition root — the only place objects are wired
│   └── routes/
│       ├── health.py
│       ├── upload.py
│       ├── analyze.py          # /analyze and /analyze/stream
│       ├── chat.py
│       └── documents.py
├── config/
│   └── settings.py             # Typed settings, single source of configuration
├── rag/                        # DOMAIN-FREE ENGINE
│   ├── base.py                 # Protocols: Embedder, VectorStore, Retriever, LLMClient, Cache
│   ├── loaders.py              # PDF/text loading + registry, page attribution
│   ├── cleaner.py              # Whitespace/ligature/control-char normalisation
│   ├── splitter.py             # RecursiveCharacterTextSplitter wrapper → Chunk[]
│   ├── embeddings.py           # MiniLM embedder (lazy, cached) + HashingEmbedder fake
│   ├── vector_store.py         # Chroma adapter + InMemoryVectorStore fake
│   ├── retriever.py            # Similarity, MMR, BM25, hybrid (RRF), reranker decorator
│   ├── prompt_builder.py       # Retrieved chunks → prompt; enforces context budget
│   ├── pipeline.py             # Generic RAGPipeline[T]: retrieve→prompt→generate→parse
│   └── ingestion.py            # load→clean→split→embed→store, fingerprint de-dup
├── llm/
│   ├── groq_client.py          # ChatGroq adapter implementing LLMClient
│   └── structured.py           # JSON-mode generation + repair-retry orchestration
├── parsers/
│   ├── json_extract.py         # Balanced-brace extraction from noisy model text
│   └── structured_parser.py    # Pydantic validation + repair-prompt construction
├── prompts/
│   ├── registry.py             # Loads templates from YAML; versioned by name
│   └── templates/
│       ├── resume_analysis_v1.yaml
│       ├── chat_qa_v1.yaml
│       └── repair_v1.yaml
├── schemas/
│   ├── rag.py                  # Chunk, RetrievedChunk, RetrievalPlan, traces, timings
│   ├── analysis.py             # ResumeAnalysis, EvidenceItem, AnalysisResult
│   └── api.py                  # Request/response envelopes
├── services/
│   ├── ingestion_service.py    # Upload orchestration + manifest persistence
│   ├── analysis_service.py     # Resume domain: plan → pipeline → result envelope
│   ├── chat_service.py         # Conversation memory + grounded Q&A
│   └── cache.py                # Analysis cache (fingerprint-keyed) + query cache
├── utils/
│   ├── exceptions.py           # AppError hierarchy ↔ HTTP status mapping
│   ├── logging.py              # JSON formatter, request-id ContextVar
│   ├── timing.py               # Stopwatch context manager
│   ├── text.py                 # Normalisation helpers
│   └── hashing.py              # Content fingerprints
└── tests/
    ├── conftest.py             # Fakes and fixtures
    ├── fakes.py
    ├── test_architecture.py    # Principle I enforcement
    ├── test_loaders.py  test_cleaner.py  test_splitter.py
    ├── test_embeddings.py  test_vector_store.py  test_retriever.py
    ├── test_prompt_builder.py  test_parsers.py  test_pipeline.py
    ├── test_analysis_schema.py  test_services.py  test_api.py
    └── test_evaluation.py

frontend/                       # Next.js 15, App Router, TypeScript, Tailwind
├── app/
│   ├── layout.tsx  page.tsx  globals.css
│   └── api/[...proxy]/route.ts # Server-side proxy to FastAPI (keeps the API origin private)
├── components/                 # UploadPanel, ScoreGauges, SkillLists, EvidenceTable, StageTrace
├── lib/
│   ├── api.ts                  # Typed fetch client
│   └── types.ts                # TypeScript mirrors of the Pydantic response models
├── package.json  tsconfig.json  next.config.mjs  tailwind.config.ts  postcss.config.mjs
└── .env.local.example

scripts/
├── generate_sample_pdfs.py     # Produces data/samples/*.pdf
└── evaluate.py                 # Retrieval + end-to-end evaluation harness

data/samples/                   # Sample resume + job description
README.md  requirements.txt  requirements-dev.txt  .env.example  pyproject.toml
```

**Structure Decision**: Web-service layout with a separate thin client. The backend is a single
`app/` package rather than `backend/src/` because the Next.js client is a consumer of the HTTP
API, not a co-equal application — it holds no business logic and shares no code with `app/`.
Within `app/`, the hard boundary is `app/rag/` (engine) versus everything else (application), and
that boundary — not the FastAPI layering — is the one the constitution polices.

The client is a separate process in its own language, which enforces that boundary physically:
there is no import path by which UI concerns could reach into the analysis pipeline. Server-side
route handlers in `app/api/[...proxy]` forward browser requests to FastAPI, so the API origin and
any future credentials stay server-side and the browser talks to one origin (no CORS preflight on
file uploads).

## Implementation Phases

- **Phase A — Foundation**: settings, exceptions, logging, timing, hashing, engine schemas.
  Blocks everything.
- **Phase B — Engine**: loaders → cleaner → splitter → embeddings → vector store → retrievers →
  prompt builder → pipeline → ingestion. Each with tests against fakes.
- **Phase C — Domain**: analysis schema, prompt registry and templates, Groq client, structured
  generation with repair, ingestion/analysis/chat services, caches.
- **Phase D — API**: composition root, routes, middleware, exception handlers, streaming.
- **Phase E — Client**: Next.js upload → analyze → dashboard, with SSE progress.
- **Phase F — Proof**: full test suite, sample PDFs, evaluation harness, README.

## Complexity Tracking

No Constitution Check violations. Table intentionally empty.
