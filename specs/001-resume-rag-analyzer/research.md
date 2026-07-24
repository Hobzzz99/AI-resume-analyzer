# Phase 0 Research: Technical Decisions

**Feature**: 001-resume-rag-analyzer | **Date**: 2026-07-24

Each entry records the decision, the reasoning, and the alternatives that were rejected. These
are the decisions a reviewer is most likely to challenge, so each one is defended explicitly.

---

## R1. How does the engine stay domain-agnostic while the app stays typed?

**Decision**: Make `RAGPipeline` generic over a `TypeVar` bound to `BaseModel`. The pipeline owns
retrieve → build prompt → generate → parse. It is constructed with injected collaborators
(`Retriever`, `PromptBuilder`, `StructuredGenerator`) and a target schema type. The resume domain
supplies `ResumeAnalysis`, a resume-specific prompt template, and a retrieval plan.

**Rationale**: Generics give the domain layer full static typing (`pipeline.run(...) ->
ResumeAnalysis`) while the engine only ever sees `type[T]`. Constitution Principle I becomes
mechanically checkable: a test walks the AST of every module under `app/rag/` and asserts no
import from `app.services`, `app.prompts`, or resume-named symbols.

**Alternatives rejected**:
- *Inheritance (`ResumeRAGPipeline(RAGPipeline)`)* — makes the domain a subtype of the engine, so
  engine changes ripple into the domain and the "swap the schema" story requires a new class.
- *Untyped `dict` in/out* — kills IDE and mypy value at exactly the boundary where LLM output is
  least trustworthy.

---

## R2. Retrieval strategy: what actually goes into the prompt?

**Decision**: A **retrieval plan** — a list of named sub-queries, each with its own filter, top-k,
and strategy. The resume plan issues facet queries ("programming languages and frameworks",
"cloud and infrastructure", "machine learning and AI experience", "work experience and impact",
"education and certifications", "leadership and soft skills") against the resume, and
requirement-extraction queries against the job description. Results are merged, deduplicated by
chunk id, and budget-capped.

**Rationale**: A single query like "analyze this resume" retrieves generic prose. The analysis
schema has distinct facets; retrieving per facet is the difference between "top 5 chunks about
nothing in particular" and coverage of every field the schema asks for. It is also the honest
implementation of FR-027, which enumerates twelve assessment dimensions.

**Alternatives rejected**:
- *One query, large k* — cheaper, but coverage of low-frequency facets (certifications,
  leadership) collapses; those chunks never make the top-k against dense skill sections.
- *Sending the whole resume* — violates Principle II, and defeats the point of the project.

---

## R3. Hybrid search fusion method

**Decision**: BM25 (`rank_bm25`, in-process, built per collection from stored chunk texts) fused
with dense results by **Reciprocal Rank Fusion**, `score = Σ 1/(k + rank)` with `k = 60`.

**Rationale**: RRF needs no score normalisation, which matters because Chroma returns cosine
*distances* and BM25 returns unbounded term-frequency scores — min-max normalising two
incomparable scales produces a fused score that means nothing. RRF only needs rank order. `k=60`
is the value from the original TREC work and is the accepted default.

**Alternatives rejected**:
- *Weighted score blending* — requires normalisation, and the weight is a magic number tuned to
  one corpus.
- *A managed hybrid backend* — would pull in a service dependency and break the offline-test
  requirement (SC-009).

---

## R4. Reranking

**Decision**: Optional cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-6-v2`) behind a
config flag, default **off**. Retrieve `k * rerank_multiplier` candidates, rerank, keep top-k.

**Rationale**: Bi-encoder retrieval scores query and document independently; a cross-encoder reads
both together and is materially more accurate at the top of the list. The cost is a second model
download and ~100 ms of CPU per query batch. Defaulting it off keeps first-run friction low and
keeps CI offline; enabling it is one environment variable.

**Alternatives rejected**:
- *Always on* — forces every user through a second model download before their first analysis.
- *LLM-as-reranker* — an extra generation round-trip per sub-query on a rate-limited free tier.

---

## R5. Structured output: how to get valid JSON out of the model

**Decision**: Three layers, in order.
1. Request `response_format={"type": "json_object"}` from the provider (JSON mode).
2. Parse with a Pydantic parser that first extracts the outermost balanced JSON object (models
   still emit prose fences under load).
3. On `ValidationError`, retry up to `LLM_MAX_RETRIES` with a repair prompt that includes the raw
   output and the exact validation errors, with exponential backoff.

**Rationale**: JSON mode removes most malformed-syntax failures but not *semantic* failures —
a score of 140, `confidence: "high"`, a missing field. Only schema validation catches those, and
feeding the concrete `ValidationError` text back is far more effective than a generic "try again".
Constitution Principle III demands the response be contractual, and this is the enforcement.

**Alternatives rejected**:
- *`with_structured_output()` / tool-calling only* — convenient, but it hides the repair loop, and
  the failure mode the project needs to demonstrate is precisely *how you recover* from invalid
  output. Also varies in support across provider/model combinations.
- *Regex extraction into a dict* — no range enforcement, no type coercion, no error messages.

---

## R6. Model selection

**Decision**: `GROQ_MODEL` environment variable, no default in code — settings supply
`llama-3.3-70b-versatile` as the documented default value in `.env.example` only, and the code
reads whatever is configured. A `GET /health` response reports the resolved model.

**Rationale**: FR-021 and Principle VII. The Groq catalogue rotates; a hardcoded id makes the repo
stale on a schedule. Surfacing the resolved model in health output makes "which model produced
this?" answerable in production.

---

## R7. De-duplicating embedding work

**Decision**: Content fingerprint = SHA-256 of the normalised text plus the ingestion parameters
that affect chunking (`chunk_size`, `chunk_overlap`, splitter version). Fingerprint is the
`document_id`. Before ingesting, query the store for `document_id`; if chunks exist, return the
existing manifest and skip embedding entirely.

**Rationale**: FR-008 / SC-006. Hashing raw bytes would be wrong — the same resume re-exported
from Word produces different bytes but identical text, and re-embedding it is waste. Including
chunk parameters in the fingerprint is what prevents a stale index after someone changes
`CHUNK_SIZE` in `.env` and silently gets a mix of old and new chunk geometries.

**Alternatives rejected**:
- *Filename-keyed cache* — two different resumes named `resume.pdf` collide, which is the single
  most likely filename in this domain.

---

## R8. Semantic cache for analyses

**Decision**: Cache analysis results keyed by `(resume_document_id, job_document_id,
prompt_template_version, model)`. Because document ids are content fingerprints, this is exact
semantic identity at the document level — no embedding-similarity threshold needed. A separate
in-memory embedding-similarity cache sits in front of *sub-query retrieval*, keyed by the query
embedding with a cosine threshold, since facet queries repeat across every analysis.

**Rationale**: The naive "embed the request and compare with a 0.95 threshold" cache is a source
of subtly wrong results — two resumes for the same role are highly similar but must not share an
analysis. Using content fingerprints makes the document-level cache exact and therefore safe, and
confines fuzzy matching to the retrieval layer where a near-miss costs nothing but is still a hit.

---

## R9. Persistence layout

**Decision**: One persistent Chroma collection (`COLLECTION_NAME`), cosine space, with chunks from
all documents co-located and separated by metadata filters (`document_id`, `doc_type`). A JSON
manifest per document under `data/manifests/` records filename, page count, chunk count, and
timestamps.

**Rationale**: One collection keeps the BM25 index and the embedding function singular, and
metadata filtering is exactly what FR-013 asks for. A collection per document would make
cross-document retrieval (needed to compare resume against job description) require N queries and
manual merging. The manifest exists because Chroma is a poor place to ask "what documents do I
have?" — that is a `get()` over the whole collection.

---

## R10. Streaming, memory, and the API surface

**Decision**: `POST /analyze/stream` emits Server-Sent Events with stage progress
(`retrieval_started`, `chunks_retrieved`, `generation_started`, token deltas, `validated`) and a
terminal event carrying the validated analysis. `POST /chat` provides follow-up Q&A over already
ingested documents with windowed conversation memory (last N turns) held per session id.

**Rationale**: Streaming the *stages* is more honest than streaming tokens alone: the analysis is
only valid after parsing, so a client must not render partial JSON as a result. Emitting stage
events gives the UI something truthful to show during the 5–20 s generation. Conversation memory
is windowed rather than summarised to avoid a second LLM call per turn on a rate-limited tier.

---

## R11. Testing without a network

**Decision**: `Embedder`, `VectorStore`, `Retriever`, and `LLMClient` are `Protocol`s. The test
suite injects `HashingEmbedder` (deterministic, dependency-free), `InMemoryVectorStore`, and
`ScriptedLLMClient`. Tests touching Groq or downloading a model are marked `integration` and
deselected by default in `pyproject.toml`.

**Rationale**: SC-009 and Principle V. A test suite that needs an API key is a test suite that
does not run in CI, and one that downloads a 90 MB model is a test suite nobody runs twice. The
fakes are ~80 lines and are the payoff for the injection discipline.

---

## R12. Rejected scope

- **OCR fallback** for scanned PDFs: pulls in Tesseract or a paid vision API for a case the spec
  explicitly excludes. Detected and rejected with a clear message instead.
- **Async embedding workers / task queue**: correct for multi-tenant production, unjustified
  complexity for a single-node portfolio deployment. Ingestion is synchronous with a size cap.
- **Database for manifests**: JSON files are sufficient at this scale and keep setup to zero.
