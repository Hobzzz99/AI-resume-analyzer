# Phase 1 Data Model

**Feature**: 001-resume-rag-analyzer | **Date**: 2026-07-24

All models are Pydantic v2. Engine models live in `app/schemas/rag.py` and carry no domain
meaning; domain models live in `app/schemas/analysis.py`.

---

## Engine models (`app/schemas/rag.py`)

### `DocumentType` (str enum)
`resume` | `job_description` | `generic`

The engine only needs *a* label to filter on; the specific members are a pragmatic concession —
`generic` is what a non-resume adopter uses. No engine logic branches on the value.

### `ChunkMetadata`
| Field | Type | Notes |
|---|---|---|
| `document_id` | `str` | Content fingerprint of the owning document |
| `filename` | `str` | Original filename, or a synthetic name for pasted text |
| `doc_type` | `DocumentType` | Used as the primary retrieval filter |
| `page` | `int` | 1-based; `0` when the source has no pagination |
| `chunk_index` | `int` | 0-based position within the document |
| `chunk_id` | `str` | `f"{document_id}:{page}:{chunk_index}"` — stable and human-readable |
| `ingested_at` | `datetime` | UTC |
| `char_count` | `int` | Used for prompt budget accounting |

Chroma metadata values must be primitives, so `to_store()` / `from_store()` handle the
enum→str and datetime→ISO-string round trip in one place rather than at every call site.

### `Chunk`
`text: str` + `metadata: ChunkMetadata`. Immutable (`model_config = ConfigDict(frozen=True)`).

### `RetrievedChunk`
`chunk: Chunk`, `score: float`, `rank: int`, `retriever: str`.

`score` is normalised to "higher is better" at the retriever boundary — Chroma returns cosine
*distance*, so the similarity retriever converts once, and no downstream code needs to know which
direction the backend's numbers run.

`citation` (computed property) → `[filename p.{page} #{chunk_index}]`, the handle placed in the
prompt and the thing the model is instructed to cite.

### `RetrievalPlanStep` / `RetrievalPlan`
| Field | Type | Notes |
|---|---|---|
| `name` | `str` | Facet label, appears in the prompt as a section header |
| `query` | `str` | Sub-query text |
| `doc_type` | `DocumentType \| None` | Metadata filter |
| `document_id` | `str \| None` | Metadata filter |
| `top_k` | `int \| None` | Falls back to settings |
| `strategy` | `RetrievalStrategy \| None` | Falls back to settings |

A plan is data, not code — which is what lets a different domain supply a different plan without
touching the engine.

### `RetrievalTrace`
Per-request diagnostics: `steps` (name, query, candidates considered, returned, duration_ms),
`total_chunks`, `unique_chunks`, `deduplicated`, `budget_truncated: bool`. Returned in the API
response and logged. This is the artefact that makes "why did it say that?" answerable.

### `DocumentManifest`
`document_id`, `filename`, `doc_type`, `page_count`, `chunk_count`, `char_count`,
`ingested_at`, `cached: bool`. Persisted as JSON under `data/manifests/{document_id}.json` and
returned from upload endpoints.

### `StageTimings`
`load_ms`, `clean_ms`, `split_ms`, `embed_ms`, `store_ms`, `retrieve_ms`, `rerank_ms`,
`prompt_ms`, `llm_ms`, `parse_ms`, `total_ms` — all `float | None`. Populated by the `Stopwatch`
context manager in `app/utils/timing.py`.

---

## Domain models (`app/schemas/analysis.py`)

### `EvidenceItem`
| Field | Type | Constraint |
|---|---|---|
| `claim` | `str` | The conclusion being supported |
| `quote` | `str` | Verbatim text from a retrieved passage |
| `citation` | `str` | Must match a citation handle that was placed in the prompt |
| `source` | `DocumentType` | Which document it came from |

A structured evidence item rather than a bare string. The spec's example schema uses
`list[str]`; that makes SC-004 ("evidence resolves to a real passage") unverifiable. The API
response therefore exposes **both**: `evidence` as `list[EvidenceItem]` for the UI, and
`evidence_strings` as the flattened `list[str]` for schema-compatibility with the brief.

### `ResumeAnalysis`
| Field | Type | Constraint |
|---|---|---|
| `overall_score` | `int` | 0–100 |
| `technical_score` | `int` | 0–100 |
| `experience_score` | `int` | 0–100 |
| `education_score` | `int` | 0–100 |
| `ats_score` | `int` | 0–100 |
| `matched_skills` | `list[str]` | ≤ 40 items, de-duplicated case-insensitively |
| `missing_skills` | `list[str]` | ≤ 40 items |
| `strengths` | `list[str]` | 1–10 items |
| `weaknesses` | `list[str]` | ≤ 10 items |
| `recommendations` | `list[str]` | 1–10 items |
| `recruiter_summary` | `str` | 40–1500 chars |
| `confidence` | `float` | 0.0–1.0 |
| `evidence` | `list[EvidenceItem]` | ≥ 1 item unless every score is 0 |

Validators:
- `_strip_and_dedupe` (list fields): trims, drops empties, removes case-insensitive duplicates
  while preserving order. Models routinely emit `["Python", "python"]`.
- `_not_found_is_empty`: the literal `"Not Found"` in a list field means "nothing here" and is
  normalised to an empty list, so the UI does not render a skill called "Not Found".
- `_evidence_required` (model validator): a non-zero `overall_score` with empty `evidence`
  violates Principle IV — rejected, which triggers the repair retry.
- `_confidence_consistency` (model validator): does not reject, but records a
  `grounding_warning` when confidence > 0.8 with fewer than 3 evidence items.

### `AnalysisResult`
The API envelope: `analysis: ResumeAnalysis`, `resume: DocumentManifest`,
`job: DocumentManifest`, `retrieval: RetrievalTrace`, `timings: StageTimings`, `model: str`,
`prompt_template: str`, `request_id: str`, `cached: bool`, `warnings: list[str]`,
`retry_count: int`.

The analysis is never returned bare. Provenance travels with it — that is what makes the result
auditable rather than merely plausible.

### `ChatTurn` / `ChatResponse`
`role`, `content`, `timestamp`; response adds `citations: list[str]` and `retrieval:
RetrievalTrace`.

---

## Persistence

**Chroma collection** (`resume_rag_chunks`, cosine space):
- `ids` ← `chunk_id`
- `documents` ← chunk text
- `embeddings` ← 384-dim MiniLM vectors
- `metadatas` ← flattened `ChunkMetadata`

**Filesystem**:
```
data/
  chroma/                     # Chroma persistent client
  manifests/{document_id}.json
  uploads/{document_id}{ext}  # retained for citation display; purgeable
  cache/analyses.json         # analysis cache index
```

---

## State transitions

```
uploaded ──validate──> extracted ──clean──> split ──embed──> indexed
    │                      │                              │
    └─ rejected            └─ rejected (no text)          └─ cached (fingerprint hit)

indexed(resume) + indexed(job) ──retrieve──> grounded ──generate──> raw
raw ──parse──> validated                    raw ──invalid──> repair ──(≤N)──> validated | failed
```

Rejection is terminal and leaves no partial index — ingestion writes to the store only after the
full chunk list is built.
