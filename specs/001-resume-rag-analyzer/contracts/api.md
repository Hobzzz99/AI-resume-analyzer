# API Contract

**Feature**: 001-resume-rag-analyzer | Base path: `/api/v1` | Version: 1.0.0

All errors share one envelope, produced by a single exception handler:

```json
{
  "error": {
    "code": "EMPTY_DOCUMENT",
    "message": "No extractable text found in 'scan.pdf'. The file appears to be image-only; OCR is not supported.",
    "details": {"filename": "scan.pdf", "pages": 3},
    "request_id": "0f9c1a2b-..."
  }
}
```

| Code | HTTP | Raised when |
|---|---|---|
| `UNSUPPORTED_FILE_TYPE` | 415 | Extension/content-type not in the allowed set |
| `FILE_TOO_LARGE` | 413 | Upload exceeds `MAX_UPLOAD_MB` |
| `INVALID_DOCUMENT` | 422 | PDF unreadable, corrupt, or encrypted |
| `EMPTY_DOCUMENT` | 422 | Parsed successfully but yielded no usable text |
| `DOCUMENT_NOT_FOUND` | 404 | Unknown `document_id` on analyze/chat |
| `EMBEDDING_FAILED` | 503 | Embedding model failed to load or encode |
| `VECTOR_STORE_ERROR` | 503 | Index unreachable or write failed |
| `INSUFFICIENT_CONTEXT` | 422 | Retrieval returned fewer than the minimum required chunks |
| `LLM_ERROR` | 502 | Provider returned an error or was unreachable |
| `LLM_RATE_LIMITED` | 429 | Provider rate limit; includes `retry_after` in details |
| `LLM_TIMEOUT` | 504 | Generation exceeded `LLM_TIMEOUT_SECONDS` |
| `OUTPUT_VALIDATION_FAILED` | 422 | Repair budget exhausted; includes last validation errors |
| `CONFIGURATION_ERROR` | 500 | Missing/invalid required setting (e.g. no API key) |

Every response carries an `X-Request-ID` header; it is also the `request_id` in bodies.

---

## `GET /api/v1/health`

**200**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "llm": {"provider": "groq", "model": "llama-3.3-70b-versatile", "configured": true},
  "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2", "dimension": 384, "loaded": true},
  "vector_store": {"backend": "chroma", "collection": "resume_rag_chunks", "chunk_count": 128, "reachable": true},
  "retrieval": {"strategy": "hybrid", "top_k": 5, "reranking": false},
  "uptime_seconds": 412.5
}
```
`status` is `degraded` when any component reports unreachable/unconfigured; still **200**, since
the endpoint itself is healthy. Liveness probes should use `/api/v1/health/live`, which returns
`{"status":"alive"}` without touching dependencies.

---

## `POST /api/v1/upload/resume`

`multipart/form-data`: `file` (required, `.pdf` | `.txt` | `.md`).

**201** → `DocumentManifest`
```json
{
  "document_id": "a3f1c9e28b7d4051",
  "filename": "jane_doe_resume.pdf",
  "doc_type": "resume",
  "page_count": 2,
  "chunk_count": 11,
  "char_count": 6218,
  "ingested_at": "2026-07-24T10:15:03.412Z",
  "cached": false,
  "timings": {"load_ms": 88.1, "clean_ms": 1.9, "split_ms": 3.2, "embed_ms": 412.7, "store_ms": 22.4, "total_ms": 528.3}
}
```
`cached: true` with `embed_ms: null` when the fingerprint was already indexed (SC-006).

Errors: `UNSUPPORTED_FILE_TYPE`, `FILE_TOO_LARGE`, `INVALID_DOCUMENT`, `EMPTY_DOCUMENT`,
`EMBEDDING_FAILED`, `VECTOR_STORE_ERROR`.

---

## `POST /api/v1/upload/job` · `POST /api/v1/upload/job/text`

Two operations rather than one content-type-switching endpoint, because FastAPI cannot express
"multipart or JSON" on a single operation without losing generated schema for both. The ingestion
path is shared from cleaning onward, so behaviour cannot diverge.

`POST /upload/job` — `multipart/form-data`: `file` (required), `title` (optional form field).

`POST /upload/job/text` — `application/json`:
```json
{"text": "We are hiring a Senior ML Engineer ...", "title": "Senior ML Engineer"}
```
Pasted text is given the synthetic filename `{title|job_description}.txt` and `page: 0`.

**201** → `DocumentManifest` with `doc_type: "job_description"`.

---

## `POST /api/v1/analyze`

```json
{
  "resume_document_id": "a3f1c9e28b7d4051",
  "job_document_id": "77b2e01c4da9f8e3",
  "top_k": 5,
  "strategy": "hybrid",
  "use_reranker": false,
  "prompt_template": "resume_analysis_v1",
  "use_cache": true
}
```
All fields except the two ids are optional and fall back to configured defaults.

**200** → `AnalysisResult`
```json
{
  "request_id": "0f9c1a2b-...",
  "cached": false,
  "model": "llama-3.3-70b-versatile",
  "prompt_template": "resume_analysis_v1",
  "retry_count": 0,
  "warnings": [],
  "analysis": {
    "overall_score": 78, "technical_score": 82, "experience_score": 71,
    "education_score": 85, "ats_score": 74,
    "matched_skills": ["Python", "PyTorch", "Docker", "AWS"],
    "missing_skills": ["Kubernetes", "Terraform"],
    "strengths": ["Three production ML deployments with measured impact"],
    "weaknesses": ["No infrastructure-as-code experience evidenced"],
    "recommendations": ["Add a Kubernetes deployment to the MLOps project section"],
    "recruiter_summary": "Strong applied-ML candidate ...",
    "confidence": 0.81,
    "evidence": [
      {"claim": "Matched: PyTorch", "quote": "Built and trained a transformer classifier in PyTorch",
       "citation": "[jane_doe_resume.pdf p.1 #3]", "source": "resume"}
    ],
    "evidence_strings": ["[jane_doe_resume.pdf p.1 #3] Built and trained a transformer classifier in PyTorch"]
  },
  "resume": { "...DocumentManifest..." },
  "job": { "...DocumentManifest..." },
  "retrieval": {
    "steps": [{"name": "programming_languages", "query": "...", "candidates": 20, "returned": 5, "duration_ms": 31.2}],
    "total_chunks": 30, "unique_chunks": 14, "deduplicated": 16, "budget_truncated": false
  },
  "timings": {"retrieve_ms": 190.4, "prompt_ms": 2.1, "llm_ms": 6120.7, "parse_ms": 4.0, "total_ms": 6320.9}
}
```

Errors: `DOCUMENT_NOT_FOUND`, `INSUFFICIENT_CONTEXT`, `LLM_ERROR`, `LLM_RATE_LIMITED`,
`LLM_TIMEOUT`, `OUTPUT_VALIDATION_FAILED`, `CONFIGURATION_ERROR`.

---

## `POST /api/v1/analyze/stream`

Same request body. Response `text/event-stream`:

```
event: stage
data: {"stage":"retrieval","status":"started"}

event: stage
data: {"stage":"retrieval","status":"completed","unique_chunks":14,"duration_ms":190.4}

event: token
data: {"delta":"{\"overall_score\":"}

event: stage
data: {"stage":"validation","status":"completed","retry_count":0}

event: result
data: { ...AnalysisResult... }

event: done
data: {}
```
An `event: error` carrying the standard error envelope terminates the stream on failure. Clients
MUST NOT render `token` payloads as a result — only the `result` event is validated.

---

## `POST /api/v1/chat`

```json
{
  "session_id": "s-91ac",
  "message": "Does this candidate have Kubernetes experience?",
  "document_ids": ["a3f1c9e28b7d4051", "77b2e01c4da9f8e3"],
  "top_k": 4
}
```

**200**
```json
{
  "session_id": "s-91ac",
  "answer": "Not Found — the retrieved passages do not mention Kubernetes.",
  "citations": ["[jane_doe_resume.pdf p.2 #7]"],
  "retrieval": { "...RetrievalTrace..." },
  "timings": {"retrieve_ms": 42.1, "llm_ms": 900.3, "total_ms": 951.0}
}
```
Memory is a sliding window of the last `CHAT_MEMORY_TURNS` turns per `session_id`.

---

## `GET /api/v1/documents` · `GET /api/v1/documents/{document_id}` · `DELETE /api/v1/documents/{document_id}`

List manifests, fetch one (`404 DOCUMENT_NOT_FOUND`), or delete a document and purge its chunks
from the index (**204**).
