/**
 * TypeScript mirrors of the service's Pydantic response models.
 *
 * Hand-written rather than generated from the OpenAPI schema, deliberately: this
 * file is small, it documents the contract at the boundary where the client
 * actually consumes it, and it does not add a codegen step to `npm run dev`.
 * For a larger surface the trade would flip and generation would win — that is
 * noted as future work in the README rather than pretended away.
 *
 * These types must stay in step with `app/schemas/`. The API contract in
 * `specs/001-resume-rag-analyzer/contracts/api.md` is the shared reference.
 */

export type DocumentType = "resume" | "job_description" | "generic";

export type RetrievalStrategy = "similarity" | "mmr" | "bm25" | "hybrid";

/** Per-stage wall-clock durations. `null` means the stage did not run. */
export interface StageTimings {
  load_ms: number | null;
  clean_ms: number | null;
  split_ms: number | null;
  /** `null` on a cached ingestion — the proof that no embedding work happened. */
  embed_ms: number | null;
  store_ms: number | null;
  retrieve_ms: number | null;
  rerank_ms: number | null;
  prompt_ms: number | null;
  llm_ms: number | null;
  parse_ms: number | null;
  total_ms: number | null;
}

export interface DocumentManifest {
  document_id: string;
  filename: string;
  doc_type: DocumentType;
  page_count: number;
  chunk_count: number;
  char_count: number;
  ingested_at: string;
  /** True when the content fingerprint was already indexed. */
  cached: boolean;
  timings: StageTimings;
}

export interface RetrievalStepTrace {
  name: string;
  query: string;
  strategy: string;
  candidates: number;
  returned: number;
  duration_ms: number;
  cached: boolean;
  top_score: number | null;
}

export interface RetrievalTrace {
  steps: RetrievalStepTrace[];
  total_chunks: number;
  unique_chunks: number;
  deduplicated: number;
  budget_truncated: boolean;
  reranked: boolean;
}

export interface EvidenceItem {
  claim: string;
  quote: string;
  citation: string;
  source: DocumentType;
}

export interface ResumeAnalysis {
  overall_score: number;
  technical_score: number;
  experience_score: number;
  education_score: number;
  ats_score: number;
  matched_skills: string[];
  missing_skills: string[];
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
  recruiter_summary: string;
  confidence: number;
  evidence: EvidenceItem[];
  /** Flat form, kept for compatibility with the original schema shape. */
  evidence_strings: string[];
  grounding_warnings: string[];
}

export interface AnalysisResult {
  request_id: string;
  analysis: ResumeAnalysis;
  resume: DocumentManifest;
  job: DocumentManifest;
  retrieval: RetrievalTrace;
  timings: StageTimings;
  model: string;
  prompt_template: string;
  /** Repair rounds the model needed before its output validated. */
  retry_count: number;
  cached: boolean;
  warnings: string[];
  created_at: string;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  environment: string;
  uptime_seconds: number;
  llm: { provider: string; model: string; configured: boolean; [key: string]: unknown };
  embeddings: { model: string; dimension: number; loaded: boolean; [key: string]: unknown };
  vector_store: { backend: string; chunk_count: number; reachable: boolean; [key: string]: unknown };
  retrieval: { strategy: string; top_k: number; reranking: boolean; [key: string]: unknown };
  prompts: { available: string[]; [key: string]: unknown };
  cache: Record<string, unknown>;
}

/** The service's uniform error envelope. */
export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    details: Record<string, unknown>;
    request_id: string | null;
  };
}

export interface AnalyzeRequest {
  resume_document_id: string;
  job_document_id: string;
  job_title?: string;
  top_k?: number;
  strategy?: RetrievalStrategy;
  prompt_template?: string;
  use_cache?: boolean;
}

/** A stage event from the SSE stream. Never a result — only `result` is validated. */
export interface StageEvent {
  stage: string;
  status?: string;
  duration_ms?: number | null;
  unique_chunks?: number;
  retry_count?: number;
  chunks?: number;
  model?: string;
  request_id?: string;
}
