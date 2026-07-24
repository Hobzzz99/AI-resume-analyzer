/**
 * Typed client for the analysis service.
 *
 * All requests go to this app's own `/api/...` routes, which proxy to FastAPI
 * (see `app/api/[...proxy]/route.ts`).
 *
 * The service returns one uniform error envelope for every failure, so this
 * module turns any non-2xx response into a single `ApiError` carrying the
 * machine-readable `code`. Callers branch on `code`, never on message text.
 */

import type {
  AnalysisResult,
  AnalyzeRequest,
  ApiErrorBody,
  DocumentManifest,
  HealthResponse,
  StageEvent,
} from "./types";

/** A failure reported by the service, preserving its error code and request id. */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details: Record<string, unknown>;
  readonly requestId: string | null;

  constructor(
    message: string,
    code: string,
    status: number,
    details: Record<string, unknown> = {},
    requestId: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.details = details;
    this.requestId = requestId;
  }

  /** Whether retrying unchanged could plausibly succeed. */
  get isRetryable(): boolean {
    return ["LLM_RATE_LIMITED", "LLM_TIMEOUT", "LLM_ERROR", "SERVICE_UNREACHABLE"].includes(
      this.code,
    );
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  let body: ApiErrorBody | null = null;
  try {
    body = (await response.json()) as ApiErrorBody;
  } catch {
    // A non-JSON error body means something upstream of the service failed —
    // a proxy, or a crash before the handler ran.
  }

  if (body?.error) {
    return new ApiError(
      body.error.message,
      body.error.code,
      response.status,
      body.error.details,
      body.error.request_id,
    );
  }
  return new ApiError(
    `Request failed with status ${response.status}.`,
    "UNKNOWN_ERROR",
    response.status,
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, { ...init, cache: "no-store" });
  if (!response.ok) throw await toApiError(response);
  return (await response.json()) as T;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export async function uploadResume(file: File): Promise<DocumentManifest> {
  const form = new FormData();
  form.append("file", file);
  return request<DocumentManifest>("/upload/resume", { method: "POST", body: form });
}

export async function uploadJobFile(file: File, title = ""): Promise<DocumentManifest> {
  const form = new FormData();
  form.append("file", file);
  form.append("title", title);
  return request<DocumentManifest>("/upload/job", { method: "POST", body: form });
}

export async function uploadJobText(text: string, title = ""): Promise<DocumentManifest> {
  return request<DocumentManifest>("/upload/job/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, title }),
  });
}

export async function analyze(payload: AnalyzeRequest): Promise<AnalysisResult> {
  return request<AnalysisResult>("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function listDocuments(): Promise<DocumentManifest[]> {
  return request<DocumentManifest[]>("/documents");
}

export async function deleteDocument(documentId: string): Promise<void> {
  await request<unknown>(`/documents/${documentId}`, { method: "DELETE" });
}

/** Callbacks for {@link analyzeStream}. */
export interface StreamHandlers {
  onStage?: (event: StageEvent) => void;
  onResult?: (result: AnalysisResult) => void;
  onError?: (error: ApiError) => void;
}

/**
 * Run an analysis, reporting stage progress as it happens.
 *
 * Uses `fetch` with a manually parsed SSE body rather than `EventSource`,
 * because `EventSource` only issues GET requests and cannot send the JSON body
 * this endpoint requires.
 *
 * Only the `result` event carries a validated analysis. Stage events are
 * progress reporting and must never be rendered as an outcome — the analysis is
 * not meaningful until the service has parsed and validated it.
 */
export async function analyzeStream(
  payload: AnalyzeRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/analyze/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(payload),
    cache: "no-store",
    signal,
  });

  if (!response.ok) {
    handlers.onError?.(await toApiError(response));
    return;
  }
  if (!response.body) {
    handlers.onError?.(new ApiError("The server sent no response body.", "EMPTY_STREAM", 502));
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line. The final fragment is kept in
    // the buffer because a frame can be split across two network chunks.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      dispatchFrame(frame, handlers);
    }
  }
}

function dispatchFrame(frame: string, handlers: StreamHandlers): void {
  let eventName = "message";
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return;

  let payload: unknown;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch {
    return; // a malformed frame is dropped rather than taking down the stream
  }

  switch (eventName) {
    case "stage":
      handlers.onStage?.(payload as StageEvent);
      break;
    case "result":
      handlers.onResult?.(payload as AnalysisResult);
      break;
    case "error": {
      const body = payload as ApiErrorBody;
      handlers.onError?.(
        new ApiError(
          body.error?.message ?? "The analysis failed.",
          body.error?.code ?? "UNKNOWN_ERROR",
          500,
          body.error?.details ?? {},
          body.error?.request_id ?? null,
        ),
      );
      break;
    }
    default:
      break; // `done` and any future event types
  }
}
