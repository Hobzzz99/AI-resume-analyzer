"use client";

/**
 * The analyzer page: upload → analyze → results.
 *
 * All state is local to this component. There is no store, no context, and no
 * server state library, because the entire flow is three values and a request —
 * introducing a state library here would be architecture for its own sake.
 *
 * The analysis runs through the streaming endpoint so the user sees which stage
 * is executing during the 5–20 seconds of generation. Stage events are progress
 * only; the analysis is rendered exclusively from the terminal `result` event,
 * which is the one the service has validated.
 */

import { useCallback, useRef, useState } from "react";
import { EvidenceTable } from "@/components/EvidenceTable";
import { Findings, RecruiterSummary, Warnings } from "@/components/Findings";
import { HealthBadge } from "@/components/HealthBadge";
import { ScoreGauges } from "@/components/ScoreGauges";
import { SkillLists } from "@/components/SkillLists";
import { LiveStages, StageTrace } from "@/components/StageTrace";
import { UploadPanel } from "@/components/UploadPanel";
import { ApiError, analyzeStream } from "@/lib/api";
import type { AnalysisResult, DocumentManifest, RetrievalStrategy, StageEvent } from "@/lib/types";

const STRATEGIES: RetrievalStrategy[] = ["hybrid", "similarity", "mmr", "bm25"];

export default function Page() {
  const [resume, setResume] = useState<DocumentManifest | null>(null);
  const [job, setJob] = useState<DocumentManifest | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [stages, setStages] = useState<StageEvent[]>([]);
  const [error, setError] = useState<ApiError | null>(null);
  const [running, setRunning] = useState(false);
  const [strategy, setStrategy] = useState<RetrievalStrategy>("hybrid");
  const abortRef = useRef<AbortController | null>(null);

  const canAnalyze =
    resume !== null && job !== null && resume.document_id !== job.document_id && !running;

  const runAnalysis = useCallback(async () => {
    if (!resume || !job) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setRunning(true);
    setError(null);
    setResult(null);
    setStages([]);

    try {
      await analyzeStream(
        {
          resume_document_id: resume.document_id,
          job_document_id: job.document_id,
          job_title: job.filename.replace(/\.[^.]+$/, ""),
          strategy,
        },
        {
          onStage: (event) => setStages((current) => [...current, event]),
          onResult: setResult,
          onError: setError,
        },
        controller.signal,
      );
    } catch (caught) {
      if (!controller.signal.aborted) {
        setError(
          caught instanceof ApiError
            ? caught
            : new ApiError("The analysis failed unexpectedly.", "UNKNOWN_ERROR", 500),
        );
      }
    } finally {
      setRunning(false);
    }
  }, [resume, job, strategy]);

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">AI Resume Analyzer</h1>
        <p className="mt-1 max-w-2xl text-sm text-slate-600 dark:text-slate-400">
          Retrieval-augmented analysis. Your documents are chunked and indexed; only the passages
          relevant to each assessment facet are shown to the model, and every conclusion cites the
          passage it came from.
        </p>
        <div className="mt-3">
          <HealthBadge />
        </div>
      </header>

      <UploadPanel resume={resume} job={job} onResume={setResume} onJob={setJob} />

      <section className="card mt-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="card-title mb-1">3 · Analyze</h2>
            <label className="text-xs text-slate-500">
              Retrieval strategy
              <select
                className="input mt-1 w-44"
                value={strategy}
                onChange={(event) => setStrategy(event.target.value as RetrievalStrategy)}
                disabled={running}
              >
                {STRATEGIES.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <button
            type="button"
            className="btn-primary min-w-40"
            disabled={!canAnalyze}
            onClick={() => void runAnalysis()}
          >
            {running ? "Analyzing…" : "Run analysis"}
          </button>
        </div>

        {resume && job && resume.document_id === job.document_id && (
          <p className="mt-3 text-sm text-weak">
            The resume and job description are the same document — upload a different job
            description to compare them.
          </p>
        )}
        {!resume || !job ? (
          <p className="mt-3 text-sm text-slate-500">
            Index a resume and a job description to enable the analysis.
          </p>
        ) : null}
      </section>

      {(running || stages.length > 0) && !result && (
        <div className="mt-6">
          <LiveStages events={stages} />
        </div>
      )}

      {error && (
        <div className="mt-6 rounded-xl border border-red-300 bg-red-50 p-4 dark:border-red-800 dark:bg-red-950/40">
          <h2 className="text-sm font-semibold text-red-900 dark:text-red-200">
            {error.message}
          </h2>
          <p className="citation mt-1">
            {error.code}
            {error.requestId ? ` · ${error.requestId.slice(0, 8)}` : ""}
          </p>
          {error.isRetryable && (
            <button
              type="button"
              className="btn-secondary mt-3"
              onClick={() => void runAnalysis()}
              disabled={running}
            >
              Try again
            </button>
          )}
        </div>
      )}

      {result && (
        <div className="mt-6 animate-fade-in space-y-6">
          <Warnings result={result} />
          <div className="grid gap-6 md:grid-cols-2">
            <ScoreGauges analysis={result.analysis} />
            <SkillLists analysis={result.analysis} />
          </div>
          <RecruiterSummary analysis={result.analysis} />
          <Findings analysis={result.analysis} />
          <EvidenceTable result={result} />
          <StageTrace result={result} />
        </div>
      )}

      <footer className="mt-12 border-t border-slate-200 pt-6 text-xs text-slate-500 dark:border-slate-800">
        Analysis is generated from retrieved passages and may be incomplete. Every score is a
        judgement about the retrieved text, not about a person.
      </footer>
    </main>
  );
}
