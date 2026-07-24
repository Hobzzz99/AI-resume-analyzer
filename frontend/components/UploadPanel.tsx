"use client";

/**
 * Document submission: a resume file, and a job description as file or paste.
 *
 * Each upload reports its manifest back — chunk count, page count, and whether
 * the content was already indexed. Surfacing `cached` is not a debug detail: it
 * is how a user understands that re-uploading the same file is free, and it is
 * the visible form of the "embed exactly once" guarantee.
 */

import { useRef, useState } from "react";
import { ApiError, uploadJobFile, uploadJobText, uploadResume } from "@/lib/api";
import type { DocumentManifest } from "@/lib/types";

type JobMode = "paste" | "file";

function ManifestSummary({ manifest }: { manifest: DocumentManifest }) {
  return (
    <div className="mt-3 rounded-lg bg-slate-100 p-3 text-xs dark:bg-slate-800">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="font-medium">{manifest.filename}</span>
        <span className="text-slate-500 tabular-nums">
          {manifest.chunk_count} passages
          {manifest.page_count > 0 && ` · ${manifest.page_count} pages`}
        </span>
        {manifest.cached ? (
          <span className="chip bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300">
            already indexed — no re-embedding
          </span>
        ) : (
          manifest.timings.embed_ms !== null && (
            <span className="text-slate-500 tabular-nums">
              embedded in {manifest.timings.embed_ms.toFixed(0)} ms
            </span>
          )
        )}
      </div>
      <p className="citation mt-1">{manifest.document_id}</p>
    </div>
  );
}

function ErrorNote({ error }: { error: ApiError }) {
  return (
    <div className="mt-3 rounded-lg border border-red-300 bg-red-50 p-3 text-sm dark:border-red-800 dark:bg-red-950/40">
      <p className="font-medium text-red-900 dark:text-red-200">{error.message}</p>
      <p className="citation mt-1">{error.code}</p>
    </div>
  );
}

export function UploadPanel({
  resume,
  job,
  onResume,
  onJob,
}: {
  resume: DocumentManifest | null;
  job: DocumentManifest | null;
  onResume: (manifest: DocumentManifest | null) => void;
  onJob: (manifest: DocumentManifest | null) => void;
}) {
  const [resumeBusy, setResumeBusy] = useState(false);
  const [jobBusy, setJobBusy] = useState(false);
  const [resumeError, setResumeError] = useState<ApiError | null>(null);
  const [jobError, setJobError] = useState<ApiError | null>(null);
  const [mode, setMode] = useState<JobMode>("paste");
  const [jobText, setJobText] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const jobFileRef = useRef<HTMLInputElement>(null);

  async function handleResume(file: File | undefined) {
    if (!file) return;
    setResumeBusy(true);
    setResumeError(null);
    try {
      onResume(await uploadResume(file));
    } catch (error) {
      onResume(null);
      setResumeError(
        error instanceof ApiError
          ? error
          : new ApiError("Upload failed.", "UNKNOWN_ERROR", 500),
      );
    } finally {
      setResumeBusy(false);
    }
  }

  async function handleJob() {
    setJobBusy(true);
    setJobError(null);
    try {
      const file = jobFileRef.current?.files?.[0];
      const manifest =
        mode === "file" && file
          ? await uploadJobFile(file, jobTitle)
          : await uploadJobText(jobText, jobTitle);
      onJob(manifest);
    } catch (error) {
      onJob(null);
      setJobError(
        error instanceof ApiError
          ? error
          : new ApiError("Upload failed.", "UNKNOWN_ERROR", 500),
      );
    } finally {
      setJobBusy(false);
    }
  }

  const jobReady = mode === "paste" ? jobText.trim().length > 0 : true;

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <section className="card">
        <h2 className="card-title">1 · Resume</h2>

        <label className="btn-secondary w-full cursor-pointer">
          <input
            type="file"
            accept=".pdf,.txt,.md"
            className="hidden"
            disabled={resumeBusy}
            onChange={(event) => void handleResume(event.target.files?.[0])}
          />
          {resumeBusy ? "Indexing…" : resume ? "Replace resume" : "Choose a PDF, TXT, or MD file"}
        </label>

        {resume && <ManifestSummary manifest={resume} />}
        {resumeError && <ErrorNote error={resumeError} />}

        <p className="mt-3 text-xs text-slate-500">
          Text-based PDFs only. Scanned documents are rejected — optical character recognition is
          out of scope.
        </p>
      </section>

      <section className="card">
        <h2 className="card-title">2 · Job description</h2>

        <div className="mb-3 flex gap-2">
          {(["paste", "file"] as const).map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => setMode(option)}
              className={
                mode === option
                  ? "chip bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                  : "chip bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300"
              }
            >
              {option === "paste" ? "Paste text" : "Upload file"}
            </button>
          ))}
        </div>

        <input
          className="input mb-2"
          placeholder="Role title (optional)"
          value={jobTitle}
          onChange={(event) => setJobTitle(event.target.value)}
        />

        {mode === "paste" ? (
          <textarea
            className="input h-32 resize-y"
            placeholder="Paste the job description here…"
            value={jobText}
            onChange={(event) => setJobText(event.target.value)}
          />
        ) : (
          <input ref={jobFileRef} type="file" accept=".pdf,.txt,.md" className="input" />
        )}

        <button
          type="button"
          className="btn-primary mt-3 w-full"
          disabled={jobBusy || !jobReady}
          onClick={() => void handleJob()}
        >
          {jobBusy ? "Indexing…" : job ? "Replace job description" : "Index job description"}
        </button>

        {job && <ManifestSummary manifest={job} />}
        {jobError && <ErrorNote error={jobError} />}
      </section>
    </div>
  );
}
