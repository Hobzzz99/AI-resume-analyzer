"use client";

/**
 * Pipeline diagnostics: live stage progress, then the retrieval trace and timings.
 *
 * This panel is the visible half of Constitution Principle VI. It is not
 * decoration — RAG failures are almost never "the app is broken", they are
 * "retrieval returned the wrong three passages", and those two look identical
 * from the outside unless the stage timings and the per-facet trace are shown.
 */

import type { AnalysisResult, StageEvent } from "@/lib/types";

const STAGE_LABELS: Record<string, string> = {
  accepted: "Request accepted",
  cache: "Cache lookup",
  retrieval: "Retrieving passages",
  prompt: "Assembling prompt",
  generation: "Generating analysis",
  validation: "Validating output",
};

export function LiveStages({ events }: { events: StageEvent[] }) {
  if (events.length === 0) return null;

  return (
    <div className="card">
      <h2 className="card-title">Progress</h2>
      <ol className="space-y-2">
        {events.map((event, index) => {
          const done = event.status === "completed" || event.stage === "accepted";
          return (
            <li key={index} className="flex items-center gap-3 text-sm">
              <span
                aria-hidden
                className={`h-2 w-2 shrink-0 rounded-full ${
                  done ? "bg-strong" : "animate-pulse bg-moderate"
                }`}
              />
              <span className="flex-1">{STAGE_LABELS[event.stage] ?? event.stage}</span>
              {typeof event.unique_chunks === "number" && (
                <span className="text-xs text-slate-500 tabular-nums">
                  {event.unique_chunks} passages
                </span>
              )}
              {typeof event.duration_ms === "number" && (
                <span className="text-xs text-slate-500 tabular-nums">
                  {event.duration_ms.toFixed(0)} ms
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function Timings({ result }: { result: AnalysisResult }) {
  const entries = Object.entries(result.timings).filter(
    ([, value]) => typeof value === "number",
  ) as Array<[string, number]>;

  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold">Stage timings</h3>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-3">
        {entries.map(([stage, ms]) => (
          <div key={stage} className="flex justify-between gap-2">
            <dt className="text-slate-500">{stage.replace(/_ms$/, "")}</dt>
            <dd className="tabular-nums">{ms.toFixed(0)} ms</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function RetrievalSteps({ result }: { result: AnalysisResult }) {
  const { retrieval } = result;

  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold">
        Retrieval — {retrieval.unique_chunks} unique passages from {retrieval.steps.length} facet
        queries
      </h3>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="py-1.5 pr-4 font-medium">Facet</th>
              <th className="py-1.5 pr-4 font-medium">Strategy</th>
              <th className="py-1.5 pr-4 text-right font-medium">Returned</th>
              <th className="py-1.5 text-right font-medium">Time</th>
            </tr>
          </thead>
          <tbody>
            {retrieval.steps.map((step) => (
              <tr
                key={step.name}
                className="border-t border-slate-200 dark:border-slate-800"
                // An empty facet is the single most useful diagnostic here: it
                // means a whole assessment dimension had nothing behind it.
                title={step.returned === 0 ? "No passages retrieved for this facet" : step.query}
              >
                <td className={`py-1.5 pr-4 ${step.returned === 0 ? "text-weak" : ""}`}>
                  {step.name}
                </td>
                <td className="py-1.5 pr-4 text-slate-500">{step.strategy}</td>
                <td className="py-1.5 pr-4 text-right tabular-nums">{step.returned}</td>
                <td className="py-1.5 text-right tabular-nums text-slate-500">
                  {step.duration_ms.toFixed(0)} ms
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {retrieval.budget_truncated && (
        <p className="mt-2 text-xs text-moderate">
          Context exceeded the prompt budget and was truncated — some passages were not shown to
          the model.
        </p>
      )}
    </div>
  );
}

export function StageTrace({ result }: { result: AnalysisResult }) {
  return (
    <section className="card space-y-6">
      <div>
        <h2 className="card-title mb-2">Pipeline diagnostics</h2>
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="chip bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
            model: {result.model}
          </span>
          <span className="chip bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
            prompt: {result.prompt_template}
          </span>
          {result.retry_count > 0 && (
            <span className="chip bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300">
              {result.retry_count} repair {result.retry_count === 1 ? "round" : "rounds"}
            </span>
          )}
          {result.cached && (
            <span className="chip bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300">
              served from cache
            </span>
          )}
          <span className="citation self-center">{result.request_id.slice(0, 8)}</span>
        </div>
      </div>

      <RetrievalSteps result={result} />
      <Timings result={result} />
    </section>
  );
}
