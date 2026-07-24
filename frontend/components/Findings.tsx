/**
 * Recruiter summary, strengths, weaknesses, and recommendations.
 *
 * `grounding_warnings` and service warnings are rendered prominently rather than
 * tucked away. A thin analysis presented with the same confidence as a
 * well-grounded one is the failure mode this whole project is built to avoid.
 */

import type { AnalysisResult, ResumeAnalysis } from "@/lib/types";

function ListCard({
  title,
  items,
  marker,
  emptyMessage,
}: {
  title: string;
  items: string[];
  marker: string;
  emptyMessage: string;
}) {
  return (
    <div className="card">
      <h2 className="card-title">{title}</h2>
      {items.length === 0 ? (
        <p className="text-sm text-slate-500">{emptyMessage}</p>
      ) : (
        <ul className="space-y-3">
          {items.map((item, index) => (
            <li key={`${title}-${index}`} className="flex gap-3 text-sm leading-relaxed">
              <span aria-hidden className="select-none pt-0.5 text-slate-400">
                {marker}
              </span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function Warnings({ result }: { result: AnalysisResult }) {
  const warnings = [...result.warnings, ...result.analysis.grounding_warnings];
  const unique = Array.from(new Set(warnings));
  if (unique.length === 0) return null;

  return (
    <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-950/40">
      <h2 className="mb-2 text-sm font-semibold text-amber-900 dark:text-amber-200">
        Treat with caution
      </h2>
      <ul className="space-y-1.5">
        {unique.map((warning, index) => (
          <li key={index} className="text-sm text-amber-900 dark:text-amber-200">
            {warning}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function RecruiterSummary({ analysis }: { analysis: ResumeAnalysis }) {
  return (
    <section className="card">
      <h2 className="card-title">Recruiter summary</h2>
      {analysis.recruiter_summary ? (
        <p className="text-sm leading-relaxed">{analysis.recruiter_summary}</p>
      ) : (
        <p className="text-sm text-slate-500">
          Not Found — the retrieved passages did not support a summary.
        </p>
      )}
    </section>
  );
}

export function Findings({ analysis }: { analysis: ResumeAnalysis }) {
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <ListCard
        title="Strengths"
        items={analysis.strengths}
        marker="+"
        emptyMessage="No evidenced strengths were identified."
      />
      <ListCard
        title="Weaknesses"
        items={analysis.weaknesses}
        marker="−"
        emptyMessage="No gaps were identified."
      />
      <div className="md:col-span-2">
        <ListCard
          title="Recommendations"
          items={analysis.recommendations}
          marker="→"
          emptyMessage="No recommendations were produced."
        />
      </div>
    </div>
  );
}
