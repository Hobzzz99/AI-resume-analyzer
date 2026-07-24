/**
 * Score meters.
 *
 * The colour bands mirror the scoring guidance in the prompt template
 * (`app/prompts/templates/resume_analysis_v1.yaml`). Keeping them aligned
 * matters: if the prompt calls 75 a "strong match" and the UI paints it amber,
 * the interface contradicts the analysis it is displaying.
 */

import type { ResumeAnalysis } from "@/lib/types";

const BANDS = [
  { min: 75, label: "Strong", bar: "bg-strong", text: "text-strong" },
  { min: 60, label: "Moderate", bar: "bg-moderate", text: "text-moderate" },
  { min: 0, label: "Weak", bar: "bg-weak", text: "text-weak" },
] as const;

function band(score: number) {
  return BANDS.find((entry) => score >= entry.min) ?? BANDS[BANDS.length - 1]!;
}

const LABELS: Record<string, { title: string; hint: string }> = {
  overall: { title: "Overall match", hint: "Holistic fit, weighted by what the role demands" },
  technical: { title: "Technical", hint: "Languages, frameworks, cloud, ML depth" },
  experience: { title: "Experience", hint: "Relevance, seniority, and measured impact" },
  education: { title: "Education", hint: "Degrees, certifications, formal training" },
  ats: { title: "ATS compatibility", hint: "Keyword coverage and parseable structure" },
};

function Gauge({ name, score }: { name: string; score: number }) {
  const { label, bar, text } = band(score);
  const meta = LABELS[name] ?? { title: name, hint: "" };

  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <span className="text-sm font-medium">{meta.title}</span>
        <span className={`text-sm font-semibold tabular-nums ${text}`}>
          {score}
          <span className="ml-1 text-xs font-normal text-slate-400">/ 100 · {label}</span>
        </span>
      </div>
      <div
        className="h-2 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800"
        role="meter"
        aria-valuenow={score}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={meta.title}
      >
        <div
          className={`h-full rounded-full transition-[width] duration-500 ${bar}`}
          style={{ width: `${score}%` }}
        />
      </div>
      {meta.hint && <p className="mt-1 text-xs text-slate-500">{meta.hint}</p>}
    </div>
  );
}

export function ScoreGauges({ analysis }: { analysis: ResumeAnalysis }) {
  const scores: Array<[string, number]> = [
    ["overall", analysis.overall_score],
    ["technical", analysis.technical_score],
    ["experience", analysis.experience_score],
    ["education", analysis.education_score],
    ["ats", analysis.ats_score],
  ];

  return (
    <section className="card">
      <h2 className="card-title">Scores</h2>
      <div className="space-y-5">
        {scores.map(([name, score]) => (
          <Gauge key={name} name={name} score={score} />
        ))}
      </div>

      <div className="mt-6 border-t border-slate-200 pt-4 dark:border-slate-800">
        <div className="flex items-baseline justify-between">
          <span className="text-sm font-medium">Model confidence</span>
          <span className="text-sm font-semibold tabular-nums">
            {(analysis.confidence * 100).toFixed(0)}%
          </span>
        </div>
        <p className="mt-1 text-xs text-slate-500">
          How well the retrieved passages supported this analysis — not how good the candidate is.
        </p>
      </div>
    </section>
  );
}
