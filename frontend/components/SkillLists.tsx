/**
 * Matched and missing skills.
 *
 * The empty states are written carefully. "No skills were identified" is a
 * statement about *retrieval*, not about the candidate, and saying so prevents
 * a user from reading an indexing problem as a verdict on their experience.
 */

import type { ResumeAnalysis } from "@/lib/types";

function SkillGroup({
  title,
  skills,
  tone,
  emptyMessage,
}: {
  title: string;
  skills: string[];
  tone: "matched" | "missing";
  emptyMessage: string;
}) {
  const chipClass =
    tone === "matched"
      ? "chip bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300"
      : "chip bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300";

  return (
    <div>
      <div className="mb-3 flex items-center gap-2">
        <h3 className="text-sm font-semibold">{title}</h3>
        <span className="text-xs text-slate-400 tabular-nums">{skills.length}</span>
      </div>
      {skills.length === 0 ? (
        <p className="text-sm text-slate-500">{emptyMessage}</p>
      ) : (
        <ul className="flex flex-wrap gap-2">
          {skills.map((skill) => (
            <li key={skill} className={chipClass}>
              {skill}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function SkillLists({ analysis }: { analysis: ResumeAnalysis }) {
  return (
    <section className="card space-y-6">
      <h2 className="card-title">Skill coverage</h2>

      <SkillGroup
        title="Matched"
        skills={analysis.matched_skills}
        tone="matched"
        emptyMessage="No required skills were evidenced in the retrieved passages."
      />

      <SkillGroup
        title="Missing"
        skills={analysis.missing_skills}
        tone="missing"
        emptyMessage="No gaps were identified against the job description."
      />

      <p className="border-t border-slate-200 pt-4 text-xs text-slate-500 dark:border-slate-800">
        Skills are judged against the passages retrieved from your documents, not the full text.
        A skill listed as missing may still appear elsewhere in your resume.
      </p>
    </section>
  );
}
