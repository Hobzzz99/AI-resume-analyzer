"use client";

/**
 * Evidence, grouped by source document.
 *
 * This component is the product's trust surface. A score with no justification
 * is a horoscope, so every claim is shown next to the verbatim passage that
 * produced it and the citation handle that locates it.
 *
 * Evidence whose citation does not appear in the retrieved set is flagged
 * inline. That is the clearest possible signal that the model invented a
 * source, and hiding it would defeat the purpose of showing evidence at all.
 */

import { useState } from "react";
import type { AnalysisResult, EvidenceItem } from "@/lib/types";

const SOURCE_LABELS: Record<string, string> = {
  resume: "Resume",
  job_description: "Job description",
  generic: "Document",
};

function EvidenceRow({ item, unsupported }: { item: EvidenceItem; unsupported: boolean }) {
  return (
    <li className="rounded-lg border border-slate-200 p-4 dark:border-slate-800">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="chip bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
          {SOURCE_LABELS[item.source] ?? item.source}
        </span>
        <span className="citation">{item.citation || "(no citation)"}</span>
        {unsupported && (
          <span className="chip bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300">
            citation not in retrieved passages
          </span>
        )}
      </div>

      <p className="text-sm font-medium">{item.claim}</p>
      <blockquote className="mt-2 border-l-2 border-slate-300 pl-3 text-sm italic text-slate-600 dark:border-slate-700 dark:text-slate-400">
        {item.quote}
      </blockquote>
    </li>
  );
}

export function EvidenceTable({ result }: { result: AnalysisResult }) {
  const [expanded, setExpanded] = useState(true);
  const evidence = result.analysis.evidence;

  // The service already reports unsupported citations as a warning; recomputing
  // the set here lets the offending row be marked individually rather than
  // leaving the user to work out which of twelve items the warning meant.
  const unsupportedCitations = new Set(
    (result.warnings.join(" ").match(/\[[^\]]+\]/g) ?? []) as string[],
  );

  return (
    <section className="card">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="card-title mb-0">
          Evidence <span className="ml-1 tabular-nums">({evidence.length})</span>
        </h2>
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          className="text-xs font-medium text-slate-500 hover:text-slate-900 dark:hover:text-slate-100"
        >
          {expanded ? "Collapse" : "Expand"}
        </button>
      </div>

      {evidence.length === 0 ? (
        <p className="text-sm text-slate-500">No evidence was returned for this analysis.</p>
      ) : (
        expanded && (
          <ul className="animate-fade-in space-y-3">
            {evidence.map((item, index) => (
              <EvidenceRow
                key={`${item.citation}-${index}`}
                item={item}
                unsupported={Boolean(item.citation) && unsupportedCitations.has(item.citation)}
              />
            ))}
          </ul>
        )
      )}

      <p className="mt-4 border-t border-slate-200 pt-4 text-xs text-slate-500 dark:border-slate-800">
        Each citation reads <span className="citation">[file p.PAGE #INDEX]</span> and points at a
        passage retrieved from your uploaded documents.
      </p>
    </section>
  );
}
