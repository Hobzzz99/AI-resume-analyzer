"use client";

/**
 * Service status indicator.
 *
 * Shows the *resolved* generation model, which is the point: `GROQ_MODEL` is
 * configuration with no in-code default, so reading the source cannot tell you
 * what a running process is using — only the service can.
 *
 * A degraded service is reported plainly with the reason, because the common
 * first-run problem is an unset API key, and "analysis failed" would send a user
 * hunting through logs for a one-line fix.
 */

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

export function HealthBadge() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((response) => {
        if (!cancelled) setHealth(response);
      })
      .catch(() => {
        if (!cancelled) setUnreachable(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (unreachable) {
    return (
      <span className="chip bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300">
        service unreachable
      </span>
    );
  }

  if (!health) {
    return <span className="chip bg-slate-100 text-slate-500 dark:bg-slate-800">checking…</span>;
  }

  const healthy = health.status === "ok";

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span
        className={
          healthy
            ? "chip bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300"
            : "chip bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"
        }
      >
        {healthy ? "service ok" : "degraded"}
      </span>

      <span className="citation">{health.llm.model}</span>
      <span className="citation">{health.retrieval.strategy} retrieval</span>
      <span className="citation">{health.vector_store.chunk_count} indexed passages</span>

      {!health.llm.configured && (
        <span className="chip bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300">
          GROQ_API_KEY not set
        </span>
      )}
      {!health.vector_store.reachable && (
        <span className="chip bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300">
          vector store unreachable
        </span>
      )}
    </div>
  );
}
