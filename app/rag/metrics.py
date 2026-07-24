"""Retrieval quality metrics.

Separated from the evaluation script so the metrics themselves are unit-testable.
A metric implementation that has never been tested against a hand-computed value
is a number, not a measurement — and the entire point of an evaluation harness is
that its numbers can be trusted.

All metrics are domain-free, which is why they live in the engine package: they
score rankings, and know nothing about what was ranked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import mean


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Proportion of relevant items appearing in the top ``k``.

    The primary metric for this system. A RAG answer can only be as good as its
    context, so "did retrieval surface the passages that contain the answer" is
    upstream of every other quality question — precision matters far less,
    because a few extra passages cost tokens while a missing one costs the
    answer.

    Returns 0.0 when nothing is relevant, treating an undefined ratio as a
    non-contribution rather than raising inside a metrics loop.
    """
    if not relevant:
        return 0.0
    found = set(retrieved[:k]) & set(relevant)
    return len(found) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Proportion of the top ``k`` that is relevant."""
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & set(relevant)) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Reciprocal of the rank of the first relevant item; 0.0 if none appears.

    Rewards getting *a* right answer high in the list, which is what matters when
    the prompt budget only admits a handful of passages.
    """
    relevant_set = set(relevant)
    for index, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / index
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Mean of the precisions measured at each relevant hit."""
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    precisions: list[float] = []
    for index, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            hits += 1
            precisions.append(hits / index)
    return sum(precisions) / len(relevant_set) if precisions else 0.0


def hit_rate(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """1.0 if any relevant item is in the top ``k``, else 0.0."""
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


@dataclass(slots=True)
class CaseResult:
    """Metrics for a single evaluation query."""

    query: str
    strategy: str
    recall: float
    precision: float
    reciprocal_rank: float
    average_precision: float
    hit: float
    duration_ms: float
    retrieved: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationReport:
    """Aggregated metrics for one retrieval strategy.

    ``mrr`` is the mean reciprocal rank and ``map_score`` the mean average
    precision — aggregates over ``cases``, computed in :meth:`summarise` so the
    report cannot drift out of sync with the cases it summarises.
    """

    strategy: str
    cases: list[CaseResult] = field(default_factory=list)
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    map_score: float = 0.0
    hit_rate: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    def summarise(self) -> EvaluationReport:
        """Compute the aggregates from the recorded cases."""
        if not self.cases:
            return self

        self.recall_at_k = round(mean(case.recall for case in self.cases), 4)
        self.precision_at_k = round(mean(case.precision for case in self.cases), 4)
        self.mrr = round(mean(case.reciprocal_rank for case in self.cases), 4)
        self.map_score = round(mean(case.average_precision for case in self.cases), 4)
        self.hit_rate = round(mean(case.hit for case in self.cases), 4)

        latencies = sorted(case.duration_ms for case in self.cases)
        self.mean_latency_ms = round(mean(latencies), 2)
        # Nearest-rank p95. With a handful of cases this is the largest sample,
        # which is the honest answer for a small set rather than an interpolated
        # number implying more precision than the data supports.
        index = max(0, min(len(latencies) - 1, round(0.95 * len(latencies)) - 1))
        self.p95_latency_ms = round(latencies[index], 2)
        return self

    def as_row(self) -> dict[str, float | str]:
        """Flatten to one row for a comparison table."""
        return {
            "strategy": self.strategy,
            "recall@k": self.recall_at_k,
            "precision@k": self.precision_at_k,
            "MRR": self.mrr,
            "MAP": self.map_score,
            "hit_rate": self.hit_rate,
            "mean_ms": self.mean_latency_ms,
            "p95_ms": self.p95_latency_ms,
        }
