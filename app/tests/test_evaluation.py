"""Retrieval metric tests.

Every expected value here is computed by hand in the docstring or the comment,
because a metric verified only against its own implementation measures nothing.
"""

from __future__ import annotations

import pytest

from app.rag.metrics import (
    CaseResult,
    EvaluationReport,
    average_precision,
    hit_rate,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_finds_every_relevant_item(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_finds_half(self) -> None:
        """1 of 2 relevant items inside the top 2 → 0.5."""
        assert recall_at_k(["a", "x"], ["a", "b"], k=2) == 0.5

    def test_respects_the_cutoff(self) -> None:
        """'b' sits at rank 3, outside k=2, so only 'a' counts."""
        assert recall_at_k(["a", "x", "b"], ["a", "b"], k=2) == 0.5

    def test_finds_nothing(self) -> None:
        assert recall_at_k(["x", "y"], ["a"], k=2) == 0.0

    def test_no_relevant_items_is_zero_not_an_error(self) -> None:
        """An undefined ratio must not raise inside a metrics loop."""
        assert recall_at_k(["a"], [], k=1) == 0.0


class TestPrecisionAtK:
    def test_all_retrieved_are_relevant(self) -> None:
        assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_half_retrieved_are_relevant(self) -> None:
        assert precision_at_k(["a", "x"], ["a", "b"], k=2) == 0.5

    def test_empty_retrieval_is_zero(self) -> None:
        assert precision_at_k([], ["a"], k=3) == 0.0


class TestReciprocalRank:
    def test_first_position(self) -> None:
        assert reciprocal_rank(["a", "x"], ["a"]) == 1.0

    def test_second_position(self) -> None:
        assert reciprocal_rank(["x", "a"], ["a"]) == 0.5

    def test_third_position(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_absent(self) -> None:
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_uses_the_first_relevant_hit(self) -> None:
        assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5


class TestAveragePrecision:
    def test_perfect_ranking(self) -> None:
        """Precision at each hit: 1/1 and 2/2 → mean 1.0."""
        assert average_precision(["a", "b"], ["a", "b"]) == 1.0

    def test_interleaved_ranking(self) -> None:
        """Hits at ranks 1 and 3: (1/1 + 2/3) / 2 = 0.8333."""
        assert average_precision(["a", "x", "b"], ["a", "b"]) == pytest.approx(0.8333, abs=1e-4)

    def test_nothing_relevant_retrieved(self) -> None:
        assert average_precision(["x", "y"], ["a"]) == 0.0

    def test_no_relevant_items(self) -> None:
        assert average_precision(["a"], []) == 0.0


class TestHitRate:
    def test_hit(self) -> None:
        assert hit_rate(["x", "a"], ["a"], k=2) == 1.0

    def test_miss_outside_the_cutoff(self) -> None:
        assert hit_rate(["x", "a"], ["a"], k=1) == 0.0


class TestEvaluationReport:
    def make_case(self, **overrides: float) -> CaseResult:
        values: dict[str, float] = {
            "recall": 1.0,
            "precision": 1.0,
            "reciprocal_rank": 1.0,
            "average_precision": 1.0,
            "hit": 1.0,
            "duration_ms": 10.0,
        }
        values.update(overrides)
        return CaseResult(query="q", strategy="hybrid", **values)  # type: ignore[arg-type]

    def test_averages_across_cases(self) -> None:
        report = EvaluationReport(
            strategy="hybrid",
            cases=[self.make_case(recall=1.0), self.make_case(recall=0.0)],
        ).summarise()
        assert report.recall_at_k == 0.5

    def test_computes_mean_reciprocal_rank(self) -> None:
        report = EvaluationReport(
            strategy="hybrid",
            cases=[self.make_case(reciprocal_rank=1.0), self.make_case(reciprocal_rank=0.5)],
        ).summarise()
        assert report.mrr == 0.75

    def test_reports_latency(self) -> None:
        report = EvaluationReport(
            strategy="hybrid",
            cases=[self.make_case(duration_ms=10.0), self.make_case(duration_ms=30.0)],
        ).summarise()
        assert report.mean_latency_ms == 20.0
        assert report.p95_latency_ms == 30.0

    def test_an_empty_report_is_safe_to_summarise(self) -> None:
        assert EvaluationReport(strategy="hybrid").summarise().recall_at_k == 0.0

    def test_flattens_to_a_comparison_row(self) -> None:
        row = EvaluationReport(strategy="hybrid", cases=[self.make_case()]).summarise().as_row()
        assert row["strategy"] == "hybrid"
        assert set(row) == {
            "strategy", "recall@k", "precision@k", "MRR", "MAP", "hit_rate", "mean_ms", "p95_ms"
        }


class TestStrategyComparison:
    """The harness must be able to tell strategies apart, or it measures nothing."""

    def test_a_better_ranking_scores_higher(self) -> None:
        good = EvaluationReport(
            strategy="hybrid",
            cases=[
                CaseResult("q", "hybrid", 1.0, 1.0, 1.0, 1.0, 1.0, 10.0),
                CaseResult("q", "hybrid", 1.0, 1.0, 1.0, 1.0, 1.0, 12.0),
            ],
        ).summarise()
        poor = EvaluationReport(
            strategy="similarity",
            cases=[
                CaseResult("q", "similarity", 0.5, 0.5, 0.5, 0.5, 1.0, 8.0),
                CaseResult("q", "similarity", 0.0, 0.0, 0.0, 0.0, 0.0, 9.0),
            ],
        ).summarise()

        assert good.recall_at_k > poor.recall_at_k
        assert good.mrr > poor.mrr
