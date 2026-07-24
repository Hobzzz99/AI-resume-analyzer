"""Retrieval evaluation harness.

Answers the question a RAG system must be able to answer about itself: *is the
retrieval any good, and which strategy is best for this corpus?* Without it,
"hybrid search improves results" is an opinion.

The harness ingests a corpus, runs a set of labelled queries through every
strategy, and prints a comparison table of recall@k, precision@k, MRR, MAP, hit
rate, and latency.

Usage::

    python scripts/evaluate.py
    python scripts/evaluate.py --dataset data/eval/retrieval_cases.json --k 5
    python scripts/evaluate.py --strategies similarity,hybrid --json report.json

Runs entirely offline against the hashing embedder by default; pass
``--real-embeddings`` to evaluate the production model (requires a download).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.cleaner import TextCleaner
from app.rag.embeddings import HashingEmbedder
from app.rag.metrics import (
    CaseResult,
    EvaluationReport,
    average_precision,
    hit_rate,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.rag.retriever import RetrieverFactory
from app.rag.vector_store import InMemoryVectorStore
from app.schemas.rag import Chunk, ChunkMetadata, DocumentType

DEFAULT_DATASET = Path("data/eval/retrieval_cases.json")
DEFAULT_STRATEGIES = ("similarity", "mmr", "bm25", "hybrid")


def load_dataset(path: Path) -> dict[str, Any]:
    """Load the evaluation dataset.

    Expected shape::

        {
          "passages": [{"id": "p1", "text": "...", "doc_type": "resume"}],
          "cases":    [{"query": "...", "relevant": ["p1", "p4"]}]
        }

    Passages carry explicit ids so relevance labels stay stable when chunking
    parameters change — labelling by chunk index would silently relabel the whole
    dataset the moment ``CHUNK_SIZE`` moved.
    """
    if not path.is_file():
        msg = f"Dataset not found: {path}. Run with --dataset or create the file."
        raise SystemExit(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(
    passages: list[dict[str, Any]], embedder: Any
) -> InMemoryVectorStore:
    """Index the labelled passages, preserving their ids as chunk ids."""
    store = InMemoryVectorStore()
    cleaner = TextCleaner(min_chars=1, min_alpha_ratio=0.0)

    chunks = [
        Chunk(
            text=cleaner.clean(passage["text"]),
            metadata=ChunkMetadata(
                # The passage id is used as the document id so the resulting
                # chunk id is `{id}:0:0`, which the scorer maps back to the label.
                document_id=passage["id"],
                filename=passage.get("filename", "eval.txt"),
                doc_type=DocumentType(passage.get("doc_type", "generic")),
                page=0,
                chunk_index=0,
                char_count=len(passage["text"]),
            ),
        )
        for passage in passages
    ]
    store.add(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    return store


def evaluate_strategy(
    strategy: str,
    factory: RetrieverFactory,
    cases: list[dict[str, Any]],
    *,
    k: int,
) -> EvaluationReport:
    """Run every case through one strategy."""
    retriever = factory.create(strategy)
    report = EvaluationReport(strategy=strategy)

    for case in cases:
        started = time.perf_counter()
        hits = retriever.retrieve(case["query"], top_k=k, filters=case.get("filters"))
        duration_ms = (time.perf_counter() - started) * 1000

        retrieved_ids = [hit.chunk.metadata.document_id for hit in hits]
        relevant = case["relevant"]

        report.cases.append(
            CaseResult(
                query=case["query"],
                strategy=strategy,
                recall=recall_at_k(retrieved_ids, relevant, k),
                precision=precision_at_k(retrieved_ids, relevant, k),
                reciprocal_rank=reciprocal_rank(retrieved_ids, relevant),
                average_precision=average_precision(retrieved_ids, relevant),
                hit=hit_rate(retrieved_ids, relevant, k),
                duration_ms=duration_ms,
                retrieved=retrieved_ids,
            )
        )

    return report.summarise()


def print_table(reports: list[EvaluationReport], k: int) -> None:
    """Print a comparison table of every strategy."""
    columns = ["strategy", "recall@k", "precision@k", "MRR", "MAP", "hit_rate", "mean_ms", "p95_ms"]
    widths = {column: max(len(column), 11) for column in columns}

    print(f"\nRetrieval evaluation (k={k}, {len(reports[0].cases)} cases)")
    print("-" * (sum(widths.values()) + len(columns) * 2))
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("-" * (sum(widths.values()) + len(columns) * 2))

    for report in reports:
        row = report.as_row()
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))

    best = max(reports, key=lambda report: report.recall_at_k)
    print(f"\nBest recall@{k}: {best.strategy} ({best.recall_at_k})")


def print_failures(reports: list[EvaluationReport], limit: int = 3) -> None:
    """Show the queries each strategy handled worst.

    The aggregate tells you *that* a strategy is weaker; the failing queries tell
    you *why*, which is the part that leads to a fix.
    """
    for report in reports:
        failures = sorted(report.cases, key=lambda case: case.recall)[:limit]
        failures = [case for case in failures if case.recall < 1.0]
        if not failures:
            continue
        print(f"\nWeakest queries for '{report.strategy}':")
        for case in failures:
            print(f"  recall={case.recall:.2f}  {case.query!r} -> {case.retrieved}")


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval strategies.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=5, help="Cutoff for recall@k / precision@k.")
    parser.add_argument(
        "--strategies", type=str, default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated strategies to compare.",
    )
    parser.add_argument(
        "--real-embeddings", action="store_true",
        help="Use the production embedding model instead of the offline fake.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the report to a file.")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    passages, cases = dataset["passages"], dataset["cases"]

    if args.real_embeddings:
        from app.rag.embeddings import SentenceTransformerEmbedder

        embedder: Any = SentenceTransformerEmbedder()
        print(f"Using embedding model: {embedder.model_name}")
    else:
        embedder = HashingEmbedder(dimension=128)
        print("Using offline hashing embedder (pass --real-embeddings for the real model)")

    store = build_index(passages, embedder)
    factory = RetrieverFactory(store, embedder, fetch_k=max(20, args.k * 4))
    print(f"Indexed {store.count()} passages, {len(cases)} evaluation cases")

    reports = [
        evaluate_strategy(strategy.strip(), factory, cases, k=args.k)
        for strategy in args.strategies.split(",")
        if strategy.strip()
    ]

    print_table(reports, args.k)
    print_failures(reports)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([report.as_row() for report in reports], indent=2), encoding="utf-8"
        )
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
