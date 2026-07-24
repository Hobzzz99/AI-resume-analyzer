"""Retrieval strategies.

Four retrievers plus a reranking decorator, all satisfying the same
:class:`~app.rag.base.Retriever` protocol so they are interchangeable at the call
site and selectable from configuration.

| Strategy | Strength | Weakness |
|---|---|---|
| ``similarity`` | Semantic; finds paraphrases | Near-duplicates; misses rare literal tokens |
| ``mmr`` | Diverse coverage across a document | Slightly lower precision at rank 1 |
| ``bm25`` | Exact terms — "Kubernetes", "CUDA", "SOC 2" | Blind to synonyms |
| ``hybrid`` | Both, fused by rank | Two searches per query |

Hybrid is the default, and the reason is specific to this domain. Resume matching
is simultaneously semantic ("led a team" ≈ "managed engineers") and brutally
literal — an ATS keyword like ``Kubernetes`` either appears or does not, and a
dense retriever will happily rank a chunk about Docker above it. Neither half is
sufficient alone.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from app.rag.base import Embedder, VectorStore
from app.rag.embeddings import cosine_similarity
from app.schemas.rag import Chunk, RetrievedChunk
from app.utils.logging import get_logger

logger = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9+#.]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenisation for lexical matching.

    ``+``, ``#``, and ``.`` are kept inside tokens deliberately: a tokenizer that
    strips them turns ``C++`` into ``c``, ``C#`` into ``c``, and ``.NET`` into
    ``net`` — collapsing three distinct, highly discriminative resume skills into
    one meaningless token. This is exactly the kind of detail that decides whether
    keyword search is useful in this domain.
    """
    return _TOKEN.findall(text.lower())


def deduplicate(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop repeated chunk ids, keeping the highest-scoring occurrence.

    Necessary because a retrieval plan issues several overlapping facet queries,
    and a chunk listing both "Python" and "AWS" will be returned by more than one
    of them. Without this, one strong chunk would consume several prompt slots.
    """
    best: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        existing = best.get(chunk.chunk_id)
        if existing is None or chunk.score > existing.score:
            best[chunk.chunk_id] = chunk

    ordered = sorted(best.values(), key=lambda item: -item.score)
    return [item.model_copy(update={"rank": rank}) for rank, item in enumerate(ordered)]


class SimilarityRetriever:
    """Dense vector similarity — the baseline strategy."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    @property
    def name(self) -> str:
        """Strategy label."""
        return "similarity"

    def retrieve(
        self, query: str, *, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` nearest chunks."""
        embedding = self._embedder.embed_query(query)
        return self._store.query(embedding, top_k=top_k, filters=filters)


class MMRRetriever:
    """Maximal Marginal Relevance — relevance traded against novelty.

    Fetches a candidate pool, then greedily selects the chunk maximising
    ``λ·sim(query, c) − (1−λ)·max sim(c, already_selected)``.

    This matters concretely here: a resume repeats its core skills across the
    summary, the skills section, and every project. Pure similarity returns five
    chunks all saying "Python, PyTorch, AWS" and the model sees one fact five
    times. MMR spends those five slots on five *different* parts of the document,
    which is what the twelve-facet analysis in FR-027 actually needs.
    """

    def __init__(
        self, store: VectorStore, embedder: Embedder, *, lambda_mult: float = 0.5, fetch_k: int = 20
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._lambda = lambda_mult
        self._fetch_k = fetch_k

    @property
    def name(self) -> str:
        """Strategy label."""
        return "mmr"

    def retrieve(
        self, query: str, *, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return up to ``top_k`` relevant but mutually diverse chunks."""
        query_embedding = self._embedder.embed_query(query)
        candidates = self._store.query(
            query_embedding, top_k=max(self._fetch_k, top_k), filters=filters
        )
        if len(candidates) <= top_k:
            return candidates

        # Re-embedding the candidate texts is the cost of keeping the VectorStore
        # protocol free of an "also return me the vectors" method. At fetch_k=20
        # on MiniLM this is a few milliseconds and buys a much smaller interface.
        embeddings = self._embedder.embed_documents([c.text for c in candidates])

        selected: list[int] = []
        remaining = set(range(len(candidates)))

        while remaining and len(selected) < top_k:
            best_index, best_value = None, float("-inf")
            for index in remaining:
                relevance = cosine_similarity(query_embedding, embeddings[index])
                redundancy = max(
                    (
                        cosine_similarity(embeddings[index], embeddings[chosen])
                        for chosen in selected
                    ),
                    default=0.0,
                )
                value = self._lambda * relevance - (1.0 - self._lambda) * redundancy
                if value > best_value:
                    best_index, best_value = index, value

            assert best_index is not None  # remaining is non-empty
            selected.append(best_index)
            remaining.discard(best_index)

        return [
            candidates[index].model_copy(update={"rank": rank, "retriever": "mmr"})
            for rank, index in enumerate(selected)
        ]


class BM25Retriever:
    """Okapi BM25 lexical retrieval.

    The index is rebuilt when the corpus fingerprint (filter signature + chunk
    count) changes rather than on every query. BM25 needs the whole corpus in
    memory to compute inverse document frequency, so there is no incremental
    update; at this scale a rebuild is milliseconds, and caching it keeps repeated
    facet queries within one analysis from paying for it six times.
    """

    def __init__(self, store: VectorStore, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._store = store
        self._k1 = k1
        self._b = b
        self._index: Any = None
        self._indexed_chunks: list[Chunk] = []
        self._signature: tuple[Any, ...] | None = None

    @property
    def name(self) -> str:
        """Strategy label."""
        return "bm25"

    def _corpus_signature(self, filters: dict[str, Any] | None) -> tuple[Any, ...]:
        """Cheap identity for the filtered corpus, used to invalidate the index."""
        filter_key = tuple(sorted((filters or {}).items()))
        return (filter_key, self._store.count(filters))

    def _ensure_index(self, filters: dict[str, Any] | None) -> None:
        """Build or reuse the BM25 index for this filter set."""
        signature = self._corpus_signature(filters)
        if signature == self._signature and self._index is not None:
            return

        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        self._indexed_chunks = self._store.all_chunks(filters)
        corpus = [tokenize(chunk.text) for chunk in self._indexed_chunks]
        # BM25Okapi divides by average document length; an empty corpus would
        # raise ZeroDivisionError inside the library.
        self._index = BM25Okapi(corpus, k1=self._k1, b=self._b) if corpus else None
        self._signature = signature
        logger.debug("built bm25 index", extra={"documents": len(corpus)})

    def retrieve(
        self, query: str, *, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks with the strongest lexical match."""
        self._ensure_index(filters)
        if self._index is None:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)
        ranked = sorted(
            zip(scores, self._indexed_chunks, strict=True),
            key=lambda pair: (-pair[0], pair[1].chunk_id),
        )
        return [
            RetrievedChunk(chunk=chunk, score=float(score), rank=rank, retriever="bm25")
            # A zero score means no query term appears; returning it would fill a
            # prompt slot with a chunk the retriever itself judged irrelevant.
            for rank, (score, chunk) in enumerate(ranked[:top_k])
            if score > 0.0
        ]


class HybridRetriever:
    """Dense + lexical, fused by Reciprocal Rank Fusion.

    ``RRF(c) = Σ_retrievers 1 / (k + rank_r(c))`` with ``k = 60``.

    RRF is used instead of weighted score blending because the two score scales
    are genuinely incomparable: cosine similarity is bounded in ``[-1, 1]`` while
    BM25 is unbounded and corpus-dependent. Min-max normalising them produces a
    number with no meaning, and its behaviour changes as documents are added.
    RRF consumes only rank order, so it is invariant to both problems and has no
    weight to tune per corpus.
    """

    def __init__(
        self,
        dense: SimilarityRetriever | MMRRetriever,
        lexical: BM25Retriever,
        *,
        rrf_k: int = 60,
        fetch_k: int = 20,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._rrf_k = rrf_k
        self._fetch_k = fetch_k

    @property
    def name(self) -> str:
        """Strategy label."""
        return "hybrid"

    def retrieve(
        self, query: str, *, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Fuse dense and lexical results and return the best ``top_k``."""
        pool = max(self._fetch_k, top_k)
        dense_hits = self._dense.retrieve(query, top_k=pool, filters=filters)
        lexical_hits = self._lexical.retrieve(query, top_k=pool, filters=filters)

        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        sources: dict[str, set[str]] = {}

        for hits, label in ((dense_hits, "dense"), (lexical_hits, "lexical")):
            for rank, hit in enumerate(hits):
                fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
                chunks[hit.chunk_id] = hit.chunk
                sources.setdefault(hit.chunk_id, set()).add(label)

        ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        return [
            RetrievedChunk(
                chunk=chunks[chunk_id],
                score=score,
                rank=rank,
                retriever=f"hybrid({'+'.join(sorted(sources[chunk_id]))})",
            )
            for rank, (chunk_id, score) in enumerate(ordered[:top_k])
        ]


class CrossEncoderReranker:
    """Cross-encoder reranking.

    A bi-encoder scores query and passage independently, so it can only compare
    two summaries of meaning. A cross-encoder reads the pair jointly and judges
    actual relevance, which is materially more accurate at the very top of the
    list — precisely the region that determines what enters the prompt.

    The cost is a second model download and roughly 100 ms per batch on CPU, so
    it is opt-in (``USE_RERANKER``). Off by default keeps first-run friction low
    and keeps CI offline.
    """

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", *, device: str = "cpu"
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    @property
    def model_name(self) -> str:
        """Configured cross-encoder id."""
        return self._model_name

    def _load(self) -> Any:
        """Load the cross-encoder on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            logger.info("loading cross-encoder", extra={"model": self._model_name})
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        """Reorder ``candidates`` and return the best ``top_k``.

        Degrades to the original ordering if the model cannot be loaded. A
        reranker is an optimisation, and taking down an analysis because an
        optional refinement failed would be the wrong trade.
        """
        if not candidates:
            return []
        try:
            model = self._load()
            scores = model.predict([(query, candidate.text) for candidate in candidates])
        except Exception:
            logger.warning("reranking failed; falling back to input order", exc_info=True)
            return list(candidates[:top_k])

        ranked = sorted(zip(scores, candidates, strict=True), key=lambda pair: -float(pair[0]))
        return [
            candidate.model_copy(
                update={"score": float(score), "rank": rank, "retriever": "cross-encoder"}
            )
            for rank, (score, candidate) in enumerate(ranked[:top_k])
        ]


class RerankingRetriever:
    """Decorator that over-fetches from an inner retriever, then reranks.

    A decorator rather than a flag inside each retriever: reranking is orthogonal
    to how candidates were found, and duplicating the logic in four classes would
    be exactly the repetition the constitution's DRY rule forbids.
    """

    def __init__(
        self, inner: Any, reranker: CrossEncoderReranker, *, multiplier: int = 4
    ) -> None:
        self._inner = inner
        self._reranker = reranker
        self._multiplier = multiplier

    @property
    def name(self) -> str:
        """Strategy label, composed from the wrapped retriever."""
        return f"{self._inner.name}+rerank"

    def retrieve(
        self, query: str, *, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Retrieve ``top_k * multiplier`` candidates, rerank, return ``top_k``."""
        candidates = self._inner.retrieve(query, top_k=top_k * self._multiplier, filters=filters)
        return self._reranker.rerank(query, candidates, top_k=top_k)


class RetrieverFactory:
    """Builds retrievers from configuration.

    Centralising construction keeps the strategy-to-class mapping in one place,
    so ``RETRIEVAL_STRATEGY=mmr`` in ``.env`` is genuinely all it takes to change
    behaviour — no call site knows which concrete class it holds.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        mmr_lambda: float = 0.5,
        fetch_k: int = 20,
        rrf_k: int = 60,
        reranker: CrossEncoderReranker | None = None,
        rerank_multiplier: int = 4,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._mmr_lambda = mmr_lambda
        self._fetch_k = fetch_k
        self._rrf_k = rrf_k
        self._reranker = reranker
        self._rerank_multiplier = rerank_multiplier
        # BM25 holds a cached index; one shared instance means the corpus is
        # tokenised once per analysis rather than once per facet query.
        self._bm25 = BM25Retriever(store)

    def create(self, strategy: str) -> Any:
        """Return the retriever for ``strategy``, wrapped in reranking if enabled.

        Raises:
            ValueError: Unknown strategy name.
        """
        base: Any
        if strategy == "similarity":
            base = SimilarityRetriever(self._store, self._embedder)
        elif strategy == "mmr":
            base = MMRRetriever(
                self._store, self._embedder, lambda_mult=self._mmr_lambda, fetch_k=self._fetch_k
            )
        elif strategy == "bm25":
            base = self._bm25
        elif strategy == "hybrid":
            base = HybridRetriever(
                SimilarityRetriever(self._store, self._embedder),
                self._bm25,
                rrf_k=self._rrf_k,
                fetch_k=self._fetch_k,
            )
        else:
            msg = f"Unknown retrieval strategy: {strategy!r}"
            raise ValueError(msg)

        if self._reranker is not None:
            return RerankingRetriever(base, self._reranker, multiplier=self._rerank_multiplier)
        return base
