"""Caching.

Two caches with deliberately different matching semantics, because the cost of a
wrong hit differs by orders of magnitude between them.

**Analysis cache — exact.** Keyed on ``(resume_id, job_id, template, model,
strategy, top_k)``. Because document ids are content fingerprints, this is exact
semantic identity at the document level. The textbook "embed the request and hit
on cosine > 0.95" design is actively dangerous here: two resumes for the same
role are highly similar by construction, and serving one candidate's analysis to
another is the single worst failure this product could have.

**Query cache — fuzzy.** Keyed on the embedded sub-query, hitting above a
similarity threshold. Safe precisely because a near-miss costs nothing: the worst
case is retrieving passages for a slightly differently worded facet, which is
what the retriever would approximately have returned anyway.

Both are bounded LRU structures. Unbounded caches in a long-running service are a
memory leak with extra steps.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Generic, TypeVar

from app.rag.embeddings import cosine_similarity
from app.utils.hashing import stable_key
from app.utils.logging import get_logger

logger = get_logger(__name__)

ValueT = TypeVar("ValueT")


class LRUCache(Generic[ValueT]):
    """A bounded least-recently-used cache.

    ``OrderedDict.move_to_end`` gives O(1) recency tracking, which
    ``functools.lru_cache`` cannot provide here because the cached values must be
    inspectable and clearable at runtime (the API exposes a cache-bypass flag and
    the tests assert hit counts).
    """

    def __init__(self, max_size: int = 128) -> None:
        self._max_size = max(1, max_size)
        self._entries: OrderedDict[str, ValueT] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> ValueT | None:
        """Return the value for ``key``, marking it recently used."""
        if key not in self._entries:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return self._entries[key]

    def set(self, key: str, value: ValueT) -> None:
        """Store a value, evicting the least recently used entry if full."""
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = value
        while len(self._entries) > self._max_size:
            evicted, _ = self._entries.popitem(last=False)
            logger.debug("cache eviction", extra={"key": evicted})

    def clear(self) -> None:
        """Drop every entry and reset counters."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def stats(self) -> dict[str, Any]:
        """Hit/miss counters, for the health endpoint."""
        total = self.hits + self.misses
        return {
            "size": len(self._entries),
            "max_size": self._max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


class AnalysisCache:
    """Exact-match cache for completed analyses.

    Every input that can change the output is part of the key. Omitting
    ``strategy`` or ``top_k`` would mean a request that explicitly asked for
    different retrieval silently received the previous configuration's result —
    a bug that is invisible in the response and would quietly invalidate any
    strategy comparison.
    """

    def __init__(self, *, max_size: int = 128, enabled: bool = True) -> None:
        self._cache: LRUCache[Any] = LRUCache(max_size)
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Whether caching is active."""
        return self._enabled

    @staticmethod
    def build_key(
        *,
        resume_id: str,
        job_id: str,
        template: str,
        model: str,
        strategy: str,
        top_k: int,
    ) -> str:
        """Compose the cache key from every output-affecting input."""
        return stable_key(resume_id, job_id, template, model, strategy, str(top_k))

    def get(self, key: str) -> Any | None:
        """Return a cached analysis, or ``None``."""
        if not self._enabled:
            return None
        hit = self._cache.get(key)
        if hit is not None:
            logger.info("analysis cache hit", extra={"cache_key": key})
        return hit

    def set(self, key: str, value: Any) -> None:
        """Store a completed analysis."""
        if self._enabled:
            self._cache.set(key, value)

    def clear(self) -> None:
        """Drop every cached analysis."""
        self._cache.clear()

    @property
    def stats(self) -> dict[str, Any]:
        """Cache counters."""
        return {"enabled": self._enabled, **self._cache.stats}


class SemanticQueryCache:
    """Similarity-matched cache in front of sub-query retrieval.

    A retrieval plan issues the same facet queries on every analysis, so the
    exact-match layer alone already earns its keep. The embedding comparison adds
    hits for paraphrases — "cloud platforms and infrastructure" against "cloud
    experience" — at the cost of one embedding per miss.

    The linear scan over stored keys is acceptable at ``QUERY_CACHE_SIZE`` (512
    entries of 384 floats). A larger cache would need an index, and at that point
    the honest answer is that the cache has become a vector store.
    """

    def __init__(
        self,
        embedder: Any = None,
        *,
        max_size: int = 512,
        threshold: float = 0.97,
        enabled: bool = True,
    ) -> None:
        self._embedder = embedder
        self._cache: LRUCache[Any] = LRUCache(max_size)
        self._embeddings: OrderedDict[str, list[float]] = OrderedDict()
        self._threshold = threshold
        self._enabled = enabled and embedder is not None
        self.semantic_hits = 0

    @staticmethod
    def build_key(query: str, strategy: str, top_k: int, filters: dict[str, Any]) -> str:
        """Compose an exact key including the retrieval parameters.

        The filters are part of the key because two facets can share wording while
        targeting different documents. A cache keyed on query text alone would
        return the resume's passages for a job-description query — silently, and
        corrupting every conclusion downstream.
        """
        filter_repr = "&".join(f"{key}={value}" for key, value in sorted(filters.items()))
        return stable_key(query, strategy, str(top_k), filter_repr)

    def get(self, key: str, *, query: str | None = None, scope: str = "") -> Any | None:
        """Return a cached result by exact key, then by embedding similarity."""
        if not self._enabled:
            return None

        exact = self._cache.get(key)
        if exact is not None:
            return exact

        if query is None:
            return None

        candidate = self._embedder.embed_query(query)
        for stored_key, stored_embedding in reversed(self._embeddings.items()):
            # `scope` prefixes the stored key with the filter signature, so a
            # similar query is only ever matched within the same retrieval scope.
            if scope and not stored_key.startswith(scope):
                continue
            if cosine_similarity(candidate, stored_embedding) >= self._threshold:
                hit = self._cache.get(stored_key)
                if hit is not None:
                    self.semantic_hits += 1
                    logger.debug("semantic query cache hit", extra={"query": query[:80]})
                    return hit
        return None

    def set(self, key: str, value: Any, *, query: str | None = None) -> None:
        """Store a retrieval result, and its query embedding when available."""
        if not self._enabled:
            return
        self._cache.set(key, value)
        if query is not None:
            self._embeddings[key] = self._embedder.embed_query(query)
            self._embeddings.move_to_end(key)
            while len(self._embeddings) > len(self._cache) + 1:
                self._embeddings.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry."""
        self._cache.clear()
        self._embeddings.clear()
        self.semantic_hits = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Cache counters, including semantic-only hits."""
        return {"enabled": self._enabled, "semantic_hits": self.semantic_hits, **self._cache.stats}
