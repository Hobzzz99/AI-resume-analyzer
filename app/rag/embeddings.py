"""Embedding stage.

Two implementations of the :class:`~app.rag.base.Embedder` protocol:

* :class:`SentenceTransformerEmbedder` — the production path, ``all-MiniLM-L6-v2``
  running locally on CPU.
* :class:`HashingEmbedder` — a deterministic, dependency-free stand-in that makes
  the entire unit suite runnable with no network and no 90 MB download (SC-009).

The production model is loaded **lazily**, on first encode rather than at
construction. The composition root builds an embedder while FastAPI is still
importing modules; loading 90 MB of weights there would add seconds to startup
and would fail a health check that only wanted to report the configured model
name.

Why ``all-MiniLM-L6-v2``: 384 dimensions, ~90 MB, ~14 ms per short passage on
CPU, and semantic quality that is competitive with far larger models on
sentence-similarity benchmarks. For chunk-level resume retrieval, the accuracy
gained from a 1024-dimension model does not pay for tripling index size and
latency — and unlike a hosted embedding API it costs nothing and leaks nothing.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Sequence

from app.utils.exceptions import EmbeddingError
from app.utils.logging import get_logger

logger = get_logger(__name__)

MINILM_DIMENSION = 384


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder.

    Args:
        model_name: Hugging Face model id.
        device: ``cpu`` or ``cuda``.
        batch_size: Passages encoded per forward pass.
        normalize: L2-normalise vectors, which reduces cosine similarity to a dot
            product and lets the vector store skip a magnitude computation on
            every comparison.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._model = None
        self._dimension: int | None = None
        # Two concurrent requests can both miss the None check; the lock ensures
        # the weights are loaded exactly once rather than twice into memory.
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """Configured model identifier."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Vector dimensionality.

        Returns the known MiniLM dimension without loading the model when the
        model has not been loaded yet, so ``/health`` can report it cheaply.
        """
        if self._dimension is not None:
            return self._dimension
        if "MiniLM-L6" in self._model_name:
            return MINILM_DIMENSION
        return self._load().get_sentence_embedding_dimension()  # type: ignore[no-any-return]

    @property
    def is_loaded(self) -> bool:
        """Whether the weights are resident in memory."""
        return self._model is not None

    def _load(self):  # type: ignore[no-untyped-def] # third-party return type
        """Load the model once, under a lock.

        Raises:
            EmbeddingError: The model could not be downloaded or instantiated.
        """
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:  # another thread won the race
                return self._model
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415

                logger.info(
                    "loading embedding model",
                    extra={"model": self._model_name, "device": self._device},
                )
                model = SentenceTransformer(self._model_name, device=self._device)
                self._dimension = model.get_sentence_embedding_dimension()
                self._model = model
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load embedding model '{self._model_name}': {exc}",
                    details={"model": self._model_name},
                ) from exc
            return self._model

    def warm_up(self) -> None:
        """Load the weights ahead of the first request.

        Called from the FastAPI lifespan so the first user does not absorb the
        model download as request latency (SC-002 is stated for a warm model).
        """
        self._load()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of passages.

        Raises:
            EmbeddingError: Encoding failed.
        """
        if not texts:
            return []
        model = self._load()
        try:
            vectors = model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed {len(texts)} passage(s): {exc}",
                details={"model": self._model_name, "count": len(texts)},
            ) from exc
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Raises:
            EmbeddingError: Encoding failed.
        """
        if not text.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        return self.embed_documents([text])[0]


class HashingEmbedder:
    """Deterministic hash-based embedder for offline tests.

    Not a semantic model and does not pretend to be one. It projects character
    trigrams into a fixed-dimensional space via SHA-256, which yields two
    properties the test suite actually depends on:

    * **Determinism** — the same text always produces the same vector, so
      assertions about caching and idempotence are meaningful.
    * **Lexical overlap sensitivity** — texts sharing trigrams land closer
      together than texts that do not, so ordering assertions in retriever tests
      are testing real ranking behaviour rather than noise.

    It exists because the alternative — mocking ``encode()`` to return random
    vectors — makes every retrieval test vacuous.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        """Identifier reported in place of a real model name."""
        return f"hashing-embedder-{self._dimension}d"

    @property
    def dimension(self) -> int:
        """Vector dimensionality."""
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        """Project trigrams into the embedding space and L2-normalise."""
        vector = [0.0] * self._dimension
        normalized = " ".join(text.lower().split())
        if not normalized:
            return vector

        for position in range(max(1, len(normalized) - 2)):
            trigram = normalized[position : position + 3]
            digest = hashlib.sha256(trigram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            # Sign from an independent byte keeps unrelated trigrams from all
            # pushing the same direction, which would make every vector similar.
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            return vector
        return [value / magnitude for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of passages."""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Raises:
            EmbeddingError: The query is empty.
        """
        if not text.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        return self._vector(text)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity between two vectors.

    Lives here rather than in a general utility module because it is an
    embedding-space operation, and it is used by MMR, the semantic query cache,
    and the in-memory vector store. Returns 0.0 for a zero vector instead of
    raising: an all-zero embedding is degenerate input, and "no similarity" is
    the correct answer rather than a crash on the retrieval path.
    """
    if len(left) != len(right):
        msg = f"Vector dimension mismatch: {len(left)} vs {len(right)}"
        raise ValueError(msg)
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_magnitude = math.sqrt(sum(a * a for a in left))
    right_magnitude = math.sqrt(sum(b * b for b in right))
    if left_magnitude == 0.0 or right_magnitude == 0.0:
        return 0.0
    return dot / (left_magnitude * right_magnitude)
