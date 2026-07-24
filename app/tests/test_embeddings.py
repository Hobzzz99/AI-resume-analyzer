"""Embedding tests.

The production embedder is exercised only under the ``integration`` marker, since
it downloads 90 MB. The properties the rest of the system actually depends on —
fixed dimension, determinism, batch/single equivalence — are verified against the
fake, which is the object the whole suite runs on.
"""

from __future__ import annotations

import pytest

from app.rag.embeddings import HashingEmbedder, cosine_similarity
from app.utils.exceptions import EmbeddingError


class TestHashingEmbedder:
    def test_reports_its_dimension(self) -> None:
        assert HashingEmbedder(dimension=64).dimension == 64

    def test_vectors_have_the_declared_dimension(self) -> None:
        embedder = HashingEmbedder(dimension=32)
        assert len(embedder.embed_query("machine learning")) == 32

    def test_is_deterministic(self) -> None:
        """Non-determinism would make every caching assertion meaningless."""
        embedder = HashingEmbedder()
        assert embedder.embed_query("PyTorch") == embedder.embed_query("PyTorch")

    def test_batch_matches_single(self) -> None:
        embedder = HashingEmbedder()
        batch = embedder.embed_documents(["Python", "Kubernetes"])
        assert batch[0] == embedder.embed_query("Python")
        assert batch[1] == embedder.embed_query("Kubernetes")

    def test_empty_batch_returns_empty(self) -> None:
        assert HashingEmbedder().embed_documents([]) == []

    def test_rejects_an_empty_query(self) -> None:
        with pytest.raises(EmbeddingError):
            HashingEmbedder().embed_query("   ")

    def test_similar_text_scores_higher_than_unrelated(self) -> None:
        """Ordering assertions in retriever tests depend on this holding."""
        embedder = HashingEmbedder(dimension=128)
        query = embedder.embed_query("machine learning engineer")
        related = embedder.embed_query("machine learning engineering role")
        unrelated = embedder.embed_query("pastry chef kitchen brigade")

        assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)

    def test_vectors_are_normalised(self) -> None:
        embedder = HashingEmbedder()
        vector = embedder.embed_query("normalisation check")
        magnitude = sum(value * value for value in vector) ** 0.5
        assert magnitude == pytest.approx(1.0, abs=1e-6)


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_scores_zero_rather_than_raising(self) -> None:
        """A degenerate embedding must not crash the retrieval path."""
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            cosine_similarity([1.0], [1.0, 2.0])


@pytest.mark.integration
class TestSentenceTransformerEmbedder:
    """Requires a model download."""

    def test_produces_384_dimensional_vectors(self) -> None:
        from app.rag.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
        assert len(embedder.embed_query("machine learning")) == 384

    def test_reports_dimension_without_loading_the_model(self) -> None:
        """Health checks must not pay for a model download."""
        from app.rag.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
        assert embedder.dimension == 384
        assert not embedder.is_loaded

    def test_semantic_similarity_beats_lexical_coincidence(self) -> None:
        from app.rag.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
        query = embedder.embed_query("experience leading a team of engineers")
        paraphrase = embedder.embed_query("managed a group of software developers")
        unrelated = embedder.embed_query("the restaurant serves excellent pasta")

        assert cosine_similarity(query, paraphrase) > cosine_similarity(query, unrelated)
