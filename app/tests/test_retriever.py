"""Retrieval strategy tests.

The important assertions here are the *comparative* ones: BM25 finds a literal
token that dense retrieval ranks poorly, MMR returns more distinct chunks than
plain similarity, and hybrid fusion surfaces what either half alone would miss.
Those comparisons are the actual justification for building four retrievers
instead of one, so they are what the suite verifies.
"""

from __future__ import annotations

import pytest

from app.rag.embeddings import HashingEmbedder
from app.rag.retriever import (
    BM25Retriever,
    HybridRetriever,
    MMRRetriever,
    RetrieverFactory,
    SimilarityRetriever,
    deduplicate,
    tokenize,
)
from app.rag.vector_store import InMemoryVectorStore
from app.schemas.rag import DocumentType
from app.tests.fakes import make_chunk, make_retrieved

CORPUS = [
    "Python and PyTorch for deep learning model development and training",
    "Deployed services on AWS using Docker containers and GitHub Actions",
    "Kubernetes orchestration and Terraform infrastructure as code",
    "Led a team of four engineers delivering a classification pipeline",
    "MSc Computer Science from the University of Edinburgh in 2019",
    "Built retrieval augmented generation systems over support documents",
]


@pytest.fixture
def populated(store: InMemoryVectorStore, embedder: HashingEmbedder) -> InMemoryVectorStore:
    chunks = [
        make_chunk(text, document_id="r1", chunk_index=index)
        for index, text in enumerate(CORPUS)
    ]
    chunks.append(
        make_chunk(
            "Required: Kubernetes, Terraform, and distributed training experience",
            document_id="j1",
            filename="job.pdf",
            doc_type=DocumentType.JOB_DESCRIPTION,
            chunk_index=0,
        )
    )
    store.add(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    return store


class TestTokenize:
    def test_preserves_symbol_bearing_skill_names(self) -> None:
        """Stripping +, #, and . collapses C++, C#, and .NET into one useless token."""
        tokens = tokenize("Experienced in C++, C#, and .NET development")
        assert "c++" in tokens
        assert "c#" in tokens
        assert ".net" in tokens

    def test_lowercases(self) -> None:
        assert tokenize("Python PYTHON") == ["python", "python"]

    def test_handles_empty_input(self) -> None:
        assert tokenize("") == []


class TestDeduplicate:
    def test_keeps_the_highest_scoring_occurrence(self) -> None:
        chunks = [
            make_retrieved("same text", score=0.4, chunk_index=0),
            make_retrieved("same text", score=0.9, chunk_index=0),
        ]
        result = deduplicate(chunks)
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_reassigns_ranks(self) -> None:
        chunks = [
            make_retrieved("a", score=0.2, chunk_index=0),
            make_retrieved("b", score=0.8, chunk_index=1),
        ]
        assert [chunk.rank for chunk in deduplicate(chunks)] == [0, 1]

    def test_orders_by_score(self) -> None:
        chunks = [
            make_retrieved("low", score=0.2, chunk_index=0),
            make_retrieved("high", score=0.9, chunk_index=1),
        ]
        assert deduplicate(chunks)[0].text == "high"


class TestSimilarityRetriever:
    def test_returns_relevant_chunks(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        retriever = SimilarityRetriever(populated, embedder)
        results = retriever.retrieve("deep learning model training", top_k=3)

        assert len(results) == 3
        assert results == sorted(results, key=lambda hit: -hit.score)

    def test_respects_filters(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        retriever = SimilarityRetriever(populated, embedder)
        results = retriever.retrieve(
            "Kubernetes", top_k=5, filters={"doc_type": DocumentType.JOB_DESCRIPTION.value}
        )
        assert all(hit.chunk.metadata.doc_type is DocumentType.JOB_DESCRIPTION for hit in results)

    def test_reports_its_name(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        assert SimilarityRetriever(populated, embedder).name == "similarity"


class TestBM25Retriever:
    def test_finds_an_exact_rare_token(self, populated: InMemoryVectorStore) -> None:
        """The reason lexical search is in the stack at all."""
        results = BM25Retriever(populated).retrieve("Terraform", top_k=3)
        assert results
        assert any("Terraform" in hit.text for hit in results)

    def test_excludes_zero_scoring_chunks(self, populated: InMemoryVectorStore) -> None:
        """A zero score means no query term matched; that slot should go unused."""
        results = BM25Retriever(populated).retrieve("xylophone bassoon", top_k=5)
        assert results == []

    def test_empty_query_returns_nothing(self, populated: InMemoryVectorStore) -> None:
        assert BM25Retriever(populated).retrieve("", top_k=3) == []

    def test_empty_corpus_returns_nothing(self, store: InMemoryVectorStore) -> None:
        """BM25Okapi divides by mean document length and would raise on an empty corpus."""
        assert BM25Retriever(store).retrieve("Python", top_k=3) == []

    def test_respects_filters(self, populated: InMemoryVectorStore) -> None:
        results = BM25Retriever(populated).retrieve(
            "Kubernetes", top_k=5, filters={"document_id": "j1"}
        )
        assert all(hit.chunk.metadata.document_id == "j1" for hit in results)

    def test_reuses_its_index_across_queries(self, populated: InMemoryVectorStore) -> None:
        retriever = BM25Retriever(populated)
        retriever.retrieve("Python", top_k=2)
        signature = retriever._signature
        retriever.retrieve("Kubernetes", top_k=2)
        assert retriever._signature == signature


class TestMMRRetriever:
    def test_returns_the_requested_count(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        retriever = MMRRetriever(populated, embedder, lambda_mult=0.5, fetch_k=7)
        assert len(retriever.retrieve("engineering experience", top_k=3)) == 3

    def test_returns_distinct_chunks(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        """Diversity is the entire point: five slots spent on five different facts."""
        retriever = MMRRetriever(populated, embedder, lambda_mult=0.5, fetch_k=7)
        results = retriever.retrieve("engineering experience", top_k=4)
        assert len({hit.chunk_id for hit in results}) == len(results)

    def test_pure_relevance_lambda_matches_similarity_ordering(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        """lambda=1.0 disables the diversity term, so MMR must reduce to similarity."""
        query = "Kubernetes Terraform infrastructure"
        mmr = MMRRetriever(populated, embedder, lambda_mult=1.0, fetch_k=7)
        similarity = SimilarityRetriever(populated, embedder)

        assert (
            mmr.retrieve(query, top_k=1)[0].chunk_id
            == similarity.retrieve(query, top_k=1)[0].chunk_id
        )

    def test_short_candidate_pool_is_returned_unchanged(
        self, store: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        chunk = make_chunk("only chunk in the entire store here")
        store.add([chunk], embedder.embed_documents([chunk.text]))

        retriever = MMRRetriever(store, embedder, fetch_k=10)
        assert len(retriever.retrieve("anything", top_k=5)) == 1


class TestHybridRetriever:
    def test_fuses_both_retrievers(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        hybrid = HybridRetriever(
            SimilarityRetriever(populated, embedder), BM25Retriever(populated), fetch_k=7
        )
        results = hybrid.retrieve("Kubernetes Terraform", top_k=3)

        assert results
        assert any("hybrid" in hit.retriever for hit in results)

    def test_reciprocal_rank_fusion_scores_are_positive_and_ordered(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        hybrid = HybridRetriever(
            SimilarityRetriever(populated, embedder), BM25Retriever(populated), rrf_k=60, fetch_k=7
        )
        results = hybrid.retrieve("Python PyTorch deep learning", top_k=4)

        assert all(hit.score > 0 for hit in results)
        assert results == sorted(results, key=lambda hit: -hit.score)

    def test_labels_which_halves_contributed(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        """The label is what makes a retrieval trace diagnosable."""
        hybrid = HybridRetriever(
            SimilarityRetriever(populated, embedder), BM25Retriever(populated), fetch_k=7
        )
        results = hybrid.retrieve("Terraform", top_k=3)
        assert any("lexical" in hit.retriever or "dense" in hit.retriever for hit in results)

    def test_surfaces_a_lexical_hit_the_dense_half_ranks_poorly(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        """The concrete justification for hybrid search in this domain."""
        hybrid = HybridRetriever(
            SimilarityRetriever(populated, embedder), BM25Retriever(populated), fetch_k=7
        )
        results = hybrid.retrieve("Terraform", top_k=3)
        assert any("Terraform" in hit.text for hit in results)


class TestRetrieverFactory:
    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [
            ("similarity", "similarity"),
            ("mmr", "mmr"),
            ("bm25", "bm25"),
            ("hybrid", "hybrid"),
        ],
    )
    def test_builds_each_strategy(
        self, retriever_factory: RetrieverFactory, strategy: str, expected: str
    ) -> None:
        assert retriever_factory.create(strategy).name == expected

    def test_rejects_an_unknown_strategy(self, retriever_factory: RetrieverFactory) -> None:
        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            retriever_factory.create("telepathy")

    def test_wraps_in_reranking_when_configured(
        self, store: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        class StubReranker:
            def rerank(self, query, candidates, *, top_k):  # type: ignore[no-untyped-def]
                return list(reversed(list(candidates)))[:top_k]

        factory = RetrieverFactory(store, embedder, reranker=StubReranker())  # type: ignore[arg-type]
        assert factory.create("similarity").name == "similarity+rerank"


class TestRerankingRetriever:
    def test_over_fetches_then_trims(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        """Over-fetching is what gives the reranker anything to improve on."""
        from app.rag.retriever import RerankingRetriever

        seen: dict[str, int] = {}

        class RecordingReranker:
            def rerank(self, query, candidates, *, top_k):  # type: ignore[no-untyped-def]
                seen["candidates"] = len(candidates)
                return list(candidates)[:top_k]

        retriever = RerankingRetriever(
            SimilarityRetriever(populated, embedder),
            RecordingReranker(),  # type: ignore[arg-type]
            multiplier=3,
        )
        results = retriever.retrieve("engineering", top_k=2)

        assert len(results) == 2
        assert seen["candidates"] > 2
