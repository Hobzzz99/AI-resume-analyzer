"""Vector store tests.

Written against the in-memory implementation, which shares its contract with the
Chroma adapter. The metadata round trip gets particular attention: it is the
boundary where enums and datetimes become primitives, and a silent failure there
would strip the provenance every citation depends on.
"""

from __future__ import annotations

import pytest

from app.rag.embeddings import HashingEmbedder
from app.rag.vector_store import InMemoryVectorStore, build_where_clause, matches_filters
from app.schemas.rag import ChunkMetadata, DocumentType
from app.tests.fakes import make_chunk


@pytest.fixture
def populated(store: InMemoryVectorStore, embedder: HashingEmbedder) -> InMemoryVectorStore:
    """A store holding two resume chunks and one job-description chunk."""
    chunks = [
        make_chunk("Python PyTorch machine learning", document_id="r1", chunk_index=0),
        make_chunk("AWS Docker cloud deployment", document_id="r1", chunk_index=1),
        make_chunk(
            "Required: Kubernetes Terraform",
            document_id="j1",
            filename="job.pdf",
            doc_type=DocumentType.JOB_DESCRIPTION,
            chunk_index=0,
        ),
    ]
    store.add(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    return store


class TestMetadataRoundTrip:
    def test_survives_the_store_boundary(self) -> None:
        original = ChunkMetadata(
            document_id="abc123",
            filename="resume.pdf",
            doc_type=DocumentType.RESUME,
            page=2,
            chunk_index=7,
            char_count=412,
        )
        restored = ChunkMetadata.from_store(dict(original.to_store()))

        assert restored.document_id == original.document_id
        assert restored.doc_type is DocumentType.RESUME
        assert restored.page == 2
        assert restored.chunk_index == 7
        assert restored.chunk_id == original.chunk_id

    def test_flattens_to_primitives_only(self) -> None:
        """Chroma rejects enums and datetimes."""
        stored = ChunkMetadata(
            document_id="a", filename="f.pdf", doc_type=DocumentType.RESUME
        ).to_store()
        assert all(isinstance(value, (str, int, float, bool)) for value in stored.values())

    def test_chunk_id_is_derived_from_position(self) -> None:
        metadata = ChunkMetadata(
            document_id="abc", filename="f.pdf", doc_type=DocumentType.RESUME, page=3, chunk_index=5
        )
        assert metadata.chunk_id == "abc:3:5"


class TestFilters:
    def test_single_condition_stays_flat(self) -> None:
        assert build_where_clause({"doc_type": "resume"}) == {"doc_type": "resume"}

    def test_multiple_conditions_become_an_and_clause(self) -> None:
        clause = build_where_clause({"doc_type": "resume", "document_id": "r1"})
        assert "$and" in clause
        assert len(clause["$and"]) == 2

    def test_no_filters_means_no_clause(self) -> None:
        assert build_where_clause(None) is None
        assert build_where_clause({}) is None

    def test_matching_requires_every_condition(self) -> None:
        metadata = {"doc_type": "resume", "document_id": "r1"}
        assert matches_filters(metadata, {"doc_type": "resume"})
        assert not matches_filters(metadata, {"doc_type": "resume", "document_id": "other"})
        assert matches_filters(metadata, None)


class TestInMemoryVectorStore:
    def test_stores_and_counts(self, populated: InMemoryVectorStore) -> None:
        assert populated.count() == 3

    def test_rejects_misaligned_inputs(self, store: InMemoryVectorStore) -> None:
        with pytest.raises(ValueError, match="align"):
            store.add([make_chunk("one")], [[0.1], [0.2]])

    def test_query_returns_scored_results_best_first(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        results = populated.query(embedder.embed_query("Python machine learning"), top_k=3)

        assert results
        assert results == sorted(results, key=lambda hit: -hit.score)
        assert [hit.rank for hit in results] == list(range(len(results)))

    def test_scores_are_higher_is_better(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        """Normalising direction at this boundary keeps every consumer simple."""
        results = populated.query(embedder.embed_query("Python PyTorch"), top_k=3)
        assert results[0].score >= results[-1].score

    def test_query_respects_metadata_filters(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        results = populated.query(
            embedder.embed_query("Kubernetes"),
            top_k=5,
            filters={"doc_type": DocumentType.JOB_DESCRIPTION.value},
        )
        assert results
        assert all(hit.chunk.metadata.doc_type is DocumentType.JOB_DESCRIPTION for hit in results)

    def test_document_filter_isolates_one_document(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        """Without this, a previously uploaded resume leaks into the analysis."""
        results = populated.query(
            embedder.embed_query("anything"), top_k=10, filters={"document_id": "r1"}
        )
        assert {hit.chunk.metadata.document_id for hit in results} == {"r1"}

    def test_top_k_is_honoured(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        assert len(populated.query(embedder.embed_query("engineering"), top_k=2)) == 2

    def test_get_by_document_returns_document_order(
        self, populated: InMemoryVectorStore
    ) -> None:
        chunks = populated.get_by_document("r1")
        assert [chunk.metadata.chunk_index for chunk in chunks] == [0, 1]

    def test_upsert_is_idempotent(
        self, populated: InMemoryVectorStore, embedder: HashingEmbedder
    ) -> None:
        """Deterministic ids mean re-ingestion overwrites instead of duplicating."""
        chunk = make_chunk("Python PyTorch machine learning", document_id="r1", chunk_index=0)
        populated.add([chunk], embedder.embed_documents([chunk.text]))
        assert populated.count() == 3

    def test_delete_removes_only_that_document(self, populated: InMemoryVectorStore) -> None:
        assert populated.delete_document("r1") == 2
        assert populated.count() == 1
        assert populated.get_by_document("r1") == []

    def test_delete_of_an_unknown_document_is_a_no_op(
        self, populated: InMemoryVectorStore
    ) -> None:
        assert populated.delete_document("missing") == 0
        assert populated.count() == 3

    def test_all_chunks_supports_filtering(self, populated: InMemoryVectorStore) -> None:
        assert len(populated.all_chunks({"doc_type": DocumentType.RESUME.value})) == 2

    def test_health_reports_reachable(self, store: InMemoryVectorStore) -> None:
        assert store.health() is True
