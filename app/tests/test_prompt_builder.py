"""Prompt assembly tests.

This module verifies Constitution Principle II and SC-005: the context placed in
a prompt is bounded and carries citations, and there is no route from a whole
document into a prompt. The structural version of that last claim lives in
``test_architecture.py``; here it is verified behaviourally.
"""

from __future__ import annotations

import pytest

from app.rag.prompt_builder import ContextBudget, PromptBuilder
from app.schemas.rag import DocumentType
from app.tests.fakes import make_retrieved
from app.utils.exceptions import ContextBudgetExceededError


class FormatTemplate:
    """Minimal stand-in for a LangChain PromptTemplate."""

    def __init__(self, template: str) -> None:
        self._template = template

    def format(self, **kwargs: object) -> str:
        return self._template.format(**kwargs)


class TestContextBudget:
    def test_trims_to_the_chunk_ceiling(self) -> None:
        budget = ContextBudget(max_chunks=3, max_chars=100_000)
        chunks = [make_retrieved(f"chunk {i}", chunk_index=i) for i in range(10)]

        kept, truncated = budget.apply(chunks)
        assert len(kept) == 3
        assert truncated

    def test_trims_to_the_character_ceiling(self) -> None:
        budget = ContextBudget(max_chunks=100, max_chars=250)
        chunks = [make_retrieved("x" * 100, chunk_index=i) for i in range(10)]

        kept, truncated = budget.apply(chunks)
        assert sum(len(chunk.text) for chunk in kept) <= 250
        assert truncated

    def test_reports_no_truncation_when_everything_fits(self) -> None:
        budget = ContextBudget(max_chunks=10, max_chars=10_000)
        chunks = [make_retrieved(f"chunk {i}", chunk_index=i) for i in range(3)]

        kept, truncated = budget.apply(chunks)
        assert len(kept) == 3
        assert not truncated

    def test_preserves_input_order(self) -> None:
        """Re-sorting globally would let one dense facet crowd out every other."""
        budget = ContextBudget(max_chunks=3, max_chars=10_000)
        chunks = [make_retrieved(f"chunk {i}", score=i / 10, chunk_index=i) for i in range(5)]

        kept, _ = budget.apply(chunks)
        assert [chunk.text for chunk in kept] == ["chunk 0", "chunk 1", "chunk 2"]

    def test_a_large_chunk_does_not_block_a_later_small_one(self) -> None:
        budget = ContextBudget(max_chunks=10, max_chars=120)
        chunks = [
            make_retrieved("a" * 50, chunk_index=0),
            make_retrieved("b" * 500, chunk_index=1),
            make_retrieved("c" * 50, chunk_index=2),
        ]

        kept, _ = budget.apply(chunks)
        assert [chunk.chunk_id for chunk in kept] == [chunks[0].chunk_id, chunks[2].chunk_id]

    def test_verify_raises_when_the_budget_is_bypassed(self) -> None:
        """Silently over-sending is the exact failure Principle II exists to prevent."""
        budget = ContextBudget(max_chunks=2, max_chars=10_000)
        chunks = [make_retrieved(f"chunk {i}", chunk_index=i) for i in range(5)]

        with pytest.raises(ContextBudgetExceededError):
            budget.verify(chunks)

    def test_verify_passes_within_budget(self) -> None:
        budget = ContextBudget(max_chunks=10, max_chars=10_000)
        budget.verify([make_retrieved("small", chunk_index=0)])


class TestPromptBuilder:
    def test_every_chunk_carries_its_citation(self) -> None:
        """Citations are what make evidence verifiable rather than plausible."""
        builder = PromptBuilder()
        chunks = [
            make_retrieved("Python and PyTorch", filename="resume.pdf", page=1, chunk_index=0)
        ]

        context, _, _ = builder.build_context({"skills": chunks})
        assert "[resume.pdf p.1 #0]" in context
        assert "Python and PyTorch" in context

    def test_groups_are_rendered_as_titled_sections(self) -> None:
        """Telling the model why a passage was retrieved improves how it uses it."""
        builder = PromptBuilder()
        context, _, _ = builder.build_context(
            {
                "resume_skills": [make_retrieved("Python", chunk_index=0)],
                "job_requirements": [
                    make_retrieved(
                        "Kubernetes required",
                        doc_type=DocumentType.JOB_DESCRIPTION,
                        filename="job.pdf",
                        chunk_index=0,
                    )
                ],
            }
        )
        assert "### resume_skills" in context
        assert "### job_requirements" in context

    def test_an_empty_group_states_so_explicitly(self) -> None:
        """Silence invites invention; an explicit negative can be reported as Not Found."""
        builder = PromptBuilder()
        context, _, _ = builder.build_context({"certifications": []})
        assert "No relevant passages retrieved" in context

    def test_scores_are_hidden_by_default(self) -> None:
        """A model shown 'relevance: 0.42' hedges; the score is a diagnostic, not evidence."""
        builder = PromptBuilder()
        context, _, _ = builder.build_context(
            {"skills": [make_retrieved("Python", score=0.4237, chunk_index=0)]}
        )
        assert "0.4237" not in context

    def test_scores_can_be_shown_when_requested(self) -> None:
        builder = PromptBuilder(include_scores=True)
        context, _, _ = builder.build_context(
            {"skills": [make_retrieved("Python", score=0.4237, chunk_index=0)]}
        )
        assert "0.424" in context

    def test_budget_applies_across_all_groups(self) -> None:
        """One facet must not consume the whole prompt while others vanish."""
        builder = PromptBuilder(ContextBudget(max_chunks=3, max_chars=10_000))
        groups = {
            "facet_a": [make_retrieved(f"a{i}", chunk_index=i) for i in range(5)],
            "facet_b": [make_retrieved(f"b{i}", chunk_index=i + 10) for i in range(5)],
        }

        _, kept, truncated = builder.build_context(groups)
        assert len(kept) == 3
        assert truncated

    def test_renders_a_template_with_the_context(self) -> None:
        builder = PromptBuilder()
        template = FormatTemplate("ROLE: {job_title}\n\nCONTEXT:\n{context}\n\nSCHEMA: {schema}")

        prompt, kept, _ = builder.build(
            template,
            {"skills": [make_retrieved("Python engineer", chunk_index=0)]},
            job_title="ML Engineer",
            schema="{...}",
        )

        assert "ROLE: ML Engineer" in prompt
        assert "Python engineer" in prompt
        assert len(kept) == 1

    def test_prompt_length_is_bounded_regardless_of_document_size(self) -> None:
        """SC-005: the prompt cannot grow with the source document."""
        builder = PromptBuilder(ContextBudget(max_chunks=5, max_chars=2000))
        template = FormatTemplate("{context}")

        huge = {"facet": [make_retrieved("x" * 400, chunk_index=i) for i in range(500)]}
        prompt, kept, truncated = builder.build(template, huge)

        assert len(kept) <= 5
        assert len(prompt) < 4000
        assert truncated

    def test_citation_index_maps_handles_to_chunks(self) -> None:
        """Used after generation to detect a citation the model invented."""
        chunks = [
            make_retrieved("Python", filename="resume.pdf", page=1, chunk_index=0),
            make_retrieved("AWS", filename="resume.pdf", page=2, chunk_index=1),
        ]
        index = PromptBuilder.citation_index(chunks)

        assert set(index) == {"[resume.pdf p.1 #0]", "[resume.pdf p.2 #1]"}
        assert index["[resume.pdf p.1 #0]"].text == "Python"
