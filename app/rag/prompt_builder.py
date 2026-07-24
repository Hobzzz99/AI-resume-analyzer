"""Prompt assembly from retrieved context.

This module is the enforcement point for Constitution Principle II. Its only
input is ``list[RetrievedChunk]`` — there is no function here that accepts a
``SourceDocument`` or a raw string of document text, so there is *no code path*
from a full document to a prompt. That is a stronger guarantee than a code
review comment, and it is what SC-005 is tested against.

Two further responsibilities:

* **Citations.** Every rendered chunk is prefixed with its citation handle, and
  the template instructs the model to reference those handles. This is what makes
  evidence verifiable rather than merely plausible (FR-019, SC-004).
* **Budget.** The assembled context is capped by both chunk count and character
  count. Exceeding the cap raises rather than silently truncating, because a
  prompt that quietly dropped half its context would produce a confident, wrong
  analysis with no signal that anything went missing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.schemas.rag import RetrievedChunk
from app.utils.exceptions import ContextBudgetExceededError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ContextBudget:
    """Ceiling on how much retrieved text may enter a single prompt.

    Two independent limits, because they fail differently: many tiny chunks blow
    the chunk count while a few large ones blow the character count, and either
    one alone lets the other through.
    """

    def __init__(self, *, max_chunks: int = 24, max_chars: int = 24_000) -> None:
        self.max_chunks = max_chunks
        self.max_chars = max_chars

    def apply(self, chunks: Sequence[RetrievedChunk]) -> tuple[list[RetrievedChunk], bool]:
        """Trim ``chunks`` to fit, preserving order.

        Order is preserved rather than re-sorted by score because callers hand
        chunks over already ordered by relevance within their facet groups;
        re-sorting globally would let one dense facet crowd out every other one.

        Returns:
            The chunks that fit, and whether anything was dropped.
        """
        kept: list[RetrievedChunk] = []
        used = 0

        for chunk in chunks:
            if len(kept) >= self.max_chunks:
                break
            length = len(chunk.text)
            if used + length > self.max_chars:
                continue  # skip this one; a later, smaller chunk may still fit
            kept.append(chunk)
            used += length

        truncated = len(kept) < len(chunks)
        if truncated:
            logger.info(
                "context truncated to budget",
                extra={
                    "stage": "prompt",
                    "kept": len(kept),
                    "dropped": len(chunks) - len(kept),
                    "chars": used,
                },
            )
        return kept, truncated

    def verify(self, chunks: Sequence[RetrievedChunk]) -> None:
        """Assert the budget holds.

        Raises:
            ContextBudgetExceededError: The budget was violated, which means a
                caller bypassed :meth:`apply`. Treated as a programming error and
                surfaced loudly, since silently over-sending is the exact failure
                Principle II exists to prevent.
        """
        total_chars = sum(len(chunk.text) for chunk in chunks)
        if len(chunks) > self.max_chunks or total_chars > self.max_chars:
            raise ContextBudgetExceededError(
                "Assembled context exceeds the configured prompt budget.",
                details={
                    "chunks": len(chunks),
                    "max_chunks": self.max_chunks,
                    "chars": total_chars,
                    "max_chars": self.max_chars,
                },
            )


class PromptBuilder:
    """Renders retrieved chunks into a prompt.

    Args:
        budget: Ceiling applied before rendering.
        include_scores: Whether to show relevance scores alongside each chunk.
            Off by default — a model shown "relevance: 0.42" tends to hedge, and
            the score is a retrieval diagnostic rather than evidence.
    """

    def __init__(
        self, budget: ContextBudget | None = None, *, include_scores: bool = False
    ) -> None:
        self._budget = budget or ContextBudget()
        self._include_scores = include_scores

    def render_chunk(self, chunk: RetrievedChunk) -> str:
        """Render one chunk with its citation handle."""
        header = chunk.citation
        if self._include_scores:
            header = f"{header} (relevance {chunk.score:.3f})"
        return f"{header}\n{chunk.text}"

    def render_group(self, title: str, chunks: Sequence[RetrievedChunk]) -> str:
        """Render one titled block of context.

        Empty groups render an explicit "No relevant passages retrieved" marker
        rather than being omitted. Omission would leave the model to infer why a
        section is absent, and models fill that silence by inventing content —
        whereas an explicit negative is a fact it can report as ``Not Found``.
        """
        if not chunks:
            return f"### {title}\n(No relevant passages retrieved for this facet.)"
        body = "\n\n".join(self.render_chunk(chunk) for chunk in chunks)
        return f"### {title}\n{body}"

    def build_context(
        self, groups: Mapping[str, Sequence[RetrievedChunk]]
    ) -> tuple[str, list[RetrievedChunk], bool]:
        """Assemble grouped chunks into a budgeted context string.

        The budget is applied across all groups together, then each group renders
        only the chunks that survived — so a single facet cannot consume the whole
        prompt while the others silently vanish.

        Returns:
            The rendered context, the chunks actually included, and whether the
            budget forced anything out.
        """
        # Deduplicate across facets before budgeting. With many facets and few
        # source chunks, the same passage is retrieved by several facet queries;
        # rendering it under each one duplicates its text many times over and
        # blows the prompt far past the character budget (a 3-chunk resume can
        # otherwise render as 40+ chunk copies across 16 facets). Each chunk is
        # therefore assigned to exactly one facet — the first (highest-priority)
        # one it appears in — so budget accounting matches what is actually sent.
        seen: set[str] = set()
        deduped_groups: dict[str, list[RetrievedChunk]] = {}
        for title, chunks in groups.items():
            unique_here: list[RetrievedChunk] = []
            for chunk in chunks:
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                unique_here.append(chunk)
            deduped_groups[title] = unique_here

        flattened = [chunk for chunks in deduped_groups.values() for chunk in chunks]
        kept, truncated = self._budget.apply(flattened)
        self._budget.verify(kept)

        kept_ids = {chunk.chunk_id for chunk in kept}
        rendered = [
            self.render_group(title, [c for c in chunks if c.chunk_id in kept_ids])
            for title, chunks in deduped_groups.items()
        ]
        return "\n\n".join(rendered), kept, truncated

    def build(
        self,
        template: Any,
        groups: Mapping[str, Sequence[RetrievedChunk]],
        *,
        context_variable: str = "context",
        **variables: Any,
    ) -> tuple[str, list[RetrievedChunk], bool]:
        """Render a LangChain prompt template with the assembled context.

        Args:
            template: Anything exposing LangChain's ``.format(**kwargs)`` —
                typed as ``Any`` so the engine does not depend on a specific
                template class, and so a plain string wrapper works in tests.
            groups: Titled groups of retrieved chunks, in presentation order.
            context_variable: Template variable receiving the rendered context.
            **variables: Remaining template variables.

        Returns:
            The formatted prompt, the chunks included, and the truncation flag.
        """
        context, kept, truncated = self.build_context(groups)
        prompt = template.format(**{context_variable: context, **variables})
        logger.debug(
            "built prompt",
            extra={
                "stage": "prompt",
                "chunks": len(kept),
                "prompt_chars": len(prompt),
                "truncated": truncated,
            },
        )
        return prompt, kept, truncated

    @staticmethod
    def citation_index(chunks: Sequence[RetrievedChunk]) -> dict[str, RetrievedChunk]:
        """Map citation handle to chunk.

        Used after generation to verify that every citation the model produced
        corresponds to a chunk that was actually in the prompt — a fabricated
        citation is the clearest possible signal of a hallucinated claim.
        """
        return {chunk.citation: chunk for chunk in chunks}
