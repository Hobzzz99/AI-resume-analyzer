"""Text cleaning stage.

A thin, injectable object around the pure functions in :mod:`app.utils.text`.
The indirection earns its place for two reasons: cleaning is a pipeline *stage*
that must be timed and logged like every other (Principle VI), and a domain with
different artefacts — OCR output, HTML scrapes, transcripts — replaces this
object rather than editing the ingestion pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.schemas.rag import SourceDocument
from app.utils.logging import get_logger
from app.utils.text import has_meaningful_text, normalize_text

logger = get_logger(__name__)


class TextCleaner:
    """Normalises extracted text before it is split and embedded.

    Args:
        min_chars: Minimum length for a page to be considered content.
        min_alpha_ratio: Minimum proportion of alphabetic characters. Together
            with ``min_chars`` this distinguishes a real short document from the
            stray glyphs an image-only PDF extracts to.
    """

    def __init__(self, *, min_chars: int = 30, min_alpha_ratio: float = 0.35) -> None:
        self._min_chars = min_chars
        self._min_alpha_ratio = min_alpha_ratio

    def clean(self, text: str) -> str:
        """Normalise a single string."""
        return normalize_text(text)

    def is_meaningful(self, text: str) -> bool:
        """Whether the text carries usable content."""
        return has_meaningful_text(
            text, min_chars=self._min_chars, min_alpha_ratio=self._min_alpha_ratio
        )

    def clean_documents(self, documents: Sequence[SourceDocument]) -> list[SourceDocument]:
        """Normalise each page and drop the ones with nothing in them.

        Pages are dropped rather than kept-and-ignored so that empty pages never
        become empty chunks. An empty chunk embeds to a near-arbitrary point in
        vector space and will surface as a spurious hit for unrelated queries —
        a subtle, hard-to-diagnose corruption of retrieval quality.
        """
        cleaned: list[SourceDocument] = []
        dropped = 0

        for document in documents:
            text = self.clean(document.text)
            if not self.is_meaningful(text):
                dropped += 1
                continue
            cleaned.append(
                SourceDocument(text=text, filename=document.filename, page=document.page)
            )

        if dropped:
            logger.info(
                "dropped %d empty page(s) during cleaning",
                dropped,
                extra={"stage": "clean", "dropped_pages": dropped, "kept_pages": len(cleaned)},
            )
        return cleaned
