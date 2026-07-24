"""Text cleaning tests.

These cover the specific artefacts PDF extraction produces, because each one has
a measurable effect on retrieval: a de-hyphenation failure means ``Kubernetes``
is never matched by keyword search, and a whitespace failure means two identical
resumes fingerprint differently and get embedded twice.
"""

from __future__ import annotations

from app.rag.cleaner import TextCleaner
from app.schemas.rag import SourceDocument
from app.utils.text import (
    collapse_soft_linebreaks,
    has_meaningful_text,
    normalize_text,
    repair_hyphenation,
    truncate,
)


def test_collapses_runs_of_whitespace() -> None:
    assert normalize_text("Python     and      SQL") == "Python and SQL"


def test_collapses_excess_blank_lines_but_keeps_paragraphs() -> None:
    result = normalize_text("First paragraph.\n\n\n\n\nSecond paragraph.")
    assert result == "First paragraph.\n\nSecond paragraph."


def test_repairs_words_split_across_a_line_break() -> None:
    """The single highest-impact cleaning rule for PDF resumes."""
    assert repair_hyphenation("Kuber-\nnetes") == "Kubernetes"
    assert repair_hyphenation("micro-\n  services") == "microservices"


def test_preserves_legitimate_hyphens() -> None:
    assert "end-to-end" in normalize_text("Built end-to-end pipelines")


def test_joins_lines_wrapped_mid_sentence() -> None:
    result = collapse_soft_linebreaks("Built a retrieval system over\ntwo million documents")
    assert result == "Built a retrieval system over two million documents"


def test_keeps_bullet_list_structure() -> None:
    """Bullets are the unit of meaning in a resume; merging them destroys it."""
    result = normalize_text("Achievements:\n- Reduced latency\n- Led a team\n- Shipped a product")
    assert result.count("\n-") == 3


def test_normalizes_bullet_glyphs_to_one_marker() -> None:
    result = normalize_text("• Python\n▪ SQL\n● Docker")
    assert result.count("- ") == 3


def test_strips_control_characters() -> None:
    assert normalize_text("Python\x00\x07 Engineer") == "Python Engineer"


def test_expands_ligatures() -> None:
    assert normalize_text("workﬂow eﬃciency") == "workflow efficiency"


def test_normalizes_smart_punctuation() -> None:
    assert normalize_text("the team’s “goal”") == "the team's \"goal\""


def test_normalization_is_idempotent() -> None:
    """Non-idempotence would give one document two fingerprints and two indexes."""
    messy = "Kuber-\nnetes   and\n\n\n\nDocker  • AWS’s stack"
    once = normalize_text(messy)
    assert normalize_text(once) == once


def test_empty_input_is_handled() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   \n\n  ") == ""


def test_meaningful_text_rejects_too_short() -> None:
    assert not has_meaningful_text("Page 1")


def test_meaningful_text_rejects_extraction_noise() -> None:
    """An image-only PDF extracts to punctuation and page furniture, not to nothing."""
    assert not has_meaningful_text("... 1 | 2 | 3 ... 4 5 6 -- 7 8 9 ... 10 11 12 13 14 15")


def test_meaningful_text_accepts_real_prose() -> None:
    assert has_meaningful_text("Machine learning engineer with six years of experience.")


def test_truncate_breaks_on_a_word_boundary() -> None:
    result = truncate("Senior machine learning engineer with production experience", 30)
    assert len(result) <= 34
    assert result.endswith(" ...")
    assert not result.replace(" ...", "").endswith("engine")


def test_truncate_leaves_short_text_alone() -> None:
    assert truncate("short", 100) == "short"


class TestTextCleaner:
    def test_cleans_each_page(self) -> None:
        cleaner = TextCleaner()
        pages = [
            SourceDocument(text="Python    Engineer  with\nexperience", filename="cv.pdf", page=1)
        ]
        assert cleaner.clean_documents(pages)[0].text == "Python Engineer with experience"

    def test_drops_pages_with_no_content(self) -> None:
        """An empty chunk embeds to an arbitrary point and pollutes retrieval."""
        cleaner = TextCleaner()
        pages = [
            SourceDocument(text="Real content about machine learning.", filename="cv.pdf", page=1),
            SourceDocument(text="   \n  \n ", filename="cv.pdf", page=2),
            SourceDocument(text="4", filename="cv.pdf", page=3),
        ]
        cleaned = cleaner.clean_documents(pages)
        assert len(cleaned) == 1
        assert cleaned[0].page == 1

    def test_preserves_page_attribution(self) -> None:
        """Page numbers cannot be reconstructed later; losing them breaks citations."""
        cleaner = TextCleaner()
        pages = [
            SourceDocument(text="Experience section with detail.", filename="cv.pdf", page=1),
            SourceDocument(text="Education section with detail.", filename="cv.pdf", page=2),
        ]
        assert [page.page for page in cleaner.clean_documents(pages)] == [1, 2]
