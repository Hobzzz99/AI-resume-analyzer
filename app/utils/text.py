r"""Text normalisation primitives.

PDF text extraction is lossy in specific, predictable ways: soft hyphens split
words across lines, ligatures survive as single codepoints, column layouts inject
runs of spaces, and page furniture leaves stray control characters. Every one of
those artefacts degrades both embedding quality and keyword matching, because
``"Kuber-\\nnetes"`` embeds nowhere near ``"Kubernetes"`` and BM25 will never
match it as a term.

These functions are deliberately pure and individually testable. Order matters
and is documented in :func:`normalize_text`.
"""

from __future__ import annotations

import re
import unicodedata

# Ligatures pypdf commonly preserves from embedded fonts.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
}

# Typographic characters that add nothing semantically but fragment tokenisation.
_PUNCTUATION = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", "​": "",
    "﻿": "", "…": "...",
}

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_LINEBREAK = re.compile(r"(\w)[-­]\s*\n\s*(\w)")
_SOFT_LINEBREAK = re.compile(r"(?<![.!?:;\n])\n(?![\n\s*\-•\d])")
_HORIZONTAL_WS = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT = re.compile(r" +([,.;:!?])")
_BULLETS = re.compile(r"^[ \t]*[•●▪◦⁃∙*+][ \t]*", re.MULTILINE)


def strip_control_characters(text: str) -> str:
    """Remove non-printable control characters, preserving tabs and newlines."""
    return _CONTROL_CHARS.sub("", text)


def replace_ligatures(text: str) -> str:
    """Expand typographic ligatures into their component letters."""
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)
    return text


def normalize_punctuation(text: str) -> str:
    """Fold smart quotes, dashes, and exotic spaces to ASCII equivalents."""
    for source, replacement in _PUNCTUATION.items():
        text = text.replace(source, replacement)
    return text


def repair_hyphenation(text: str) -> str:
    r"""Rejoin words split by a hyphen at a line break.

    ``"Kuber-\nnetes"`` becomes ``"Kubernetes"``. Applied before line-break
    collapsing, because once the newline is gone the hyphen is indistinguishable
    from a legitimate compound like ``"end-to-end"``.
    """
    return _HYPHEN_LINEBREAK.sub(r"\1\2", text)


def collapse_soft_linebreaks(text: str) -> str:
    """Join lines wrapped mid-sentence, keeping paragraph and list structure.

    A newline is treated as wrapping only when the previous line did not end in
    terminal punctuation and the next line does not begin a new list item. That
    rule keeps a resume's bullet list intact — which matters, because bullets are
    the unit of meaning in a resume and merging them destroys the structure the
    splitter relies on.
    """
    return _SOFT_LINEBREAK.sub(" ", text)


def normalize_bullets(text: str) -> str:
    """Rewrite assorted bullet glyphs as a single ``- `` marker."""
    return _BULLETS.sub("- ", text)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces and excess blank lines, and trim line ends."""
    text = _HORIZONTAL_WS.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def normalize_text(text: str) -> str:
    """Apply the full normalisation pipeline in dependency order.

    Order is load-bearing:

    1. Unicode NFKC — canonicalise compatibility forms first, so later rules see
       one representation of each character.
    2. Ligatures and punctuation — before any regex that counts on ASCII.
    3. Control characters — before whitespace rules, which would otherwise
       preserve invisible junk.
    4. Hyphenation repair — must precede line-break collapsing (see above).
    5. Bullets, soft line breaks, whitespace — structural cleanup last.

    The function is idempotent: ``normalize_text(normalize_text(x)) ==
    normalize_text(x)``, which is asserted in the test suite. Idempotence matters
    because the same text is normalised on both the ingestion path and the
    fingerprinting path, and a non-idempotent cleaner would produce two different
    document ids for one document.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = replace_ligatures(text)
    text = normalize_punctuation(text)
    text = strip_control_characters(text)
    text = repair_hyphenation(text)
    text = normalize_bullets(text)
    text = collapse_soft_linebreaks(text)
    return collapse_whitespace(text)


def has_meaningful_text(text: str, *, min_chars: int = 30, min_alpha_ratio: float = 0.35) -> bool:
    """Decide whether extracted text is usable content or extraction noise.

    An image-only PDF does not extract to an empty string — it extracts to a
    handful of stray glyphs, page numbers, and punctuation from the page
    furniture. Length alone therefore cannot distinguish it from a real
    one-line document, so this also requires a minimum ratio of alphabetic
    characters. This is the check behind FR-004 and the ``EMPTY_DOCUMENT`` error.
    """
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    alpha = sum(1 for char in stripped if char.isalpha())
    return (alpha / len(stripped)) >= min_alpha_ratio


def truncate(text: str, max_chars: int, *, suffix: str = " ...") -> str:
    """Truncate at a word boundary, appending a marker when text was removed."""
    if len(text) <= max_chars:
        return text
    cut = text[: max(0, max_chars - len(suffix))]
    boundary = cut.rfind(" ")
    if boundary > max_chars * 0.6:
        cut = cut[:boundary]
    return cut.rstrip() + suffix
