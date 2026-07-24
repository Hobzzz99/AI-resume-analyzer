"""Content fingerprints.

A document's identity in this system is its *content*, not its filename and not
its bytes. Two consequences follow, both intentional:

* The same resume re-exported from Word has different bytes but identical text,
  so it fingerprints the same and is never re-embedded (FR-008, SC-006).
* Two different resumes both named ``resume.pdf`` — the single most likely
  filename in this domain — never collide.

The fingerprint also folds in the chunking parameters, so that changing
``CHUNK_SIZE`` produces new ids instead of silently mixing two chunk geometries
inside one collection (research.md R7).
"""

from __future__ import annotations

import hashlib

_FINGERPRINT_LENGTH = 16  # 64 bits of SHA-256: collision-free at this scale, readable in a URL


def content_fingerprint(text: str, *, salt: str = "", length: int = _FINGERPRINT_LENGTH) -> str:
    """Return a stable short fingerprint for normalised text.

    Args:
        text: Text that has already passed through ``normalize_text``. Passing
            raw text here defeats the purpose — whitespace differences would
            produce different ids for identical documents.
        salt: Configuration that changes how the text will be processed, such as
            the chunking signature. Different salt means a different document id.
        length: Hex characters to keep.

    Returns:
        A lowercase hex string of ``length`` characters.
    """
    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    if salt:
        digest.update(b"\x00")
        digest.update(salt.encode("utf-8"))
    return digest.hexdigest()[:length]


def stable_key(*parts: str) -> str:
    r"""Fingerprint an ordered tuple of strings.

    Used for cache keys — ``(resume_id, job_id, template, model)`` — where the
    parts must be unambiguously separated. The ``\\x1f`` unit separator cannot
    appear in any of the inputs, so ``("ab", "c")`` and ``("a", "bc")`` are
    guaranteed to hash differently.
    """
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:_FINGERPRINT_LENGTH]
