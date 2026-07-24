r"""Extracting JSON from imperfect model output.

Even in JSON mode, models under load emit prose preambles, markdown fences,
trailing commentary, and occasionally two objects where one was requested. This
module recovers the intended object from all of those, and does so with a brace
scanner rather than a regular expression.

The regex approach (``r"\\{.*\\}"`` with DOTALL) fails on the two cases that
matter most: it cannot match nested objects correctly, and a brace inside a
string value — ``{"quote": "use {} for a dict"}`` — truncates the match. A scanner
that tracks string state and escapes handles both, which is why this is fifty
lines instead of one.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def strip_code_fences(text: str) -> str:
    """Return the contents of the first fenced block, or the text unchanged."""
    match = _FENCE.search(text)
    return match.group(1).strip() if match else text.strip()


def find_json_object(text: str) -> str | None:
    """Return the first balanced top-level JSON object in ``text``.

    Scans character by character, tracking whether the cursor is inside a string
    and whether the previous character was an escape, so braces in string values
    do not affect nesting depth.

    Returns:
        The substring from the first ``{`` to its matching ``}``, or ``None``.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def repair_common_defects(text: str) -> str:
    """Fix syntax mistakes that are unambiguous to correct.

    Only trailing commas and Python literals (``None``/``True``/``False``) are
    touched. The bar for inclusion here is that the intended meaning must be
    beyond doubt — anything requiring a guess belongs in the model's repair loop,
    where the model itself decides, rather than being silently patched by a regex.
    """
    text = _TRAILING_COMMA.sub(r"\1", text)
    for python_literal, json_literal in (("None", "null"), ("True", "true"), ("False", "false")):
        text = re.sub(rf"(?<![\"\w]){python_literal}(?![\"\w])", json_literal, text)
    return text


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of raw model output.

    Escalates through progressively more forgiving strategies, stopping at the
    first that succeeds:

    1. Parse the whole string — the common case in JSON mode.
    2. Strip markdown fences and parse.
    3. Scan for the first balanced object and parse.
    4. Apply unambiguous syntax repairs to that object and parse.

    Raises:
        ValueError: No JSON object could be recovered. The caller turns this into
            a repair round-trip, so the message is written to be useful when it
            reaches the model.
    """
    if not text or not text.strip():
        msg = "Model returned an empty response."
        raise ValueError(msg)

    for candidate in _candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        # A bare list where an object was required: recoverable only if it holds
        # exactly one object, which is a real and common single-item response.
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]

    preview = text[:300].replace("\n", " ")
    logger.warning("could not extract json from model output", extra={"preview": preview})
    msg = f"No valid JSON object found in the response. Response began: {preview!r}"
    raise ValueError(msg)


def _candidates(text: str) -> list[str]:
    """Ordered parse candidates, cheapest and most faithful first."""
    stripped = text.strip()
    unfenced = strip_code_fences(stripped)
    candidates = [stripped, unfenced]

    scanned = find_json_object(unfenced)
    if scanned:
        candidates.extend([scanned, repair_common_defects(scanned)])

    candidates.append(repair_common_defects(unfenced))
    # Preserve order while dropping duplicates, so identical candidates are not
    # parsed twice on the failure path.
    return list(dict.fromkeys(candidates))
