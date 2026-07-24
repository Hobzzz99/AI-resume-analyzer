"""Resume-analysis domain schemas.

Constitution Principle III: this file *is* the contract with the model. Anything
the model returns that does not satisfy these constraints is rejected and repaired
rather than passed on, so the strictness here is load-bearing — every validator is
a hallucination the API cannot emit.

Design note on the validators: they split into two kinds, and the distinction
matters. **Normalisers** silently fix output that is correct but untidy —
``["Python", "python"]``, a stray ``"Not Found"`` in a list. **Rejecters** raise,
which triggers a repair round-trip; those are reserved for output that is
*wrong*, such as a confident score with no supporting evidence. Using a rejecter
where a normaliser would do burns provider quota on cosmetics; using a normaliser
where a rejecter belongs launders a fabrication into the response.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from app.schemas.rag import DocumentManifest, DocumentType, RetrievalTrace, StageTimings

Score = Annotated[int, Field(ge=0, le=100, description="0-100 match score.")]

NOT_FOUND = "Not Found"
"""Sentinel the model is instructed to emit when retrieved context cannot support
a field (FR-020). Recognised here so it never reaches a user as a literal skill
named "Not Found"."""


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    """Strip, drop empties and the ``Not Found`` sentinel, remove case-insensitive dupes.

    Order is preserved because the model emits its most confident items first,
    and sorting would discard that signal for no benefit.
    """
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        value = raw.strip().strip("•-–— ").strip()
        if not value or value.casefold() == NOT_FOUND.casefold():
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


class EvidenceItem(BaseModel):
    """One grounded justification for a conclusion.

    Structured rather than a bare string, which the brief's example schema used.
    A ``list[str]`` makes SC-004 — "every evidence entry resolves to a passage
    that exists" — unverifiable, because there is nothing to resolve *against*.
    Separating the claim, the quote, and the citation makes verification a
    dictionary lookup, and the flat string form is still exposed for
    compatibility via :attr:`ResumeAnalysis.evidence_strings`.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    claim: str = Field(min_length=1, max_length=300, description="The conclusion being supported.")
    quote: str = Field(min_length=1, max_length=1000, description="Verbatim retrieved text.")
    citation: str = Field(
        default="", max_length=200, description="Citation handle from the prompt."
    )
    source: DocumentType = Field(default=DocumentType.RESUME)

    def as_string(self) -> str:
        """Flatten to the ``"[citation] quote"`` form."""
        return f"{self.citation} {self.quote}".strip()


class ResumeAnalysis(BaseModel):
    """The validated analysis of one resume against one job description.

    ``extra="ignore"`` is deliberate: models frequently add a helpful-looking
    field such as ``"reasoning"``. Rejecting the whole response over an extra key
    would spend a repair round on something harmless, while accepting and
    discarding it costs nothing.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    overall_score: Score = Field(description="Overall fit of the candidate for the role.")
    technical_score: Score = Field(description="Technical skill alignment.")
    experience_score: Score = Field(description="Relevance and depth of work experience.")
    education_score: Score = Field(description="Education and certification alignment.")
    ats_score: Score = Field(description="Applicant-tracking-system compatibility.")

    matched_skills: list[str] = Field(
        default_factory=list, max_length=40, description="Required skills evidenced in the resume."
    )
    missing_skills: list[str] = Field(
        default_factory=list, max_length=40, description="Required skills absent from the resume."
    )
    strengths: list[str] = Field(default_factory=list, max_length=10)
    weaknesses: list[str] = Field(default_factory=list, max_length=10)
    recommendations: list[str] = Field(
        default_factory=list, max_length=10, description="Specific, actionable improvements."
    )

    recruiter_summary: str = Field(
        default="", max_length=2000, description="Recruiter-facing verdict in prose."
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The model's own confidence, given how much context supported the analysis.",
    )
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=30)

    grounding_warnings: list[str] = Field(
        default_factory=list,
        description="Populated by validation, not by the model. Flags weakly grounded output.",
    )

    # --------------------------------------------------------- normalisers ---

    @field_validator(
        "matched_skills", "missing_skills", "strengths", "weaknesses", "recommendations",
        mode="before",
    )
    @classmethod
    def _normalize_lists(cls, value: Any) -> Any:
        """Coerce and tidy list fields.

        Accepts a bare string because models under JSON-mode pressure sometimes
        return ``"Python, Docker"`` where a list was requested. Splitting it is
        unambiguous and saves a repair round-trip; rejecting it would be
        pedantry, since the intent is not in doubt.
        """
        if value is None:
            return []
        if isinstance(value, str):
            value = value.split(",") if "," in value else [value]
        if not isinstance(value, list):
            return value  # let pydantic report the type error
        return _dedupe_preserving_order([str(item) for item in value])

    @field_validator("recruiter_summary", mode="before")
    @classmethod
    def _normalize_summary(cls, value: Any) -> str:
        """Map the ``Not Found`` sentinel onto an empty summary."""
        text = "" if value is None else str(value).strip()
        return "" if text.casefold() == NOT_FOUND.casefold() else text

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> Any:
        """Accept a percentage where a 0-1 fraction was requested.

        ``confidence: 85`` is a common and unambiguous mistake. Rescaling is
        safe because a genuine fraction can never exceed 1.0.
        """
        if isinstance(value, (int, float)) and value > 1.0:
            return float(value) / 100.0
        if isinstance(value, str):
            cleaned = value.strip().rstrip("%")
            try:
                number = float(cleaned)
            except ValueError:
                return value
            return number / 100.0 if number > 1.0 else number
        return value

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence(cls, value: Any) -> Any:
        """Accept the flat ``list[str]`` evidence form and lift it into items.

        The brief's example schema used ``list[str]``, and models prompted with a
        structured shape occasionally revert to it. Rather than fail, the string
        is parsed into ``(citation, quote)`` when it carries a bracketed handle.
        """
        if not isinstance(value, list):
            return value
        lifted: list[Any] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if not text or text.casefold() == NOT_FOUND.casefold():
                    continue
                citation, quote = "", text
                if text.startswith("[") and "]" in text:
                    end = text.index("]")
                    citation, quote = text[: end + 1], text[end + 1 :].strip()
                lifted.append(
                    {"claim": quote[:120] or "unspecified", "quote": quote or citation,
                     "citation": citation}
                )
            else:
                lifted.append(item)
        return lifted

    # ---------------------------------------------------------- rejecters ---

    @model_validator(mode="after")
    def _require_grounding(self) -> ResumeAnalysis:
        """Reject a confident verdict with no evidence behind it.

        This is Principle IV made executable. A model that returns
        ``overall_score: 82`` and ``evidence: []`` has asserted a conclusion it
        did not derive from the retrieved passages. Raising here sends it back
        through the repair loop with the reason attached, which in practice is
        what converts an ungrounded first attempt into a grounded second one.

        The zero-score case is exempt: "nothing matched" is a legitimate verdict
        with genuinely nothing to quote.

        Raises:
            ValueError: A non-zero analysis with no evidence.
        """
        scores_present = any(
            score > 0
            for score in (
                self.overall_score, self.technical_score, self.experience_score,
                self.education_score, self.ats_score,
            )
        )
        if scores_present and not self.evidence:
            msg = (
                "evidence must not be empty when any score is greater than zero: every "
                "conclusion must cite a retrieved passage"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _flag_thin_grounding(self) -> ResumeAnalysis:
        """Warn — but do not reject — when confidence outruns the evidence.

        A warning rather than an error because the output is well-formed and
        usable; the caller simply deserves to know it rests on three quotes. The
        UI renders these, so a thin analysis is visibly thin instead of quietly
        authoritative.
        """
        warnings: list[str] = []
        if self.confidence > 0.8 and len(self.evidence) < 3:
            warnings.append(
                f"High confidence ({self.confidence:.2f}) supported by only "
                f"{len(self.evidence)} evidence item(s)."
            )
        if not self.matched_skills and not self.missing_skills:
            warnings.append("No skills were identified in either document.")
        if not self.recruiter_summary:
            warnings.append("No recruiter summary was produced.")
        if warnings:
            # Assigned via object.__setattr__-free path: the model is mutable, and
            # re-validation is not triggered for a plain list assignment.
            self.grounding_warnings = [*self.grounding_warnings, *warnings]
        return self

    # ---------------------------------------------------------- accessors ---

    @computed_field  # type: ignore[prop-decorator]
    @property
    def evidence_strings(self) -> list[str]:
        """Evidence flattened to ``list[str]``.

        Present for compatibility with the schema shape in the original brief,
        and convenient for clients that only want to display quotes.
        """
        return [item.as_string() for item in self.evidence]

    @property
    def score_breakdown(self) -> dict[str, int]:
        """All five scores, for rendering as progress indicators."""
        return {
            "overall": self.overall_score,
            "technical": self.technical_score,
            "experience": self.experience_score,
            "education": self.education_score,
            "ats": self.ats_score,
        }

    def unsupported_citations(self, valid_citations: set[str]) -> list[str]:
        """Citations the model produced that were never in the prompt.

        A non-empty result is the strongest available signal of fabrication: the
        model invented a source. The analysis service records it as a warning
        rather than discarding the analysis, so the caller can see both the
        result and the reason to distrust it.
        """
        return [
            item.citation
            for item in self.evidence
            if item.citation and item.citation not in valid_citations
        ]


class AnalysisResult(BaseModel):
    """API envelope: the analysis plus everything needed to audit it."""

    request_id: str = Field(default="")
    analysis: ResumeAnalysis
    resume: DocumentManifest
    job: DocumentManifest
    retrieval: RetrievalTrace = Field(default_factory=RetrievalTrace)
    timings: StageTimings = Field(default_factory=StageTimings)
    model: str = Field(default="", description="Generation model that produced this analysis.")
    prompt_template: str = Field(default="")
    retry_count: int = Field(
        default=0, description="Repair rounds needed before validation passed."
    )
    cached: bool = False
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatTurn(BaseModel):
    """One message in a conversation."""

    role: str = Field(pattern="^(user|assistant)$")
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatAnswer(BaseModel):
    """A grounded answer to a follow-up question.

    A second schema on the same engine — the concrete demonstration that the
    pipeline is not shaped around ``ResumeAnalysis`` (US5, SC-008).
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    answer: str = Field(min_length=1, max_length=2000)
    citations: list[str] = Field(default_factory=list, max_length=10)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("citations", mode="before")
    @classmethod
    def _normalize_citations(cls, value: Any) -> Any:
        """Tidy the citation list."""
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return value
        return _dedupe_preserving_order([str(item) for item in value])
