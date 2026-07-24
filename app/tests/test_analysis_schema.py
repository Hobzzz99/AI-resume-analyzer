"""Analysis schema tests.

Every validator here is a hallucination the API cannot emit, so each gets a test.
The suite also distinguishes the two validator kinds deliberately: *normalisers*
must silently fix untidy-but-correct output, and *rejecters* must raise so the
repair loop engages. Getting that split wrong either burns provider quota on
cosmetics or launders a fabrication into the response.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.analysis import ChatAnswer, EvidenceItem, ResumeAnalysis
from app.schemas.rag import DocumentType


def analysis(**overrides: object) -> ResumeAnalysis:
    """Build a valid analysis with overrides, so a test states only its subject."""
    payload: dict[str, object] = {
        "overall_score": 70,
        "technical_score": 70,
        "experience_score": 70,
        "education_score": 70,
        "ats_score": 70,
        "evidence": [
            {"claim": "c", "quote": "q", "citation": "[r.pdf p.1 #0]", "source": "resume"}
        ],
    }
    payload.update(overrides)
    return ResumeAnalysis.model_validate(payload)


class TestRanges:
    @pytest.mark.parametrize("score", [-1, 101, 250])
    def test_rejects_out_of_range_scores(self, score: int) -> None:
        with pytest.raises(ValidationError):
            analysis(overall_score=score)

    @pytest.mark.parametrize("score", [0, 50, 100])
    def test_accepts_in_range_scores(self, score: int) -> None:
        assert analysis(overall_score=score).overall_score == score

    def test_rejects_confidence_above_one_after_rescaling(self) -> None:
        with pytest.raises(ValidationError):
            analysis(confidence=150)  # 1.5 after the percentage rescale

    def test_rescales_a_percentage_confidence(self) -> None:
        """`confidence: 85` is common and unambiguous; rescaling saves a repair round."""
        assert analysis(confidence=85).confidence == pytest.approx(0.85)

    def test_accepts_a_percentage_string(self) -> None:
        assert analysis(confidence="85%").confidence == pytest.approx(0.85)

    def test_leaves_a_genuine_fraction_alone(self) -> None:
        assert analysis(confidence=0.72).confidence == pytest.approx(0.72)


class TestListNormalisation:
    def test_removes_case_insensitive_duplicates(self) -> None:
        """Models routinely emit ["Python", "python"]."""
        result = analysis(matched_skills=["Python", "python", "PYTHON", "AWS"])
        assert result.matched_skills == ["Python", "AWS"]

    def test_preserves_order(self) -> None:
        """The model emits its most confident items first; sorting discards that."""
        result = analysis(matched_skills=["Zebra", "Alpha", "Mango"])
        assert result.matched_skills == ["Zebra", "Alpha", "Mango"]

    def test_strips_whitespace_and_bullet_glyphs(self) -> None:
        result = analysis(strengths=["  • Strong ML background  ", "- Team leadership"])
        assert result.strengths == ["Strong ML background", "Team leadership"]

    def test_drops_the_not_found_sentinel(self) -> None:
        """A user must never see a skill called "Not Found"."""
        assert analysis(missing_skills=["Not Found"]).missing_skills == []

    def test_splits_a_comma_separated_string(self) -> None:
        """Under JSON-mode pressure models return "Python, Docker" for a list field."""
        assert analysis(matched_skills="Python, Docker, AWS").matched_skills == [
            "Python",
            "Docker",
            "AWS",
        ]

    def test_drops_empty_entries(self) -> None:
        assert analysis(weaknesses=["", "   ", "Real gap"]).weaknesses == ["Real gap"]

    def test_none_becomes_an_empty_list(self) -> None:
        assert analysis(recommendations=None).recommendations == []

    def test_enforces_the_length_ceiling(self) -> None:
        with pytest.raises(ValidationError):
            analysis(matched_skills=[f"skill{i}" for i in range(50)])


class TestSummaryNormalisation:
    def test_maps_the_sentinel_to_an_empty_summary(self) -> None:
        assert analysis(recruiter_summary="Not Found").recruiter_summary == ""

    def test_keeps_real_prose(self) -> None:
        assert analysis(recruiter_summary="Strong candidate.").recruiter_summary == (
            "Strong candidate."
        )


class TestEvidence:
    def test_rejects_a_confident_verdict_with_no_evidence(self) -> None:
        """Principle IV made executable: this failure triggers the repair loop."""
        with pytest.raises(ValidationError, match="evidence"):
            analysis(evidence=[])

    def test_allows_an_all_zero_analysis_with_no_evidence(self) -> None:
        """"Nothing matched" is a legitimate verdict with nothing to quote."""
        result = ResumeAnalysis.model_validate(
            {
                "overall_score": 0,
                "technical_score": 0,
                "experience_score": 0,
                "education_score": 0,
                "ats_score": 0,
                "evidence": [],
            }
        )
        assert result.evidence == []

    def test_lifts_the_flat_string_evidence_form(self) -> None:
        """The brief's original schema used list[str]; models sometimes revert to it."""
        result = analysis(evidence=["[resume.pdf p.1 #0] Built ML systems in PyTorch"])
        assert len(result.evidence) == 1
        assert result.evidence[0].citation == "[resume.pdf p.1 #0]"
        assert "PyTorch" in result.evidence[0].quote

    def test_exposes_the_flat_form_for_compatibility(self) -> None:
        result = analysis()
        assert result.evidence_strings == ["[r.pdf p.1 #0] q"]

    def test_detects_a_citation_the_model_invented(self) -> None:
        """The sharpest available signal of fabrication."""
        result = analysis(
            evidence=[
                {"claim": "c", "quote": "q", "citation": "[real.pdf p.1 #0]", "source": "resume"},
                {"claim": "c", "quote": "q", "citation": "[fake.pdf p.9 #9]", "source": "resume"},
            ]
        )
        unsupported = result.unsupported_citations({"[real.pdf p.1 #0]"})
        assert unsupported == ["[fake.pdf p.9 #9]"]

    def test_reports_no_unsupported_citations_when_all_are_real(self) -> None:
        assert analysis().unsupported_citations({"[r.pdf p.1 #0]"}) == []


class TestGroundingWarnings:
    def test_flags_confidence_outrunning_evidence(self) -> None:
        """A warning, not an error: the output is usable but visibly thin."""
        result = analysis(confidence=0.95)
        assert any("High confidence" in warning for warning in result.grounding_warnings)

    def test_flags_an_absent_skill_assessment(self) -> None:
        result = analysis(matched_skills=[], missing_skills=[])
        assert any("No skills" in warning for warning in result.grounding_warnings)

    def test_flags_a_missing_summary(self) -> None:
        result = analysis(recruiter_summary="")
        assert any("recruiter summary" in warning for warning in result.grounding_warnings)

    def test_a_well_grounded_analysis_warns_about_nothing(self) -> None:
        result = analysis(
            confidence=0.6,
            matched_skills=["Python"],
            recruiter_summary="A solid match for the role with relevant production experience.",
        )
        assert result.grounding_warnings == []


class TestMisc:
    def test_ignores_unrequested_extra_fields(self) -> None:
        """Rejecting a harmless extra key would spend a repair round for nothing."""
        result = ResumeAnalysis.model_validate(
            {
                "overall_score": 70,
                "technical_score": 70,
                "experience_score": 70,
                "education_score": 70,
                "ats_score": 70,
                "reasoning": "I thought about it carefully",
                "evidence": [{"claim": "c", "quote": "q", "citation": "x", "source": "resume"}],
            }
        )
        assert not hasattr(result, "reasoning")

    def test_score_breakdown_exposes_all_five(self) -> None:
        assert set(analysis().score_breakdown) == {
            "overall",
            "technical",
            "experience",
            "education",
            "ats",
        }

    def test_evidence_item_flattens_to_a_string(self) -> None:
        item = EvidenceItem(
            claim="c", quote="Built RAG systems", citation="[r.pdf p.1 #0]",
            source=DocumentType.RESUME,
        )
        assert item.as_string() == "[r.pdf p.1 #0] Built RAG systems"


class TestChatAnswer:
    def test_validates_a_grounded_answer(self) -> None:
        answer = ChatAnswer.model_validate(
            {"answer": "No Kubernetes experience is evidenced.", "citations": ["[r.pdf p.2 #7]"]}
        )
        assert answer.citations == ["[r.pdf p.2 #7]"]

    def test_rejects_an_empty_answer(self) -> None:
        with pytest.raises(ValidationError):
            ChatAnswer.model_validate({"answer": ""})

    def test_normalises_a_single_citation_string(self) -> None:
        answer = ChatAnswer.model_validate({"answer": "yes", "citations": "[r.pdf p.1 #0]"})
        assert answer.citations == ["[r.pdf p.1 #0]"]
