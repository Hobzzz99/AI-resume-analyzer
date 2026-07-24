"""Test doubles.

These are what make Constitution Principle V pay off. Every one of them satisfies
a Protocol from :mod:`app.rag.base`, so the whole stack can be assembled with no
network, no API key, and no model download (SC-009).

They are *fakes*, not mocks: each has a real, if simplified, implementation.
A mocked retriever returning a canned list makes a retrieval test assert nothing
about retrieval. A fake that genuinely ranks by trigram overlap makes the same
test meaningful, and it fails when the ranking logic breaks.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from app.rag.base import StructuredResult
from app.schemas.rag import Chunk, ChunkMetadata, DocumentType, RetrievedChunk

SAMPLE_RESUME = """
JANE DOE
Senior Machine Learning Engineer | jane.doe@example.com | github.com/janedoe

PROFESSIONAL SUMMARY
Machine learning engineer with 6 years building and deploying production AI systems.
Specialises in natural language processing and retrieval-augmented generation.

TECHNICAL SKILLS
Languages: Python, SQL, TypeScript, Go
Frameworks: PyTorch, TensorFlow, scikit-learn, FastAPI, LangChain, React
Cloud and Infrastructure: AWS (SageMaker, Lambda, S3), Docker, GitHub Actions
Data: PostgreSQL, Redis, ChromaDB, Apache Spark

EXPERIENCE
Senior Machine Learning Engineer, Northwind AI (2022-present)
- Built a retrieval-augmented question answering system over 2 million support
  documents, reducing average handling time by 34%.
- Led a team of four engineers delivering a document classification pipeline
  processing 500,000 documents per day.
- Reduced model inference latency by 61% through quantisation and batching.

Machine Learning Engineer, Dataworks (2019-2022)
- Developed churn prediction models improving retention forecasting accuracy by 18%.
- Migrated the training pipeline from local notebooks to AWS SageMaker.

EDUCATION
MSc Computer Science, University of Edinburgh, 2019
BSc Mathematics, University of Manchester, 2017

CERTIFICATIONS
AWS Certified Machine Learning - Specialty (2023)
Deep Learning Specialisation, DeepLearning.AI (2021)
"""

SAMPLE_JOB = """
SENIOR AI ENGINEER - PLATFORM TEAM

We are hiring a Senior AI Engineer to build our retrieval and generation platform.

REQUIRED QUALIFICATIONS
- 5+ years of professional software engineering experience
- Expert-level Python
- Production experience with large language models and RAG architectures
- Deep familiarity with PyTorch
- Experience deploying services on AWS
- Strong grasp of vector databases and embedding models

PREFERRED QUALIFICATIONS
- Kubernetes and Terraform for infrastructure as code
- Experience with distributed training
- Open source contributions

RESPONSIBILITIES
- Design and operate the retrieval layer serving our AI products
- Mentor engineers and set technical direction
- Own model evaluation and quality measurement
"""


def make_chunk(
    text: str,
    *,
    document_id: str = "doc1",
    filename: str = "sample.pdf",
    doc_type: DocumentType = DocumentType.RESUME,
    page: int = 1,
    chunk_index: int = 0,
) -> Chunk:
    """Build a chunk with sensible defaults, so tests state only what they mean."""
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            document_id=document_id,
            filename=filename,
            doc_type=doc_type,
            page=page,
            chunk_index=chunk_index,
            char_count=len(text),
        ),
    )


def make_retrieved(
    text: str, *, score: float = 1.0, rank: int = 0, **kwargs: Any
) -> RetrievedChunk:
    """Build a retrieved chunk."""
    return RetrievedChunk(chunk=make_chunk(text, **kwargs), score=score, rank=rank)


class ScriptedLLMClient:
    """Returns queued responses in order. Satisfies ``LLMClient``.

    Records every prompt it received, which is how prompt-content assertions
    (citations present, budget respected, ``Not Found`` instructed) are made
    without inspecting private state.
    """

    def __init__(self, responses: Sequence[str], *, model_name: str = "fake-model") -> None:
        self._responses = list(responses)
        self._model_name = model_name
        self.prompts: list[str] = []
        self.systems: list[str | None] = []
        self.call_count = 0

    @property
    def model_name(self) -> str:
        """Identifier reported in place of a real model."""
        return self._model_name

    def generate(self, prompt: str, *, system: str | None = None, json_mode: bool = False) -> str:
        """Return the next scripted response.

        The last response repeats once the script is exhausted, so a test for the
        *success* path does not have to script one entry per retry.
        """
        self.prompts.append(prompt)
        self.systems.append(system)
        index = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        return self._responses[index]

    def stream(
        self, prompt: str, *, system: str | None = None, json_mode: bool = False
    ) -> Iterator[str]:
        """Yield the next scripted response in small pieces."""
        response = self.generate(prompt, system=system, json_mode=json_mode)
        for start in range(0, len(response), 16):
            yield response[start : start + 16]


class FailingLLMClient:
    """Always raises. For testing provider-failure propagation."""

    def __init__(self, error: Exception, *, model_name: str = "failing-model") -> None:
        self._error = error
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """Identifier reported in place of a real model."""
        return self._model_name

    def generate(self, prompt: str, *, system: str | None = None, json_mode: bool = False) -> str:
        """Raise the configured error."""
        raise self._error

    def stream(
        self, prompt: str, *, system: str | None = None, json_mode: bool = False
    ) -> Iterator[str]:
        """Raise the configured error."""
        raise self._error


class StaticGenerator:
    """Returns a fixed schema instance. Satisfies ``StructuredGenerator``.

    For pipeline tests that care about retrieval and prompt assembly rather than
    about generation, so they do not need valid JSON for every schema field.
    """

    def __init__(self, value: Any, *, model_name: str = "static-model") -> None:
        self._value = value
        self._model_name = model_name
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    @property
    def model_name(self) -> str:
        """Identifier reported in place of a real model."""
        return self._model_name

    def generate(self, prompt: str, schema: type[Any], *, system: str | None = None) -> Any:
        """Record the prompt and return the fixed value."""
        self.prompts.append(prompt)
        self.systems.append(system)
        return StructuredResult(value=self._value, raw=json.dumps({"static": True}), retry_count=0)


def valid_analysis_json(**overrides: Any) -> str:
    """A well-formed ``ResumeAnalysis`` payload, with optional field overrides.

    Centralised so a schema change breaks one fixture instead of nine tests, and
    so each test can express only the field it is actually about.
    """
    payload: dict[str, Any] = {
        "overall_score": 78,
        "technical_score": 82,
        "experience_score": 74,
        "education_score": 85,
        "ats_score": 70,
        "matched_skills": ["Python", "PyTorch", "AWS", "RAG"],
        "missing_skills": ["Kubernetes", "Terraform"],
        "strengths": ["Six years of production ML experience with quantified impact"],
        "weaknesses": ["No infrastructure-as-code experience evidenced"],
        "recommendations": ["Add a Kubernetes deployment to the MLOps project section"],
        "recruiter_summary": (
            "Strong applied machine learning candidate with directly relevant RAG "
            "experience. Main gap is infrastructure-as-code tooling."
        ),
        "confidence": 0.78,
        "evidence": [
            {
                "claim": "Matched: PyTorch",
                "quote": "Frameworks: PyTorch, TensorFlow, scikit-learn",
                "citation": "[sample.pdf p.1 #0]",
                "source": "resume",
            },
            {
                "claim": "Missing: Kubernetes",
                "quote": "Kubernetes and Terraform for infrastructure as code",
                "citation": "[job.pdf p.1 #0]",
                "source": "job_description",
            },
            {
                "claim": "Six years of experience",
                "quote": "6 years building and deploying production AI systems",
                "citation": "[sample.pdf p.1 #1]",
                "source": "resume",
            },
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)
