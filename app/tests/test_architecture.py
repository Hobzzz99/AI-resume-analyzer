"""Constitution Principle I, enforced mechanically.

The claim "the RAG engine is reusable across domains" is easy to make in a README
and easy to violate in the first week of maintenance — one convenient import of a
resume schema into the retriever and the engine is a resume engine forever.

This module walks the AST of every module under ``app/rag/`` and fails if the
engine has grown a dependency on the application, or if a resume-domain word has
appeared in engine code. It is the executable form of SC-008.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ENGINE_DIR = Path(__file__).resolve().parents[1] / "rag"

FORBIDDEN_IMPORT_PREFIXES = (
    "app.services",
    "app.prompts",
    "app.llm",
    "app.api",
    "app.parsers",
)
"""Packages the engine must not import.

``app.schemas.rag``, ``app.config``, and ``app.utils`` are permitted: the first is
the engine's own vocabulary, the second is configuration, and the third is
cross-cutting infrastructure. ``app.schemas.analysis`` is *not* permitted, which
is checked separately below.
"""

DOMAIN_TERMS = frozenset(
    {
        "resume",
        "resumes",
        "recruiter",
        "applicant",
        "hiring",
        "candidacy",
        "skill",
        "skills",
        "employer",
        "job",
    }
)
"""Words that would betray domain knowledge if they appeared as an identifier.

Checked against identifiers only, not comments or strings: an explanatory comment
saying "for resumes, this means..." is documentation, whereas a variable named
``resume_chunks`` is coupling.

Notably absent: ``candidate``. It is standard information-retrieval vocabulary —
a *retrieval* candidate — and banning it would force the reranker to rename its
central parameter to satisfy a linter rather than a design principle. A gate that
produces false positives gets disabled, so it has to be right.
"""


def identifier_words(name: str) -> set[str]:
    """Split an identifier into lowercase words.

    Whole-word matching rather than substring matching, for the same reason:
    substring matching flags ``job`` inside ``adjoining`` and ``cv`` inside
    ``recv``, and a gate that cries wolf is a gate someone deletes.
    """
    spaced = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return {word for word in spaced.lower().split("_") if word}


def engine_modules() -> list[Path]:
    """Every Python module in the engine package."""
    return sorted(path for path in ENGINE_DIR.rglob("*.py") if path.name != "__init__.py")


def imported_modules(tree: ast.AST) -> list[str]:
    """Every module name imported by an AST."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize("module_path", engine_modules(), ids=lambda p: p.name)
def test_engine_does_not_import_application_packages(module_path: Path) -> None:
    """No engine module may depend on the application layer."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    violations = [
        name
        for name in imported_modules(tree)
        if name.startswith(FORBIDDEN_IMPORT_PREFIXES)
    ]
    assert not violations, (
        f"{module_path.name} imports application module(s) {violations}. The RAG engine must "
        f"stay domain-free (Constitution Principle I)."
    )


@pytest.mark.parametrize("module_path", engine_modules(), ids=lambda p: p.name)
def test_engine_does_not_import_domain_schemas(module_path: Path) -> None:
    """The engine may use ``app.schemas.rag`` but never ``app.schemas.analysis``."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    violations = [name for name in imported_modules(tree) if name == "app.schemas.analysis"]
    assert not violations, (
        f"{module_path.name} imports the resume analysis schema. The engine is generic over "
        f"its answer schema and must receive it as a parameter."
    )


@pytest.mark.parametrize("module_path", engine_modules(), ids=lambda p: p.name)
def test_engine_identifiers_are_domain_free(module_path: Path) -> None:
    """No engine identifier may name a resume-domain concept."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.arg):
            name = node.arg
        else:
            continue

        lowered = name.lower()
        # `doc_type` is the single sanctioned exception: the engine needs *a*
        # label to filter on, and the enum's members are values it never inspects.
        if lowered in {"doc_type", "document_type", "documenttype"}:
            continue
        offenders.extend(sorted(identifier_words(name) & DOMAIN_TERMS))

    assert not offenders, (
        f"{module_path.name} contains domain-specific identifier(s) referencing "
        f"{sorted(set(offenders))}. Domain vocabulary belongs in app/services and app/schemas."
    )


def test_engine_exposes_the_protocols_that_make_it_swappable() -> None:
    """Every replaceable component is defined as a Protocol.

    Guards against the regression where a concrete class is imported directly and
    the injection seam quietly disappears (Constitution Principle V).
    """
    from app.rag import base

    required = {
        "DocumentLoader",
        "Embedder",
        "VectorStore",
        "Retriever",
        "Reranker",
        "LLMClient",
        "StructuredGenerator",
        "ResultCache",
    }
    missing = required - set(vars(base))
    assert not missing, f"app/rag/base.py is missing protocol(s): {sorted(missing)}"


def test_prompt_builder_only_accepts_retrieved_chunks() -> None:
    """There must be no code path from a whole document into a prompt.

    Constitution Principle II. Asserted structurally rather than by inspecting a
    generated prompt: if the builder cannot *express* accepting a document, the
    guarantee holds for every call site including ones not yet written.
    """
    import inspect

    from app.rag.prompt_builder import PromptBuilder

    source = inspect.getsource(PromptBuilder)
    assert "SourceDocument" not in source, (
        "PromptBuilder references SourceDocument. Prompts must be assembled from retrieved "
        "chunks only (Constitution Principle II)."
    )
