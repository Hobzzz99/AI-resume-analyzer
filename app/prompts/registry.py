"""File-backed prompt template registry.

Prompts are the most frequently edited artefact in a RAG system and the one most
worth versioning, so they live in YAML rather than in Python string literals
(FR-022). Three concrete benefits:

* **Editable without a code change.** Tuning a scoring band is a data edit, and a
  non-engineer can review the diff.
* **Versioned by name.** ``resume_analysis_v1`` and a future ``v2`` coexist; a
  request names the template it wants, so an A/B comparison needs no branch.
* **Auditable.** The ``prompt_template`` field in every analysis response records
  which template produced it, which is what makes a historical result
  reproducible.

Templates are compiled to LangChain ``PromptTemplate`` objects for variable
validation: a template referencing ``{job_title}`` while the caller supplies only
``{context}`` fails at format time with a clear error, rather than emitting a
prompt containing a literal ``{job_title}`` to the model.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.utils.exceptions import PromptTemplateNotFoundError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PromptSpec(BaseModel):
    """A loaded prompt template and its metadata."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    system: str = ""
    template: str
    input_variables: list[str] = Field(default_factory=list)

    def compile(self) -> Any:
        """Build a LangChain ``PromptTemplate`` from this spec.

        Imported lazily so the registry can be constructed — and inspected — in a
        test without paying for the LangChain import tree.
        """
        from langchain_core.prompts import PromptTemplate  # noqa: PLC0415

        return PromptTemplate(
            template=self.template,
            input_variables=self.input_variables or self._infer_variables(),
        )

    def _infer_variables(self) -> list[str]:
        """Derive input variables from the template body.

        A fallback for a spec that omits ``input_variables``. Declaring them
        explicitly is preferred, because then a typo in the template becomes a
        load-time error instead of an undeclared variable that silently formats
        to nothing.
        """
        import string  # noqa: PLC0415

        return sorted(
            {
                field
                for _, field, _, _ in string.Formatter().parse(self.template)
                if field
            }
        )


class PromptRegistry:
    """Loads and caches prompt specs from a directory of YAML files.

    Args:
        directory: Directory containing ``*.yaml`` template definitions.
        eager: Load every template at construction. Used at startup so a
            malformed template fails fast rather than during a user's request.
    """

    def __init__(self, directory: Path | str, *, eager: bool = False) -> None:
        self._directory = Path(directory)
        self._specs: dict[str, PromptSpec] = {}
        self._compiled: dict[str, Any] = {}
        if eager:
            self.load_all()

    @property
    def directory(self) -> Path:
        """Directory the registry reads from."""
        return self._directory

    def load_all(self) -> int:
        """Load every template in the directory. Returns how many were loaded."""
        if not self._directory.is_dir():
            logger.warning("prompt directory not found", extra={"path": str(self._directory)})
            return 0
        for path in sorted(self._directory.glob("*.yaml")):
            spec = self._load_file(path)
            self._specs[spec.name] = spec
        logger.info(
            "loaded prompt templates",
            extra={"count": len(self._specs), "names": sorted(self._specs)},
        )
        return len(self._specs)

    def _load_file(self, path: Path) -> PromptSpec:
        """Parse one YAML template file.

        Raises:
            PromptTemplateNotFoundError: The file is unreadable or malformed.
        """
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise PromptTemplateNotFoundError(
                f"Could not read prompt template '{path.name}': {exc}",
                details={"path": str(path)},
            ) from exc
        # Filename wins over an absent name key, so a template can never be
        # registered under a name that does not match its file.
        raw.setdefault("name", path.stem)
        return PromptSpec.model_validate(raw)

    def get(self, name: str) -> PromptSpec:
        """Return the spec registered under ``name``, loading it on demand.

        Raises:
            PromptTemplateNotFoundError: No such template.
        """
        if name in self._specs:
            return self._specs[name]

        path = self._directory / f"{name}.yaml"
        if not path.is_file():
            available = sorted({p.stem for p in self._directory.glob("*.yaml")} | set(self._specs))
            raise PromptTemplateNotFoundError(
                f"Prompt template '{name}' not found. Available: {', '.join(available) or 'none'}.",
                details={"requested": name, "available": available},
            )

        spec = self._load_file(path)
        self._specs[spec.name] = spec
        return spec

    def compiled(self, name: str) -> Any:
        """Return the compiled ``PromptTemplate`` for ``name``, cached.

        Compilation is cached because a plan-driven analysis formats the same
        template on every request and re-parsing it each time is pure waste.
        """
        if name not in self._compiled:
            self._compiled[name] = self.get(name).compile()
        return self._compiled[name]

    def names(self) -> list[str]:
        """Every template name available on disk or already loaded."""
        on_disk = {path.stem for path in self._directory.glob("*.yaml")}
        return sorted(on_disk | set(self._specs))

    def register(self, spec: PromptSpec) -> None:
        """Add a template programmatically.

        Used by tests and by adopters embedding this engine, who may want to
        supply a prompt without writing a file.
        """
        self._specs[spec.name] = spec
        self._compiled.pop(spec.name, None)


@lru_cache(maxsize=4)
def get_registry(directory: str) -> PromptRegistry:
    """Return a cached registry for ``directory``."""
    return PromptRegistry(directory)


def schema_instructions(model: type[BaseModel]) -> str:
    """Render a Pydantic model as compact JSON-schema instructions for a prompt.

    Pydantic's full JSON schema is verbose — ``$defs``, ``anyOf``, titles — and
    spending hundreds of tokens on it measurably degrades attention to the actual
    task. This emits a flat field list with types, constraints, and descriptions:
    everything the model needs to comply, and nothing it does not.
    """
    schema = model.model_json_schema()
    required = set(schema.get("required", []))
    lines: list[str] = ["{"]

    for field_name, spec in schema.get("properties", {}).items():
        if field_name == "grounding_warnings":
            continue  # populated by validation, never by the model
        type_hint = _describe_type(spec)
        constraints = _describe_constraints(spec)
        description = spec.get("description", "")
        marker = "required" if field_name in required else "optional"
        detail = " ".join(part for part in (constraints, description) if part)
        lines.append(f'  "{field_name}": {type_hint},  // {marker}. {detail}'.rstrip())

    lines.append("}")
    return "\n".join(lines)


def _describe_type(spec: dict[str, Any]) -> str:
    """Render a field's type as a JSON-ish literal."""
    kind = spec.get("type")
    if kind == "array":
        items = spec.get("items", {})
        if "$ref" in items or items.get("type") == "object":
            return "[ { ... } ]"
        return f"[{_describe_type(items)}]"
    return {
        "integer": "<int>",
        "number": "<float>",
        "string": '"<string>"',
        "boolean": "<bool>",
    }.get(str(kind), "<value>")


def _describe_constraints(spec: dict[str, Any]) -> str:
    """Render range and length constraints in a compact form."""
    parts: list[str] = []
    if "minimum" in spec or "maximum" in spec:
        parts.append(f"Range {spec.get('minimum', '-inf')}-{spec.get('maximum', 'inf')}.")
    if "maxItems" in spec:
        parts.append(f"Max {spec['maxItems']} items.")
    if "maxLength" in spec:
        parts.append(f"Max {spec['maxLength']} chars.")
    return " ".join(parts)
