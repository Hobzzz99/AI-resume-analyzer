<!--
Sync Impact Report
==================
Version change: (none) → 1.0.0
Bump rationale: Initial ratification. All principles newly defined.

Modified principles: n/a (initial adoption)
Added sections:
  - Core Principles I–VII
  - Engineering Standards
  - Development Workflow & Quality Gates
  - Governance
Removed sections: none

Template alignment:
  - .specify/templates/plan-template.md            ✅ Constitution Check gate wired in specs/001-*/plan.md
  - .specify/templates/spec-template.md            ✅ no constitution-driven section changes required
  - .specify/templates/tasks-template.md           ✅ observability/testing task categories already supported
  - README.md                                      ✅ authored after this constitution; references Principles I, II, V

Deferred TODOs: none
-->

# Resume RAG Constitution

Governing principles for `resume-rag` — a modular Retrieval-Augmented Generation engine whose
first application is an AI Resume Analyzer. These rules are binding on all code in this
repository.

## Core Principles

### I. Engine/Application Separation (NON-NEGOTIABLE)

The RAG engine (`app/rag/`) MUST NOT import from, reference, or encode knowledge of any
specific domain — resumes, job descriptions, scoring, or recruiting. Domain logic lives in
`app/services/`, `app/prompts/`, and `app/schemas/`. The engine is parameterised by
configuration, injected collaborators, and generic types only.

*Rationale*: The stated goal is reuse across legal, medical, research, and knowledge-base
applications. Any resume-shaped concept leaking into the engine converts a framework into a
one-off script. This principle is verifiable: a static test asserts no engine module imports a
domain module.

### II. Retrieval Is Mandatory, Never Full-Context

The full text of a source document MUST NEVER be sent to the LLM. Every prompt is assembled
exclusively from chunks returned by a retriever, and every chunk placed in a prompt MUST carry
a citation handle traceable to `(document_id, page, chunk_index)`.

*Rationale*: This is the difference between a RAG system and a wrapper around a long-context
call. It bounds cost, bounds latency, and makes every generated claim auditable. A guard in the
prompt builder raises if the assembled context exceeds the configured chunk budget.

### III. Structured Output Is Contractual

Every LLM response consumed by the system MUST be validated against a Pydantic v2 model before
crossing a module boundary. Validation failures trigger a bounded repair-and-retry loop; on
exhaustion the system raises a typed error. Unvalidated model text MUST NOT reach the API layer
or the frontend.

*Rationale*: LLM output is untrusted input. Treating it as a contract — parse, don't validate at
the edges — keeps probabilistic behaviour quarantined in one layer.

### IV. Grounding Over Fluency

Prompts MUST instruct the model to answer only from retrieved context, to emit the literal
string `"Not Found"` when the context does not support a field, and to attach evidence for every
conclusion. Responses whose evidence list is empty are treated as low-confidence and flagged.

*Rationale*: A resume analyzer that invents skills is worse than useless — it is actively
harmful to a candidate. Grounding is a product requirement, not a stylistic preference.

### V. Dependency Injection Everywhere

Embedders, vector stores, retrievers, LLM clients, and caches MUST be defined as `Protocol`
interfaces in `app/rag/base.py` and supplied through constructor injection or the FastAPI
dependency system. No module may construct a network client or load a model at import time.

*Rationale*: This is what makes the suite runnable offline in CI with fakes, and what makes
swapping Chroma for pgvector or Groq for another provider a configuration change rather than a
rewrite.

### VI. Observability By Default

Every pipeline stage (load, split, embed, store, retrieve, rerank, prompt, generate, parse) MUST
emit a structured log record carrying a request-scoped correlation id and its wall-clock
duration. Every API response MUST include a stage-level timing breakdown.

*Rationale*: RAG failures are almost never "the app is broken" — they are "retrieval returned
the wrong three chunks". Stage timings and retrieval traces are the only way to tell those apart
in production.

### VII. Configuration Over Constants

Model identifiers, chunk sizes, top-k, retrieval strategy, temperature, timeouts, and file-size
limits MUST be read from a single typed settings object sourced from the environment. Magic
numbers and hardcoded model names in application code are defects.

*Rationale*: The Groq model catalogue changes faster than this codebase will. Hardcoding a model
id guarantees a stale repository.

## Engineering Standards

- **Language**: Python 3.12+, PEP 8, full type hints on every public function and method.
- **Docstrings**: required on every module, public class, and public function; state *why* the
  design is what it is where the reason is not obvious from the signature.
- **Function size**: prefer functions under ~40 lines with a single responsibility. Extract
  rather than nest.
- **DRY**: any logic appearing a third time MUST be extracted into `app/utils/` or the relevant
  engine module.
- **Errors**: all failure modes raised as typed exceptions deriving from `AppError`
  (`app/utils/exceptions.py`), each mapping to exactly one HTTP status via a single handler.
  Bare `except Exception` without re-raise or structured log is forbidden.
- **Secrets**: never logged, never returned in an API response, never committed. `.env` is
  git-ignored; `.env.example` documents every key.

## Development Workflow & Quality Gates

1. **Spec first**: features enter through `specs/NNN-slug/spec.md`, then `plan.md`, then
   `tasks.md`. Code that has no task is out of scope.
2. **Tests offline**: the entire unit suite MUST pass with no network access and no API key, via
   injected fakes. Tests requiring Groq or a model download MUST be marked
   `@pytest.mark.integration` and excluded from the default run.
3. **Coverage of contracts**: loaders, splitter, embeddings, retriever, prompt builder, output
   parser, and each API route MUST each have direct test coverage.
4. **Definition of done**: task checked in `tasks.md`, tests green, docstrings present,
   `.env.example` updated if a new setting was introduced, README updated if behaviour changed.

## Governance

This constitution supersedes ad hoc preference in code review. Amendments require an entry in
the Sync Impact Report above, a semantic version bump, and propagation to any template or doc
the change invalidates.

- **MAJOR**: a principle is removed or redefined in a backward-incompatible way.
- **MINOR**: a principle or section is added, or guidance materially expanded.
- **PATCH**: clarification, wording, or typo fixes with no semantic change.

Compliance is verified at review time against the Constitution Check gate in `plan.md`. Any
violation must be recorded in that plan's Complexity Tracking table with the simpler alternative
that was rejected and why — an unjustified violation blocks merge.

**Version**: 1.0.0 | **Ratified**: 2026-07-24 | **Last Amended**: 2026-07-24
