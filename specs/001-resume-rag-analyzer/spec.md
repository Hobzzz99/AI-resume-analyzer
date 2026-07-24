# Feature Specification: AI Resume Analyzer on a Reusable Retrieval Engine

**Feature Branch**: `001-resume-rag-analyzer`

**Created**: 2026-07-24

**Status**: Draft

**Input**: User description: "Build an AI Resume Analyzer powered by a modular, reusable RAG engine. The user uploads a resume and a job description; the system reads, chunks, embeds, stores, retrieves, prompts, generates, validates, and displays a structured analysis. Never send full documents to the model."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Analyze a resume against a job description (Priority: P1)

A job seeker has a resume as a PDF and a job description as either a PDF or pasted text. They
submit both and receive a structured verdict: how well the resume matches the role, which
required skills are present, which are missing, what to fix, and — critically — the exact
passages from their own documents that justify each conclusion.

**Why this priority**: This is the entire product. Everything else is scaffolding around it. If
only this story ships, a candidate can already get real value.

**Independent Test**: Submit a known resume and a known job description; assert the response
contains numeric scores in range, non-empty matched/missing skill lists, and at least one
evidence string that appears verbatim in one of the uploaded documents.

**Acceptance Scenarios**:

1. **Given** a valid resume document and a valid job description, **When** the user requests an
   analysis, **Then** the system returns scores for overall, technical, experience, education,
   and ATS fit, plus strengths, weaknesses, matched skills, missing skills, recommendations, a
   recruiter summary, a confidence value, and supporting evidence.
2. **Given** a resume that does not mention any cloud platform and a job description that
   requires one, **When** the analysis runs, **Then** the missing-skills list names that cloud
   requirement and the recommendations reference it.
3. **Given** a job description asking for a qualification the resume is silent on, **When** the
   analysis runs, **Then** the corresponding field reports `Not Found` rather than an invented
   value.
4. **Given** an analysis result, **When** the user inspects any conclusion, **Then** each piece
   of evidence is attributable to a named source document and a location within it.
5. **Given** two documents totalling far more text than a single model request could carry,
   **When** the analysis runs, **Then** only a bounded subset of the documents' passages is used
   to produce the answer.

---

### User Story 2 - Upload and prepare documents (Priority: P1)

Before analysis, the user submits a resume file and a job description (file or pasted text). The
system extracts the text, normalises it, breaks it into passages, indexes those passages for
retrieval, and confirms how many passages were created. Re-submitting an identical document does
not repeat the indexing work.

**Why this priority**: Analysis is impossible without it, and it is the stage where most real
failures occur (scanned PDFs, empty files, garbage extraction). It must fail loudly and
informatively.

**Independent Test**: Upload a document, confirm the response reports a stable identifier and a
passage count greater than zero; upload the identical bytes again and confirm the system reports
a cached result and creates no additional passages.

**Acceptance Scenarios**:

1. **Given** a text-bearing PDF, **When** the user uploads it, **Then** the system reports a
   document identifier, the source filename, the detected page count, and the number of passages
   indexed.
2. **Given** a PDF containing no extractable text (e.g. a pure scan), **When** the user uploads
   it, **Then** the system rejects it with an explanation naming the cause, and indexes nothing.
3. **Given** a file that is not a supported document type, **When** the user uploads it,
   **Then** the system rejects it before attempting extraction.
4. **Given** a file larger than the configured limit, **When** the user uploads it, **Then** the
   system rejects it without reading the whole file into memory.
5. **Given** a job description supplied as pasted text rather than a file, **When** the user
   submits it, **Then** it is prepared identically to an uploaded file.

---

### User Story 3 - Inspect and trust the result (Priority: P2)

The user reviews the analysis in a visual dashboard: scores as progress indicators, skills as
lists, and every conclusion expandable to the passage that produced it. They can see how
confident the system is and where that confidence came from.

**Why this priority**: A score with no justification is a horoscope. Trust is what makes the
output actionable, but the underlying analysis (P1) must exist first.

**Independent Test**: With a completed analysis, confirm the interface renders every field of
the result, and that evidence entries display their source document and location.

**Acceptance Scenarios**:

1. **Given** a completed analysis, **When** the user opens the results view, **Then** all five
   scores render as labelled progress indicators with their numeric values.
2. **Given** a completed analysis, **When** the user expands the evidence section, **Then** each
   evidence entry shows the passage text and its source document and page.
3. **Given** an analysis where the system had thin supporting context, **When** the user views
   the result, **Then** a low-confidence indicator is visible.

---

### User Story 4 - Operate the service (Priority: P2)

An operator needs to know the service is alive, which model is configured, whether the index is
reachable, and where time is going on each request.

**Why this priority**: Required for any deployment beyond a laptop, but it does not block the
candidate-facing value.

**Independent Test**: Query the health endpoint with the service running and confirm it reports
component status; run an analysis and confirm the response and logs carry a per-stage timing
breakdown and a correlation identifier.

**Acceptance Scenarios**:

1. **Given** a running service, **When** the operator checks health, **Then** the response
   reports overall status, the configured generation model, the embedding model, and the
   reachability of the index.
2. **Given** any analysis request, **When** it completes or fails, **Then** the logs contain one
   correlated trace with durations for retrieval, generation, and validation.
3. **Given** the generation provider is unreachable, **When** an analysis is requested, **Then**
   the user receives an explicit upstream-failure message rather than a generic error or a
   fabricated result.

---

### User Story 5 - Reuse the engine for another document domain (Priority: P3)

An engineer adopting this repository points the same retrieval engine at a different corpus —
contracts, papers, policies — by supplying a new answer shape and a new prompt, without editing
the retrieval code.

**Why this priority**: It is the architectural thesis of the project and the reason a reviewer
would rate it above a tutorial, but it delivers no end-user function on its own.

**Independent Test**: Define an alternative answer schema and prompt, run the pipeline over a
non-resume document, and obtain a validated result with zero changes to files under the engine
package.

**Acceptance Scenarios**:

1. **Given** a new answer shape and prompt template, **When** the pipeline is invoked with them,
   **Then** a validated result of that new shape is produced.
2. **Given** the engine package, **When** its dependencies are inspected, **Then** no module in
   it references resume- or hiring-specific concepts.

---

### Edge Cases

- **Empty or image-only PDF** → rejected with a cause-naming message; nothing is indexed.
- **Corrupt or password-protected PDF** → rejected as unreadable; no partial index is left behind.
- **Resume and job description are unrelated** (e.g. a chef's resume, a compiler-engineer role)
  → low scores, populated missing-skills list, no fabricated matches.
- **Job description is one line** → analysis still returns, with reduced confidence and `Not Found`
  in fields the context cannot support.
- **Retrieval returns nothing** for a query → the system reports insufficient context rather than
  prompting the model with an empty context block.
- **Model returns text that is not the expected shape** → the system repairs and retries within a
  bounded budget, then fails explicitly.
- **Model returns the correct shape with out-of-range values** (e.g. a score of 140) → rejected by
  validation and retried.
- **Provider timeout or rate limit** → surfaced as an upstream error with a retry indication, never
  as an empty or partial analysis.
- **Concurrent uploads of the same document** → one indexing operation, one identifier.
- **Analysis requested with an unknown document identifier** → explicit not-found response.

## Requirements *(mandatory)*

### Functional Requirements

**Ingestion**

- **FR-001**: System MUST accept resumes as PDF and job descriptions as either PDF or plain text.
- **FR-002**: System MUST extract text with page-level attribution preserved.
- **FR-003**: System MUST normalise extracted text — collapse runs of whitespace, repair
  line-break artefacts from PDF extraction, and strip control characters — before indexing.
- **FR-004**: System MUST reject documents that yield no usable text, naming the reason.
- **FR-005**: System MUST reject files above a configurable size limit and unsupported types.
- **FR-006**: System MUST divide document text into overlapping passages using configurable
  passage size and overlap.
- **FR-007**: System MUST record, for every passage, its source filename, page, position within
  the document, document role (resume or job description), owning document identifier, and
  ingestion timestamp.
- **FR-008**: System MUST compute embeddings for each passage exactly once and MUST NOT recompute
  them for content it has already indexed.
- **FR-009**: System MUST persist passages and their embeddings so that they survive a restart.

**Retrieval**

- **FR-010**: System MUST retrieve passages by semantic similarity to a query.
- **FR-011**: System MUST offer a diversity-aware retrieval mode that reduces redundancy among
  returned passages.
- **FR-012**: System MUST offer a hybrid mode combining semantic similarity with keyword matching.
- **FR-013**: System MUST support restricting retrieval by passage metadata, at minimum by
  document identifier and document role.
- **FR-014**: System MUST expose the number of passages returned as a configurable value.
- **FR-015**: Each retrieved passage MUST be returned with its text, its metadata, and a
  relevance score.
- **FR-016**: System MUST support an optional re-ranking stage that reorders candidates by a more
  precise relevance judgement before they enter a prompt.

**Prompting & generation**

- **FR-017**: System MUST assemble prompts from retrieved passages only; it MUST NOT place a full
  source document into a prompt.
- **FR-018**: Every prompt MUST contain: role instructions, the retrieved resume passages, the
  retrieved job-description passages, the required answer shape, formatting rules, and explicit
  anti-fabrication instructions.
- **FR-019**: Every passage placed in a prompt MUST carry a citation handle the model is
  instructed to reference.
- **FR-020**: Prompts MUST instruct the model to return the literal value `Not Found` for any
  field the retrieved context does not support.
- **FR-021**: The generation model identifier MUST be supplied by configuration, never hardcoded
  in application logic.
- **FR-022**: Prompt templates MUST be editable without changing application code.

**Validation**

- **FR-023**: System MUST validate every model response against a declared answer shape covering:
  overall, technical, experience, education, and ATS scores; matched skills; missing skills;
  strengths; weaknesses; recommendations; recruiter summary; confidence; and evidence.
- **FR-024**: System MUST enforce value ranges — scores 0–100, confidence 0–1 — and reject
  responses that violate them.
- **FR-025**: On validation failure, System MUST retry a bounded number of times, feeding the
  validation error back to the model, before failing explicitly.
- **FR-026**: System MUST NOT return unvalidated model output to a caller.

**Analysis content**

- **FR-027**: Analysis MUST assess programming languages, frameworks and libraries, cloud
  platforms, machine learning and AI experience, projects, work experience, education,
  leadership, certifications, soft skills, and applicant-tracking-system compatibility.
- **FR-028**: Analysis MUST produce an overall match verdict, strengths, weaknesses, missing
  skills, actionable recommendations, a recruiter-facing summary, and evidence.
- **FR-029**: Every evidence entry MUST be traceable to a specific source document and location.

**Interfaces**

- **FR-030**: System MUST expose operations to submit a resume, submit a job description, request
  an analysis, and report health.
- **FR-031**: System MUST return distinct, meaningful failure responses for: unreadable input,
  unknown document, insufficient retrieved context, upstream generation failure, validation
  exhaustion, and timeout.
- **FR-032**: System MUST provide a visual interface covering document submission, analysis
  triggering, and a results dashboard showing scores, skill lists, recommendations, and evidence.

**Operability**

- **FR-033**: System MUST log the duration of embedding, retrieval, generation, and validation,
  plus total request duration, under a single correlation identifier.
- **FR-034**: System MUST log errors and warnings with enough context to identify the failing
  stage and document.
- **FR-035**: Analysis responses MUST carry a stage-level timing breakdown.

**Reuse**

- **FR-036**: The retrieval engine MUST be usable with a different answer shape and prompt without
  modification to engine code.
- **FR-037**: Engine components (extraction, splitting, embedding, storage, retrieval, prompt
  assembly, validation) MUST each be replaceable independently.

### Key Entities

- **Document**: a submitted resume or job description. Identity derived from its content, so
  identical submissions are the same document. Attributes: identifier, original filename, role
  (resume or job description), page count, ingestion timestamp, content fingerprint.
- **Passage**: a contiguous, overlapping slice of a document's text. Attributes: identifier,
  owning document, page, position index, text, and the metadata used for filtering.
- **Retrieved Passage**: a passage returned for a query, carrying a relevance score and the rank
  at which it was returned.
- **Answer Shape**: the declared structure a model response must satisfy. For this feature, the
  resume analysis: five scores, four string lists, a recruiter summary, a confidence value, and
  an evidence list.
- **Analysis**: a validated answer plus provenance — the documents analysed, the passages used,
  the model that produced it, timing, and a correlation identifier.
- **Analysis Session**: the pairing of one resume and one job description for a single analysis.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user with two prepared documents obtains a complete analysis in under 30 seconds
  end to end.
- **SC-002**: Document preparation for a typical two-page resume completes in under 5 seconds
  after the embedding model is warm.
- **SC-003**: 100% of returned analyses conform to the declared answer shape and value ranges —
  no caller ever receives a malformed result.
- **SC-004**: 100% of evidence entries resolve to a passage that exists in one of the submitted
  documents.
- **SC-005**: The volume of document text placed in any single prompt never exceeds the
  configured passage budget, regardless of source document length.
- **SC-006**: Re-submitting an identical document performs zero additional embedding work.
- **SC-007**: Every failure listed in Edge Cases produces a distinct, actionable message; none
  produces a generic failure or a partially populated analysis.
- **SC-008**: A new domain (different answer shape and prompt) can be run through the engine with
  zero edits to engine files, demonstrated by an executable example.
- **SC-009**: The automated test suite runs to completion without network access and without
  provider credentials.
- **SC-010**: Every analysis response and log trace attributes time to each pipeline stage.

## Assumptions

- Single-user, single-node deployment; no authentication, multi-tenancy, or per-user data
  isolation is required for this version. Documents are namespaced by identifier, not by owner.
- PDFs are digital-native (text extractable). Optical character recognition for scanned documents
  is explicitly out of scope; such files are rejected with a clear message.
- Documents are in English. Multilingual analysis is out of scope.
- Analysis is one-shot: one resume against one job description. Batch comparison of many resumes
  is out of scope for this version.
- The generation provider is a hosted API subject to rate limits on a free tier; the system must
  degrade with an explicit error rather than retry indefinitely.
- The embedding model runs locally on CPU; first use may incur a one-time model download.
- Persistence is local to the host filesystem. Distributed or managed vector storage is out of
  scope.
- The visual interface is a thin client over the service API; interface polish is secondary to
  the correctness and observability of the analysis.
- No personal data is transmitted anywhere except to the configured generation provider as
  retrieved passages; no analytics or third-party telemetry is included.
