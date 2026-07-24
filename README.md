# 🏆 [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July) 2026

> 🚀 This repository is my official submission for the [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

## 👤 Participant

| Field            | Value                                            |
| ---------------- | ------------------------------------------------ |
| Full Name        | _<add your full name>_                           |
| Project Name     | AI Resume Analyzer — a modular RAG engine        |
| GitHub Username   | [Hobzzz99](https://github.com/Hobzzz99)          |
| Challenge Batch  | June–July 2026                                    |
| Training Program | Large Language Models (LLMs) Program             |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en)      |

---

# 📌 Project Overview

The **AI Resume Analyzer** compares a candidate's résumé against a job description and returns a
structured, evidence-backed analysis: match scores, matched and missing skills, strengths,
weaknesses, actionable recommendations, a recruiter summary — and, for every conclusion, the
**exact passage from the source documents** that supports it.

The interesting part is what sits underneath: a **modular, reusable Retrieval-Augmented Generation
(RAG) engine** that knows nothing about résumés. The résumé analyzer is just one application of it —
the same engine could power legal-document analysis, a research assistant, a company knowledge base,
or medical-document search. That separation is not a claim in this README; it is **enforced by an
automated test** that fails the build if the engine ever imports domain-specific code.

Three principles define the system:

1. **Retrieval, never full-context.** The model never sees a whole document — only the passages
   retrieved for each question, under a hard token budget.
2. **Structured, validated output.** Every model response is parsed into a Pydantic model with range
   checks; invalid output triggers a bounded repair-and-retry loop. Unvalidated text never reaches
   the user.
3. **Everything is grounded and auditable.** Each conclusion cites the passage it came from, and
   every response carries a retrieval trace, per-stage timings, and the model that produced it.

```python
# The whole thesis in three lines: one engine, many domains, zero engine changes.
analysis = pipeline.run(plan=resume_plan,  template=resume_tpl,  schema=ResumeAnalysis)
contract = pipeline.run(plan=clause_plan,  template=contract_tpl, schema=ClauseRisk)
```

---

# ✨ Features

- **Reusable RAG engine** — domain-free, generic over the answer schema, enforced by an AST test.
- **16-facet retrieval plan** — instead of one vague query, the analyzer issues 12 résumé facet
  queries (languages, frameworks, cloud, ML/AI, projects, experience, education, certifications,
  leadership, soft skills, achievements, summary) and 4 job-description queries, so low-frequency
  details are actually retrieved.
- **Four retrieval strategies** — similarity, MMR, BM25, and **hybrid** (dense + lexical fused by
  Reciprocal Rank Fusion), selectable from configuration.
- **Grounded evidence with citations** — every claim points to `[file p.PAGE #INDEX]` in the source.
- **Structured output + auto-repair** — Pydantic validation with a bounded repair loop that feeds
  the exact validation errors back to the model.
- **Pluggable LLM providers** — Groq and Google Gemini, swappable with a single `.env` setting,
  both behind the same interface.
- **Streaming progress** — the UI shows each pipeline stage (retrieval → prompt → generation →
  validation) live over Server-Sent Events.
- **Content-fingerprint caching** — re-uploading an identical document performs zero re-embedding.
- **Full observability** — per-stage timings, a retrieval trace, and a correlated request id on
  every response.
- **Retrieval evaluation harness** — measures recall@k, MRR, MAP, and latency per strategy.
- **367 automated tests** that run fully offline (no API key, no network).

---

# 🛠️ Technologies Used

**Backend / RAG engine**
- Python 3.12+
- FastAPI + Uvicorn (web service, SSE streaming)
- LangChain (`PyPDFLoader`, `RecursiveCharacterTextSplitter`, prompt templates)
- Pydantic v2 + pydantic-settings (validation, typed config)
- ChromaDB (vector store)
- `sentence-transformers/all-MiniLM-L6-v2` (embeddings)
- `rank-bm25` (lexical retrieval)

**LLM providers**
- Groq (`llama-3.1-8b-instant` / `llama-3.3-70b-versatile`)
- Google Gemini (`gemini-2.0-flash`) — optional, swappable via config

**Frontend**
- Next.js 15 (App Router) + TypeScript + Tailwind CSS

**Tooling**
- pytest (367 tests), Ruff (lint), mypy (types)

---

# ⚙️ Installation

**Requirements:** Python 3.12+, Node.js 20+, and a free [Groq API key](https://console.groq.com/keys).

### 1. Backend

```bash
# create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: tests & linting

# configure secrets
copy .env.example .env          # Windows  (cp on macOS/Linux)
```

Open `.env` and set your key and model:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

> To use Google Gemini instead, set `LLM_PROVIDER=gemini` and add a
> [free Gemini key](https://aistudio.google.com/apikey) as `GEMINI_API_KEY`.

### 2. Frontend

```bash
cd frontend
copy .env.local.example .env.local   # points at http://localhost:8000
npm install
```

---

# 🚀 Usage

Run the backend and frontend in **two separate terminals**.

**Terminal 1 — API** (API docs at http://localhost:8000/docs):

```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

**Terminal 2 — Frontend** (opens at http://localhost:3000):

```bash
cd frontend
npm run dev
```

Then, in the browser:

1. **Upload your résumé** (PDF / TXT / MD) in panel 1.
2. **Paste or upload a job description** in panel 2, then index it.
3. Click **Run analysis** — the dashboard appears in a few seconds.

> The first analysis downloads the embedding model (~90 MB) once.

**Optional — generate sample documents:**

```bash
python scripts/generate_sample_pdfs.py
# -> data/samples/sample_resume.pdf, sample_job_description.pdf
```

**Optional — run the tests (fully offline):**

```bash
pytest
```

---

# 📸 Demo

> _Add screenshots or a short GIF of the results dashboard here._

Suggested captures:
- The upload panel with a résumé and job description indexed
- The results dashboard (score meters, matched/missing skills)
- The **evidence panel** expanded, showing citations back to the résumé
- The pipeline diagnostics (retrieval trace + stage timings)

---

# 📊 Results

- **End-to-end analysis** completes in **~3–5 seconds** (retrieval + generation + validation).
- **Retrieval evaluation** (bundled labelled set, k=5) shows **hybrid** as the strongest strategy:

  | Strategy   | recall@5  | MRR       | MAP       | hit rate |
  | ---------- | --------- | --------- | --------- | -------- |
  | similarity | 0.767     | 0.658     | 0.623     | 0.80     |
  | mmr        | 0.700     | 0.683     | 0.573     | 0.80     |
  | bm25       | 0.750     | 0.800     | 0.708     | 0.80     |
  | **hybrid** | **0.867** | **0.833** | **0.786** | **0.90** |

- **367 automated tests pass** offline (no API key, no network) — covering loading, cleaning,
  chunking, embeddings, all four retrievers, the prompt builder, the output parser, every API
  route, and an architecture test that guarantees the engine stays domain-free.
- **100% of returned analyses** conform to the declared schema and value ranges — a malformed
  result never reaches the caller.

---

# 🔮 Future Improvements

- Generate the frontend TypeScript types automatically from the OpenAPI schema.
- Add a golden-set **answer-quality evaluation** (LLM-as-judge) on top of the retrieval metrics.
- Persist the BM25 index for larger corpora instead of rebuilding it per query.
- Add authentication and per-user document isolation for multi-user deployment.
- Add OCR support so scanned (image-only) PDFs can be analyzed.
- Support batch comparison of many résumés against one job description.

---

# 📚 About the Challenge

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/)
**Challenge (June–July) 2026**.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of
[**Edrak for Ai**](https://edrak4ai.com/en), and the challenge encourages participants to build
real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming batches, visit the
official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📝 License

This project is shared for educational and portfolio purposes.
