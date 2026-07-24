# Quickstart

**Feature**: 001-resume-rag-analyzer | **Date**: 2026-07-24

## 1. Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # tests, linting
```

## 2. Configure

```bash
cp .env.example .env
```

Set at minimum:

```env
GROQ_API_KEY=gsk_...                 # https://console.groq.com/keys (free tier)
GROQ_MODEL=llama-3.3-70b-versatile   # no default in code — must be set
```

Everything else has a working default. `GET /api/v1/health` reports the resolved values.

## 3. Verify offline

The unit suite runs with no API key and no network:

```bash
pytest -q
```

Integration tests (real Groq call, real MiniLM download) are deselected by default:

```bash
pytest -m integration
```

Architecture gate only — proves the engine has no domain dependencies:

```bash
pytest tests/test_architecture.py -v
```

## 4. Generate sample documents

```bash
python scripts/generate_sample_pdfs.py
# → data/samples/sample_resume.pdf
# → data/samples/sample_job_description.pdf
```

## 5. Run

Terminal 1 — API:
```bash
uvicorn app.main:app --reload --port 8000
```
Docs at http://localhost:8000/docs

Terminal 2 — UI (Node 20+):
```bash
cd frontend
cp .env.local.example .env.local     # points at http://localhost:8000
npm install
npm run dev
```
UI at http://localhost:3000

First analysis downloads `all-MiniLM-L6-v2` (~90 MB) once.

## 6. End-to-end via curl

```bash
# health
curl -s localhost:8000/api/v1/health | jq

# upload resume
RESUME=$(curl -s -F "file=@data/samples/sample_resume.pdf" \
  localhost:8000/api/v1/upload/resume | jq -r .document_id)

# upload job description (file or pasted text)
JOB=$(curl -s -H 'Content-Type: application/json' \
  -d '{"text":"Senior ML Engineer. Required: Python, PyTorch, Kubernetes, AWS, MLOps.","title":"Senior ML Engineer"}' \
  localhost:8000/api/v1/upload/job | jq -r .document_id)

# analyze
curl -s -H 'Content-Type: application/json' \
  -d "{\"resume_document_id\":\"$RESUME\",\"job_document_id\":\"$JOB\"}" \
  localhost:8000/api/v1/analyze | jq '.analysis.overall_score, .analysis.missing_skills, .timings'

# streaming
curl -N -H 'Content-Type: application/json' \
  -d "{\"resume_document_id\":\"$RESUME\",\"job_document_id\":\"$JOB\"}" \
  localhost:8000/api/v1/analyze/stream
```

## 7. Validate the acceptance criteria

| Criterion | How to check |
|---|---|
| SC-003 conformance | `pytest tests/test_analysis_schema.py tests/test_parsers.py` |
| SC-005 bounded context | `pytest tests/test_prompt_builder.py -k budget` |
| SC-006 no re-embedding | Upload the same resume twice; second manifest has `cached: true`, `embed_ms: null` |
| SC-008 engine reuse | `pytest tests/test_pipeline.py -k different_schema` |
| SC-009 offline suite | `pytest -q` with `GROQ_API_KEY` unset and no network |
| SC-010 stage timings | Any `/analyze` response body → `timings` |

## 8. Evaluate retrieval quality

```bash
python scripts/evaluate.py --dataset data/eval/retrieval_cases.json
# reports recall@k, MRR, mean latency per strategy (similarity / mmr / hybrid)
```
