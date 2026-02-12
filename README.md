# Config-driven Grok Demo Engine

FastAPI service that loads domain configs from YAML and answers questions via xAI Grok, with optional RAG context and hallucination mitigation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Optional: create a `.env` file in the **project root** (same folder as `configs/`). The app loads it from there no matter where you start the server. Use exactly:

```
GROK_API_KEY=your_xai_api_key
```

(No spaces around `=`. You can also set `GROK_API_KEY` in your shell environment.)

## Run

From repo root (so `configs/` is found):

```bash
PYTHONPATH=src uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

- **GET /domains** — list domains from `configs/*.yaml`
- **POST /ask** — body: `{"domain_id": "iam", "question": "Your question"}`

## Docker

Reproducible run via Docker (uses mock LLM by default):

```bash
docker compose up --build
```

Swagger UI: http://localhost:8000/docs

```bash
# List domains
curl http://localhost:8000/domains

# Ask a question (mock mode returns deterministic answers)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"domain_id": "iam", "question": "How often should access keys be rotated?"}'
```

To use real Grok instead of the mock: create a `.env` in the project root with:

```
USE_MOCK_LLM=false
GROK_API_KEY=your_xai_api_key
```

Then run `docker compose up --build`. Compose loads `.env` for variable substitution.

## Hallucination Mitigation

When RAG is enabled, the pipeline enforces strict grounding to reduce hallucinations:

1. **Context-only answers** — The model is instructed to answer only from retrieved excerpts; otherwise it must refuse with "I don't know based on the provided documents."
2. **Mandatory citations** — Every factual statement must cite an excerpt by `excerpt_id` (numbered [1], [2], …). Missing citations trigger refusal.
3. **Structured JSON output** — Responses follow `{answer, confidence, citations}`; the model outputs no markdown or extra text.
4. **Post-hoc grounding checks** — Code validates that cited excerpt IDs exist; invalid or empty citations are replaced with a safe refusal response.
5. **Invalid JSON fallback** — If the model returns invalid JSON, the response is replaced with a safe refusal.

RAG responses always return `{answer, confidence, citations}`. Set `USE_MOCK_LLM=true` for deterministic mock behavior in tests.

## Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

## Eval

```bash
PYTHONPATH=src python scripts/eval_domain.py --domain_id iam --limit 20
```

Uses `eval/<domain_id>.jsonl` (each line: `{"question": "...", "should_refuse": bool}`). Runs in `USE_MOCK_LLM=true` mode.

## Layout

- **core** — domain loader (YAML), RAG engine, Grok client, grounding module
- **domains** — domain adapters and registry
- **api** — FastAPI routes
- **configs** — one YAML per domain (domain_id, data_dir, system_prompt, RAG settings, etc.)
