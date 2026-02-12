# Config-driven Grok Demo Engine

FastAPI service that loads domain configs from YAML and answers questions via xAI Grok, with optional RAG context.

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
PYTHONPATH=src uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or from `src`:

```bash
cd src && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- **GET /domains** — list domains from `configs/*.yaml`
- **POST /ask** — body: `{"domain_id": "example", "question": "Your question"}`

## Layout

- **core** — domain loader (YAML), RAG engine, Grok client
- **domains** — domain adapters and registry
- **api** — FastAPI routes
- **configs** — one YAML per domain (id, name, system_prompt, model, documents_path, etc.)
