"""FastAPI application entry for the config-driven Grok demo engine."""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic_settings import BaseSettings

from src.app.api.routes import router
from src.app.domains.registry import get_domain_registry


def _project_root() -> Path:
    """Project root (directory containing configs/ and .env)."""
    return Path(__file__).resolve().parent.parent


_PROJECT_ROOT = _project_root()
_ENV_FILE = _PROJECT_ROOT / ".env"

# Load .env into os.environ before any settings are read (works regardless of cwd)
# Try project-root .env first, then cwd .env as fallback
_load_result = load_dotenv(_ENV_FILE, override=False)
if not _load_result and Path.cwd() != _PROJECT_ROOT:
    load_dotenv(Path.cwd() / ".env", override=False)


class Settings(BaseSettings):
    """App settings from environment."""

    grok_api_key: str = ""
    configs_dir: Path | None = None

    model_config = {
        "env_file": str(_ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="Config-driven Grok Demo Engine",
        description="Domain configs from YAML, GET /domains, POST /ask with RAG.",
        version="0.1.0",
    )
    settings = Settings()
    # Accept GROK_API_KEY or XAI_API_KEY from .env / environment
    raw = (
        settings.grok_api_key
        or os.environ.get("GROK_API_KEY")
        or os.environ.get("XAI_API_KEY")
        or ""
    )
    api_key = raw.strip() or None

    # Initialize registry once (reads configs, builds adapters)
    get_domain_registry(
        configs_dir=settings.configs_dir,
        grok_api_key=api_key,
    )

    # Startup hint so you can confirm the key is loaded when running uvicorn
    if not api_key or not api_key.strip():
        import sys
        print(
            "WARNING: GROK_API_KEY is not set. Set it in .env (project root) or export GROK_API_KEY.",
            file=sys.stderr,
        )
        print(f"         Looked for .env at: {_ENV_FILE}", file=sys.stderr)
    else:
        import sys
        print(f"GROK_API_KEY loaded from {_ENV_FILE}", file=sys.stderr)

    app.include_router(router, tags=["demo"])

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    return app


app = create_app()
