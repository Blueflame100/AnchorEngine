"""FastAPI application entry for the config-driven Grok demo engine."""

from pathlib import Path

from fastapi import FastAPI
from pydantic_settings import BaseSettings

from src.app.api.routes import router
from src.app.domains.registry import get_domain_registry


class Settings(BaseSettings):
    """App settings from environment."""

    grok_api_key: str = ""
    configs_dir: Path | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Config-driven Grok Demo Engine",
        description="Domain configs from YAML, GET /domains, POST /ask with RAG.",
        version="0.1.0",
    )
    settings = Settings()

    # Initialize registry once (reads configs, builds adapters)
    get_domain_registry(
        configs_dir=settings.configs_dir,
        grok_api_key=settings.grok_api_key or None,
    )

    app.include_router(router, tags=["demo"])

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    return app


app = create_app()
