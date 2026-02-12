"""Load domain configurations from YAML files in /configs."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class DomainConfig(BaseModel):
    """Schema for a single domain configuration."""

    id: str
    name: str
    description: str = ""
    system_prompt: str = ""
    model: str = Field(default="grok-2-1212", description="xAI model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    documents_path: Optional[str] = None


def _configs_dir() -> Path:
    """Resolve configs directory: project root / configs."""
    # Support running from project root or from src/
    base = Path(__file__).resolve().parent.parent.parent.parent
    return base / "configs"


def load_domain_config(path: Path) -> DomainConfig:
    """Load and validate a single domain config from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return DomainConfig.model_validate(data)


def load_domain_configs(configs_dir: Optional[Path] = None) -> list[DomainConfig]:
    """
    Load all domain configs from YAML files in the configs directory.
    Expects files named *.yaml or *.yml.
    """
    directory = configs_dir or _configs_dir()
    if not directory.is_dir():
        return []

    configs: list[DomainConfig] = []
    for path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
        try:
            configs.append(load_domain_config(path))
        except Exception:
            continue
    return configs
