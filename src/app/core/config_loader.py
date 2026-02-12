"""Load domain configurations from YAML files in /configs."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, ValidationError


class RAGChunkingConfig(BaseModel):
    """RAG chunking configuration."""

    method: str = Field(default="paragraph", description="Chunking method: 'paragraph' or 'fixed'")
    max_chars: int = Field(default=1000, ge=1, description="Maximum characters per chunk")
    overlap: int = Field(default=100, ge=0, description="Character overlap between chunks")


class RAGRetrievalConfig(BaseModel):
    """RAG retrieval configuration."""

    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")


class RAGConfig(BaseModel):
    """RAG configuration."""

    enabled: bool = Field(default=True, description="Whether RAG is enabled")
    chunking: RAGChunkingConfig = Field(default_factory=RAGChunkingConfig)
    retrieval: RAGRetrievalConfig = Field(default_factory=RAGRetrievalConfig)


class OutputConfig(BaseModel):
    """Output format configuration."""

    format: str = Field(default="text", description="Output format: 'text' or 'json'")
    json_schema: Optional[dict] = Field(default=None, description="Optional JSON schema for structured outputs")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v not in ("text", "json"):
            raise ValueError(f"output.format must be 'text' or 'json', got '{v}'")
        return v


class DomainConfig(BaseModel):
    """Schema for a single domain configuration."""

    domain_id: str = Field(..., description="Unique domain identifier")
    display_name: str = Field(..., description="Human-readable domain name")
    description: str = Field(default="", description="Domain description")
    data_dir: Optional[str] = Field(default=None, description="Path to folder containing .txt documents")
    system_prompt: str = Field(default="", description="System prompt for the domain")
    user_prompt_template: str = Field(
        default="Question: {question}\n\nContext:\n{context}",
        description="User prompt template with {question} and {context} placeholders"
    )
    rag: RAGConfig = Field(default_factory=RAGConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    # Legacy fields for backward compatibility (mapped from old configs)
    model: str = Field(default="grok-2-1212", description="xAI model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class ConfigValidationError(Exception):
    """Raised when a domain config fails validation."""

    def __init__(self, file_path: Path, errors: list[dict]):
        self.file_path = file_path
        self.errors = errors
        error_messages = []
        for error in errors:
            field_path = " -> ".join(str(loc) for loc in error.get("loc", []))
            error_msg = error.get("msg", "Validation error")
            error_type = error.get("type", "value_error")
            error_messages.append(f"  {field_path}: {error_msg} ({error_type})")
        super().__init__(
            f"Invalid config in {file_path}:\n" + "\n".join(error_messages)
        )


def _configs_dir() -> Path:
    """Resolve configs directory: project root / configs."""
    # Support running from project root or from src/
    base = Path(__file__).resolve().parent.parent.parent.parent
    return base / "configs"


def load_domain_config(path: Path) -> DomainConfig:
    """
    Load and validate a single domain config from a YAML file.
    
    Raises ConfigValidationError with structured error details if validation fails.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(
            path,
            [{"loc": [], "msg": f"YAML parsing error: {str(e)}", "type": "yaml_error"}]
        )
    except Exception as e:
        raise ConfigValidationError(
            path,
            [{"loc": [], "msg": f"File read error: {str(e)}", "type": "file_error"}]
        )

    if data is None:
        raise ConfigValidationError(
            path,
            [{"loc": [], "msg": "Config file is empty", "type": "empty_file"}]
        )

    # Handle legacy field names for backward compatibility
    if "id" in data and "domain_id" not in data:
        data["domain_id"] = data.pop("id")
    if "name" in data and "display_name" not in data:
        data["display_name"] = data.pop("name")
    if "documents_path" in data and "data_dir" not in data:
        data["data_dir"] = data.pop("documents_path")

    try:
        return DomainConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigValidationError(path, e.errors())


def load_domain_configs(configs_dir: Optional[Path] = None) -> list[DomainConfig]:
    """
    Load all domain configs from YAML files in the configs directory.
    Expects files named *.yaml or *.yml.
    
    Skips invalid configs and continues loading others.
    """
    directory = configs_dir or _configs_dir()
    if not directory.is_dir():
        return []

    configs: list[DomainConfig] = []
    for path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
        try:
            configs.append(load_domain_config(path))
        except ConfigValidationError as e:
            # Log error but continue loading other configs
            print(f"Warning: Skipping {path.name}: {e}")
            continue
        except Exception as e:
            print(f"Warning: Unexpected error loading {path.name}: {e}")
            continue
    return configs
