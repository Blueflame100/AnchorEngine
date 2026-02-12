"""Registry of domain adapters built from configs."""

from pathlib import Path
from typing import Optional

from src.app.core import GrokClient, RAGEngine, load_domain_configs
from src.app.domains.adapter import DomainAdapter


class DomainRegistry:
    """Holds all domain adapters and provides lookup by id."""

    def __init__(
        self,
        configs_dir: Optional[Path] = None,
        grok_api_key: Optional[str] = None,
    ):
        configs = load_domain_configs(configs_dir)
        _project_root = Path(__file__).resolve().parent.parent.parent.parent
        base = configs_dir if configs_dir is not None else _project_root / "configs"
        self._rag = RAGEngine(configs_base=base)
        self._grok = GrokClient(api_key=grok_api_key)
        self._adapters: dict[str, DomainAdapter] = {
            c.domain_id: DomainAdapter(config=c, grok_client=self._grok, rag_engine=self._rag)
            for c in configs
        }
        self._configs = {c.domain_id: c for c in configs}

    def list_domains(self) -> list[dict]:
        """Return list of domain summaries for GET /domains."""
        return [
            {
                "domain_id": c.domain_id,
                "display_name": c.display_name,
                "description": c.description,
            }
            for c in self._configs.values()
        ]

    def get_adapter(self, domain_id: str) -> Optional[DomainAdapter]:
        return self._adapters.get(domain_id)

    def get_config(self, domain_id: str):
        return self._configs.get(domain_id)


_registry: Optional[DomainRegistry] = None


def get_domain_registry(
    configs_dir: Optional[Path] = None,
    grok_api_key: Optional[str] = None,
) -> DomainRegistry:
    """Singleton access to the domain registry."""
    global _registry
    if _registry is None:
        _registry = DomainRegistry(configs_dir=configs_dir, grok_api_key=grok_api_key)
    return _registry
