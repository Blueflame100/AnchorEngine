"""Simple RAG engine: load documents and retrieve relevant chunks for context."""

from pathlib import Path
from typing import Optional

from .config_loader import DomainConfig


class RAGEngine:
    """
    Retrieval-augmented context builder.
    Uses simple keyword/snippet matching for demo; can be swapped for vector search.
    """

    def __init__(self, configs_base: Optional[Path] = None):
        self._configs_base = configs_base or Path(__file__).resolve().parent.parent.parent.parent / "configs"

    def _documents_path(self, domain: DomainConfig) -> Optional[Path]:
        if not domain.documents_path:
            return None
        path = Path(domain.documents_path)
        if not path.is_absolute():
            path = self._configs_base.parent / path
        return path if path.exists() else None

    def load_documents(self, domain: DomainConfig) -> list[str]:
        """Load and split documents for a domain into chunks (one chunk per line or paragraph)."""
        path = self._documents_path(domain)
        if not path or not path.is_file():
            return []
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]
        return chunks

    def retrieve(
        self,
        domain: DomainConfig,
        query: str,
        top_k: int = 5,
    ) -> list[str]:
        """
        Return top_k relevant chunks for the query.
        Demo: returns first top_k chunks; replace with embedding + vector search for production.
        """
        chunks = self.load_documents(domain)
        if not chunks:
            return []
        # Simple relevance: prefer chunks containing query words
        q_lower = query.lower()
        scored = []
        for c in chunks:
            score = sum(1 for w in q_lower.split() if w in c.lower())
            scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_k]]

    def build_context(self, domain: DomainConfig, query: str, top_k: int = 5) -> str:
        """Build a single context string from retrieved chunks."""
        chunks = self.retrieve(domain, query, top_k=top_k)
        if not chunks:
            return ""
        return "\n\n".join(chunks)
