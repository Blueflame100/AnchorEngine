"""RAG engine: load documents and retrieve relevant chunks for context."""

from pathlib import Path
from typing import Optional

from .config_loader import DomainConfig


class RAGEngine:
    """
    Retrieval-augmented context builder.
    Loads documents from data_dir, chunks them according to config, and retrieves relevant chunks.
    """

    def __init__(self, configs_base: Optional[Path] = None):
        self._configs_base = configs_base or Path(__file__).resolve().parent.parent.parent.parent / "configs"

    def _data_dir_path(self, domain: DomainConfig) -> Optional[Path]:
        """Resolve the data_dir path for a domain."""
        if not domain.data_dir:
            return None
        path = Path(domain.data_dir)
        if not path.is_absolute():
            # Relative to project root
            path = self._configs_base.parent / path
        return path if path.exists() and path.is_dir() else None

    def _chunk_text(self, text: str, domain: DomainConfig) -> list[str]:
        """Chunk text according to domain's chunking configuration."""
        config = domain.rag.chunking
        
        if config.method == "paragraph":
            # Split by paragraphs (double newlines) and respect max_chars
            paragraphs = text.split("\n\n")
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                # If adding this paragraph would exceed max_chars, save current chunk
                if current_chunk and len(current_chunk) + len(para) + 2 > config.max_chars:
                    chunks.append(current_chunk)
                    # Start new chunk with overlap
                    if config.overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-config.overlap:]
                        current_chunk = overlap_text + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
        
        elif config.method == "fixed":
            # Fixed-size chunks with overlap
            chunks = []
            start = 0
            while start < len(text):
                end = start + config.max_chars
                chunk = text[start:end]
                if chunk.strip():
                    chunks.append(chunk.strip())
                start = end - config.overlap
            return chunks
        
        else:
            # Fallback: simple split by newlines
            return [line.strip() for line in text.splitlines() if line.strip()]

    def load_documents(self, domain: DomainConfig) -> list[str]:
        """Load and chunk documents from data_dir for a domain."""
        data_dir = self._data_dir_path(domain)
        if not data_dir:
            return []
        
        chunks = []
        # Load all .txt files from data_dir
        for txt_file in sorted(data_dir.glob("*.txt")):
            try:
                text = txt_file.read_text(encoding="utf-8", errors="ignore")
                file_chunks = self._chunk_text(text, domain)
                chunks.extend(file_chunks)
            except Exception:
                continue
        
        return chunks

    def retrieve(
        self,
        domain: DomainConfig,
        query: str,
    ) -> list[str]:
        """
        Return top_k relevant chunks for the query.
        Uses simple keyword matching; can be replaced with embedding + vector search.
        """
        if not domain.rag.enabled:
            return []
        
        chunks = self.load_documents(domain)
        if not chunks:
            return []
        
        top_k = domain.rag.retrieval.top_k
        
        # Simple relevance scoring: count query word matches
        q_lower = query.lower()
        q_words = set(q_lower.split())
        
        scored = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for word in q_words if word in chunk_lower)
            scored.append((score, chunk))
        
        # Sort by score (descending) and return top_k
        scored.sort(key=lambda x: -x[0])
        return [chunk for _, chunk in scored[:top_k]]

    def build_context(self, domain: DomainConfig, query: str) -> str:
        """Build a single context string from retrieved chunks."""
        if not domain.rag.enabled:
            return ""
        
        chunks = self.retrieve(domain, query)
        if not chunks:
            return ""
        
        return "\n\n".join(chunks)
