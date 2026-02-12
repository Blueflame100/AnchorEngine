"""RAG engine: load documents and retrieve relevant chunks for context."""

import hashlib
from pathlib import Path
from typing import Optional

from .config_loader import DomainConfig, RAGChunkingConfig


# ---------------------------------------------------------------------------
# Embedding model singleton (all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = None


def _get_embedding_model():
    """Lazy-load sentence-transformers model once."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


class _SentenceTransformerEmbeddingFn:
    """Chroma-compatible embedding function using sentence-transformers."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed documents (used when adding to collection)."""
        model = _get_embedding_model()
        embeddings = model.encode(input, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    def embed_query(self, input: list[str] | str) -> list[list[float]]:
        """Embed query (used when querying). Chroma passes list of query strings."""
        texts = [input] if isinstance(input, str) else input
        return self(texts)


# ---------------------------------------------------------------------------
# RAGStore: build_from_dir + query with Chroma, cached per (data_dir, chunking)
# ---------------------------------------------------------------------------

def _chunk_text_paragraph(text: str, max_chars: int, overlap: int) -> list[str]:
    """Paragraph-based chunking with max_chars and overlap."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current_chunk and len(current_chunk) + len(para) + 2 > max_chars:
            chunks.append(current_chunk)
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para
        else:
            current_chunk = (current_chunk + "\n\n" + para) if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _cache_key(data_dir: Path, chunking: RAGChunkingConfig) -> str:
    """Deterministic cache key for (data_dir, chunking settings)."""
    s = f"{data_dir.resolve()}|{chunking.method}|{chunking.max_chars}|{chunking.overlap}"
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# Module-level cache: cache_key -> RAGStore (collection + metadata)
_RAG_STORE_CACHE: dict[str, "_RAGStore"] = {}


class _RAGStore:
    """
    Internal store: builds Chroma index from data_dir, queries with embeddings.
    Cached per (data_dir, chunking_config).
    """

    def __init__(self, data_dir: Path, chunking_config: RAGChunkingConfig):
        import chromadb
        self._data_dir = data_dir.resolve()
        self._chunking = chunking_config
        self._embedding_fn = _SentenceTransformerEmbeddingFn()

        # In-memory Chroma client
        self._client = chromadb.Client()
        cid = _cache_key(self._data_dir, chunking_config)
        self._collection = self._client.create_collection(
            name=f"rag_{cid}",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._build()

    def _build(self) -> None:
        """Load .txt files, chunk, embed, add to Chroma."""
        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for txt_path in sorted(self._data_dir.glob("*.txt")):
            try:
                text = txt_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            method = self._chunking.method
            max_chars = self._chunking.max_chars
            overlap = self._chunking.overlap

            if method == "paragraph":
                chunk_texts = _chunk_text_paragraph(text, max_chars, overlap)
            elif method == "fixed":
                chunk_texts = []
                start = 0
                while start < len(text):
                    end = start + max_chars
                    chunk = text[start:end].strip()
                    if chunk:
                        chunk_texts.append(chunk)
                    start = end - overlap
            else:
                chunk_texts = [line.strip() for line in text.splitlines() if line.strip()]

            source = txt_path.name
            for i, chunk in enumerate(chunk_texts):
                doc_id = f"{source}:{i}"
                documents.append(chunk)
                metadatas.append({"source": source})
                ids.append(doc_id)

        if documents:
            self._collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

    def query(self, question: str, top_k: int) -> list[dict]:
        """Query and return list of {source, text, score}."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[question],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns cosine distance; convert to similarity score (1 - distance)
        out = []
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists):
            score = 1.0 - float(dist) if dist is not None else 1.0
            out.append({
                "source": meta.get("source", "unknown"),
                "text": doc,
                "score": round(score, 4),
            })
        return out

    def get_all_chunks(self) -> list[str]:
        """Return all chunk texts (for load_documents compatibility)."""
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=["documents"])
        return result["documents"] or []


def _get_or_build_store(
    data_dir: Path,
    chunking_config: RAGChunkingConfig,
) -> Optional[_RAGStore]:
    """Get cached RAGStore or build and cache."""
    if not data_dir.exists() or not data_dir.is_dir():
        return None

    key = _cache_key(data_dir, chunking_config)
    if key not in _RAG_STORE_CACHE:
        _RAG_STORE_CACHE[key] = _RAGStore(data_dir, chunking_config)
    return _RAG_STORE_CACHE[key]


# ---------------------------------------------------------------------------
# RAGEngine: public API (DomainAdapter / registry)
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Retrieval-augmented context builder.
    Uses Chroma + sentence-transformers (all-MiniLM-L6-v2).
    Index cached per (data_dir, chunking settings).
    """

    def __init__(self, configs_base: Optional[Path] = None):
        self._configs_base = configs_base or Path(__file__).resolve().parent.parent.parent.parent / "configs"

    def _data_dir_path(self, domain: DomainConfig) -> Optional[Path]:
        """Resolve the data_dir path for a domain."""
        if not domain.data_dir:
            return None
        path = Path(domain.data_dir)
        if not path.is_absolute():
            path = self._configs_base.parent / path
        return path if path.exists() and path.is_dir() else None

    def load_documents(self, domain: DomainConfig) -> list[str]:
        """Load and chunk documents from data_dir for a domain."""
        store = self._get_store(domain)
        if store is None:
            return []
        return store.get_all_chunks()

    def retrieve(self, domain: DomainConfig, query: str) -> list[str]:
        """Return top_k relevant chunks for the query (semantic search)."""
        if not domain.rag.enabled:
            return []
        store = self._get_store(domain)
        if store is None:
            return []
        top_k = domain.rag.retrieval.top_k
        results = store.query(query, top_k=top_k)
        return [r["text"] for r in results]

    def build_context(self, domain: DomainConfig, query: str) -> str:
        """Build a single context string from retrieved chunks."""
        if not domain.rag.enabled:
            return ""
        chunks = self.retrieve(domain, query)
        if not chunks:
            return ""
        return "\n\n".join(chunks)

    def _get_store(self, domain: DomainConfig) -> Optional[_RAGStore]:
        """Get or build RAGStore for domain's data_dir + chunking."""
        data_dir = self._data_dir_path(domain)
        if data_dir is None:
            return None
        return _get_or_build_store(data_dir, domain.rag.chunking)


# ---------------------------------------------------------------------------
# Standalone RAGStore-like API (build_from_dir, query)
# ---------------------------------------------------------------------------

def build_from_dir(
    data_dir: Path,
    chunking_config: RAGChunkingConfig,
) -> "_RAGStore":
    """Build a RAGStore from a directory of .txt files. Cached by (data_dir, chunking)."""
    key = _cache_key(data_dir.resolve(), chunking_config)
    if key not in _RAG_STORE_CACHE:
        _RAG_STORE_CACHE[key] = _RAGStore(data_dir, chunking_config)
    return _RAG_STORE_CACHE[key]


def query(
    data_dir: Path,
    chunking_config: RAGChunkingConfig,
    question: str,
    top_k: int = 5,
) -> list[dict]:
    """Query a RAG store: returns list of {source, text, score}."""
    store = build_from_dir(data_dir, chunking_config)
    return store.query(question, top_k=top_k)
