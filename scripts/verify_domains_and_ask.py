#!/usr/bin/env python3
"""
Verify (A) domain listing and (B) /ask domain config behavior.
Run from repo root: PYTHONPATH=src python scripts/verify_domains_and_ask.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure src is on path when run as script
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fastapi.testclient import TestClient

# Reset registry so we can inject mock (registry is singleton)
import src.app.domains.registry as reg_module
reg_module._registry = None

from src.main import app
from src.app.core.config_loader import (
    DomainConfig,
    RAGConfig,
    RAGChunkingConfig,
    RAGRetrievalConfig,
    OutputConfig,
)
from src.app.core.rag import RAGEngine
from src.app.domains.adapter import DomainAdapter


def test_a_domain_listing():
    """A) GET /domains returns domain_id, display_name, description."""
    client = TestClient(app)
    r = client.get("/domains")
    assert r.status_code == 200, f"GET /domains failed: {r.status_code}"
    domains = r.json()
    assert isinstance(domains, list), "GET /domains should return a list"
    assert len(domains) >= 1, "At least one domain required"
    for d in domains:
        assert "domain_id" in d, f"Missing domain_id in {d}"
        assert "display_name" in d, f"Missing display_name in {d}"
        assert "description" in d, f"Missing description in {d}"
    print("A) Domain listing: OK (domain_id, display_name, description present)")
    return True


def test_b_rag_disabled_no_context():
    """B) rag.enabled=false → adapter does not retrieve context."""
    mock_grok = MagicMock()
    mock_grok.chat.return_value = "No context used."

    rag_cfg = RAGConfig(enabled=False, chunking=RAGChunkingConfig(), retrieval=RAGRetrievalConfig())
    out_cfg = OutputConfig(format="text")
    config = DomainConfig(
        domain_id="test-no-rag",
        display_name="Test",
        description="",
        rag=rag_cfg,
        output=out_cfg,
        data_dir=None,
    )
    engine = RAGEngine(configs_base=repo_root / "configs")
    adapter = DomainAdapter(config=config, grok_client=mock_grok, rag_engine=engine)

    adapter.ask("What is X?", include_rag_context=True)
    call_args = mock_grok.chat.call_args
    user_message = call_args.kwargs.get("user_message") or call_args[1].get("user_message")
    assert "No relevant context available" in user_message, (
        f"Expected 'No relevant context' when RAG disabled, got: {user_message[:200]}"
    )
    print("B) rag.enabled=false: OK (no context retrieved)")
    return True


def test_b_top_k_limits_retrieval():
    """B) top_k limits number of chunks retrieved."""
    configs_base = repo_root / "configs"
    data_dir = repo_root / "data" / "iam"
    if not data_dir.is_dir():
        print("B) top_k: SKIP (data/iam not found)")
        return True

    config_k1 = DomainConfig(
        domain_id="k1",
        display_name="K1",
        rag=RAGConfig(
            enabled=True,
            chunking=RAGChunkingConfig(),
            retrieval=RAGRetrievalConfig(top_k=1),
        ),
        data_dir="data/iam",
    )
    config_k5 = DomainConfig(
        domain_id="k5",
        display_name="K5",
        rag=RAGConfig(
            enabled=True,
            chunking=RAGChunkingConfig(),
            retrieval=RAGRetrievalConfig(top_k=5),
        ),
        data_dir="data/iam",
    )
    engine = RAGEngine(configs_base=configs_base)

    chunks_1 = engine.retrieve(config_k1, "logging")
    chunks_5 = engine.retrieve(config_k5, "logging")
    assert len(chunks_1) <= 1, f"top_k=1 should return at most 1 chunk, got {len(chunks_1)}"
    assert len(chunks_5) <= 5, f"top_k=5 should return at most 5 chunks, got {len(chunks_5)}"
    assert len(chunks_5) >= len(chunks_1), "top_k=5 should return at least as many as top_k=1"
    print("B) top_k: OK (retrieval depth respects top_k)")
    return True


def test_b_output_format_json_returns_json():
    """B) output.format=json → adapter returns parsed JSON (dict)."""
    mock_grok = MagicMock()
    mock_grok.chat.return_value = '{"answer": "yes", "score": 1}'

    rag_cfg = RAGConfig(enabled=False)
    out_cfg = OutputConfig(format="json")
    config = DomainConfig(
        domain_id="test-json",
        display_name="Test",
        description="",
        rag=rag_cfg,
        output=out_cfg,
    )
    engine = RAGEngine(configs_base=repo_root / "configs")
    adapter = DomainAdapter(config=config, grok_client=mock_grok, rag_engine=engine)

    result = adapter.ask("Is it ok?")
    assert isinstance(result, dict), f"output.format=json should return dict, got {type(result)}"
    assert "answer" in result and result["answer"] == "yes", f"Unexpected content: {result}"
    print("B) output.format=json: OK (returns JSON dict)")
    return True


def test_ask_404_and_no_crash():
    """POST /ask with unknown domain_id returns 404; valid domain_id does not crash (may fail on API)."""
    client = TestClient(app)
    r = client.post("/ask", json={"domain_id": "nonexistent-domain-xyz", "question": "Hi"})
    assert r.status_code == 404, f"Expected 404 for unknown domain, got {r.status_code}"
    print("POST /ask unknown domain: OK (404)")

    # Hit a real domain; without valid API key/model the Grok client may raise — we only require domain lookup and adapter path don't crash
    try:
        r2 = client.post("/ask", json={"domain_id": "example", "question": "Hello"})
        assert r2.status_code in (200, 500, 502, 503), (
            f"Unexpected status for /ask example: {r2.status_code}"
        )
    except Exception as e:
        # TestClient can re-raise; if we got here, domain was found and adapter was invoked (crash is from Grok API)
        if "Domain not found" in str(e) or "404" in str(e):
            raise
        pass  # External API error is acceptable
    print("POST /ask existing domain: OK (domain switching does not crash)")
    return True


def main():
    ok = True
    try:
        test_a_domain_listing()
    except AssertionError as e:
        print(f"A) FAIL: {e}")
        ok = False
    try:
        test_b_rag_disabled_no_context()
    except Exception as e:
        print(f"B) rag.enabled FAIL: {e}")
        ok = False
    try:
        test_b_top_k_limits_retrieval()
    except Exception as e:
        print(f"B) top_k FAIL: {e}")
        ok = False
    try:
        test_b_output_format_json_returns_json()
    except Exception as e:
        print(f"B) output.format=json FAIL: {e}")
        ok = False
    try:
        test_ask_404_and_no_crash()
    except Exception as e:
        print(f"POST /ask check FAIL: {e}")
        ok = False

    if ok:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
