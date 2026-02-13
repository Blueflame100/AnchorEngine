"""Unit tests for hallucination mitigation / grounding."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure src is on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.core.grounding import (
    REFUSAL_MSG,
    SAFE_REFUSAL_RESPONSE,
    parse_and_validate_response,
    apply_grounding_checks,
)
from src.app.core.config_loader import DomainConfig, RAGConfig
from src.app.core.grok_client import GrokClient
from src.app.domains.adapter import DomainAdapter


class _MockRAGEngine:
    """Mock RAG that returns configurable excerpts."""

    def __init__(self, excerpts: list[dict] | None = None):
        self._excerpts = excerpts or []

    def retrieve_with_sources(self, domain, query):
        return self._excerpts


def test_refusal_when_no_context():
    """When no RAG context is retrieved, adapter returns refusal."""
    os.environ["USE_MOCK_LLM"] = "true"
    config = DomainConfig(
        domain_id="test",
        display_name="Test",
        data_dir=None,  # No data_dir → no context
        rag=RAGConfig(enabled=True),
    )
    mock_grok = GrokClient(use_mock=True)
    engine = _MockRAGEngine(excerpts=[])  # No context
    adapter = DomainAdapter(config=config, grok_client=mock_grok, rag_engine=engine)

    result = adapter.ask("What is the access key rotation policy?")

    assert isinstance(result, dict)
    assert result["answer"] == REFUSAL_MSG
    assert result["confidence"] == "low"
    assert result["citations"] == []


def test_clarification_when_ambiguous():
    """When question is ambiguous (e.g. 'how many minutes?'), ask for clarification."""
    os.environ["USE_MOCK_LLM"] = "true"
    config = DomainConfig(
        domain_id="test",
        display_name="Test",
        data_dir="data/iam",
        rag=RAGConfig(enabled=True),
    )
    mock_grok = GrokClient(use_mock=True)
    engine = _MockRAGEngine(excerpts=[{"source": "policy.txt", "text": "Lockout: 15 min.", "score": 0.9}])
    adapter = DomainAdapter(config=config, grok_client=mock_grok, rag_engine=engine)

    result = adapter.ask("how many minutes?")

    assert isinstance(result, dict)
    assert "clarify" in result["answer"].lower()
    assert result["confidence"] == "low"
    assert result["citations"] == []


def test_refusal_when_missing_citations():
    """When model returns answer without citations, grounding check replaces with refusal."""
    # Valid excerpt_ids 1, 2; model returns empty citations
    parsed = {
        "answer": "Some answer without citations",
        "confidence": "high",
        "citations": [],
    }
    result = apply_grounding_checks(parsed, num_excerpts=2)

    assert result["answer"] == REFUSAL_MSG
    assert result["confidence"] == "low"
    assert result["citations"] == []


def test_invalid_json_fallback():
    """When model output is invalid JSON, fall back to safe refusal."""
    result = parse_and_validate_response("not json at all", num_excerpts=3)
    assert result == SAFE_REFUSAL_RESPONSE

    result = parse_and_validate_response("", num_excerpts=3)
    assert result["answer"] == REFUSAL_MSG
    assert result["confidence"] == "low"

    result = parse_and_validate_response("{invalid}", num_excerpts=3)
    assert result["answer"] == REFUSAL_MSG


def test_valid_answer_with_citations():
    """When model returns valid answer with citations, pass through."""
    raw = '{"answer":"Access keys must be rotated every 90 days.","confidence":"high","citations":[{"excerpt_id":1,"source":"policy.txt","snippet":"Access keys must be rotated every 90 days."}]}'
    result = parse_and_validate_response(raw, num_excerpts=2)

    assert result["answer"] == "Access keys must be rotated every 90 days."
    assert result["confidence"] == "high"
    assert len(result["citations"]) == 1
    assert result["citations"][0]["excerpt_id"] == 1
    assert result["citations"][0]["source"] == "policy.txt"


def test_invalid_excerpt_id_replaced_with_refusal():
    """When citations reference nonexistent excerpt_id, replace with refusal."""
    parsed = {
        "answer": "Some answer",
        "confidence": "high",
        "citations": [{"excerpt_id": 99, "source": "x", "snippet": "y"}],
    }
    result = apply_grounding_checks(parsed, num_excerpts=2)

    assert result["answer"] == REFUSAL_MSG
    assert result["confidence"] == "low"
    assert result["citations"] == []


def test_mock_mode_deterministic():
    """Adapter with mock Grok returns deterministic grounded response when context present."""
    os.environ["USE_MOCK_LLM"] = "true"
    config = DomainConfig(
        domain_id="iam",
        display_name="IAM",
        data_dir="data/iam",
        rag=RAGConfig(enabled=True),
    )
    mock_grok = GrokClient(use_mock=True)
    engine = _MockRAGEngine(excerpts=[
        {"source": "policy.txt", "text": "Access keys must be rotated every 90 days.", "score": 0.9},
    ])
    adapter = DomainAdapter(config=config, grok_client=mock_grok, rag_engine=engine)

    result = adapter.ask("How often should access keys be rotated?")

    assert isinstance(result, dict)
    assert "answer" in result
    assert "confidence" in result
    assert "citations" in result
    assert result["confidence"] == "high"
    assert len(result["citations"]) >= 1
    assert result["citations"][0]["excerpt_id"] == 1
