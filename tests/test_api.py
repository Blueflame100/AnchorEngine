"""API tests: GET /domains, POST /ask. No external API calls (USE_MOCK_LLM=true)."""

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_domains_endpoint():
    """GET /domains returns 200 and each item has domain_id and display_name."""
    resp = client.get("/domains")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    for item in data:
        assert "domain_id" in item
        assert "display_name" in item


def test_ask_returns_schema():
    """POST /ask with answerable question returns 200 and answer/confidence/citations."""
    resp = client.post(
        "/ask",
        json={"domain_id": "iam", "question": "How often should access keys be rotated?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "domain_id" in data
    assert data["domain_id"] == "iam"
    assert "answer" in data

    ans = data["answer"]
    assert isinstance(ans, dict)
    assert "answer" in ans
    assert "confidence" in ans
    assert "citations" in ans
    assert len(ans["citations"]) > 0


def test_refusal_when_unanswerable():
    """POST /ask with out-of-domain question returns refusal and empty citations."""
    resp = client.post(
        "/ask",
        json={"domain_id": "iam", "question": "What is the meaning of life?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    ans = data["answer"]
    assert isinstance(ans, dict)
    answer_text = ans.get("answer", "").lower().replace("\u2019", "'")
    assert "don't know" in answer_text or "do not know" in answer_text
    assert ans.get("citations", []) == []
