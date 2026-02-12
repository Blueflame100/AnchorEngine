"""Hallucination mitigation: grounding checks and safe refusal for RAG responses."""

import json
from typing import Any

REFUSAL_MSG = "I don't know based on the provided documents."

SAFE_REFUSAL_RESPONSE = {
    "answer": REFUSAL_MSG,
    "confidence": "low",
    "citations": [],
}


def _build_numbered_context(excerpts: list[dict]) -> str:
    """Build context as [1], [2], ... numbered excerpts."""
    lines = []
    for i, ex in enumerate(excerpts, 1):
        source = ex.get("source", "unknown")
        text = ex.get("text", "")
        lines.append(f"[{i}] (source: {source})\n{text}")
    return "\n\n".join(lines)


def build_grounding_prompts(question: str, numbered_context: str) -> tuple[str, str]:
    """
    Return (system_prompt, user_message) for strict grounding with citations.
    """
    system = """You are a grounding assistant. You MUST only answer using the provided excerpts.
Rules:
- If the answer cannot be directly supported by the excerpts, respond with: "I don't know based on the provided documents."
- Every factual statement must have at least one citation.
- Citations must use excerpt_id (the number in brackets [1], [2], etc.).
- If you cannot cite, you must refuse.
- Output ONLY valid JSON, no markdown, no extra text."""

    user = f"""Question: {question}

Numbered excerpts (cite by excerpt_id):
{numbered_context}

Respond with exactly this JSON structure:
{{"answer": "<string>", "confidence": "high"|"medium"|"low", "citations": [{{"excerpt_id": <int>, "source": "<string>", "snippet": "<short quote>"}}]}}"""

    return system, user


def apply_grounding_checks(parsed: dict, num_excerpts: int) -> dict:
    """
    Apply hallucination checks. Returns safe response.
    - Invalid excerpt_id → refusal
    - Empty citations → refusal
    - Invalid confidence → coerce to low
    """
    result = {
        "answer": parsed.get("answer", REFUSAL_MSG),
        "confidence": parsed.get("confidence", "low"),
        "citations": parsed.get("citations", []),
    }

    # Validate confidence
    if result["confidence"] not in ("high", "medium", "low"):
        result["confidence"] = "low"

    # Validate citations
    citations = result["citations"]
    if not isinstance(citations, list):
        citations = []

    valid_ids = set(range(1, num_excerpts + 1)) if num_excerpts > 0 else set()

    for c in citations:
        if not isinstance(c, dict):
            continue
        eid = c.get("excerpt_id")
        if eid is None:
            continue
        try:
            eid = int(eid)
        except (TypeError, ValueError):
            eid = -1
        if num_excerpts > 0 and eid not in valid_ids:
            result["answer"] = REFUSAL_MSG
            result["confidence"] = "low"
            result["citations"] = []
            return result

    # Empty citations → refusal
    if num_excerpts > 0 and not citations:
        result["answer"] = REFUSAL_MSG
        result["confidence"] = "low"
        result["citations"] = []

    return result


def parse_and_validate_response(
    raw: str,
    num_excerpts: int,
) -> dict[str, Any]:
    """
    Parse JSON from model output, apply grounding checks, return safe response.
    """
    try:
        # Strip markdown code blocks if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return SAFE_REFUSAL_RESPONSE.copy()

    if not isinstance(parsed, dict):
        return SAFE_REFUSAL_RESPONSE.copy()

    return apply_grounding_checks(parsed, num_excerpts)
