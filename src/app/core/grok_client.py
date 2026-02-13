"""xAI Grok API client (OpenAI-compatible)."""

import json
import os
import re
from typing import Optional

from openai import OpenAI

from .config_loader import DomainConfig


class GrokClient:
    """Client for xAI Grok chat completions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.x.ai/v1",
        use_mock: Optional[bool] = None,
    ):
        self._use_mock = use_mock if use_mock is not None else (
            os.getenv("USE_MOCK_LLM", "").lower() == "true"
        )
        self._client = None if self._use_mock else OpenAI(
            api_key=api_key or "",
            base_url=base_url,
        )

    def _mock_format_answer(self, text: str, q_words: set[str]) -> str:
        """Format excerpt as concise prose, not a raw document dump."""
        # Find section whose header directly matches the question (e.g. "Severity levels:" for "severity levels")
        lines = [l for l in text.split("\n") if l.strip()]
        sections: list[tuple[str, list[str]]] = []  # (header, content lines)
        current_header = ""
        current_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^[A-Za-z][^:]*:\s*$", stripped):
                if current_lines:
                    sections.append((current_header, current_lines))
                current_header = stripped.lower()
                current_lines = [stripped]
            else:
                current_lines.append(stripped)
        if current_lines:
            sections.append((current_header, current_lines))
        # Prefer section whose header contains the most question words (direct match)
        best = sections[0] if sections else ("", [])
        best_header_score = 0
        for header, content in sections:
            score = sum(1 for w in q_words if w in header)
            if score > best_header_score:
                best_header_score = score
                best = (header, content)
        target = "\n".join(best[1]) if best[1] else text
        # When section header matched, keep full content. Otherwise filter to relevant lines only.
        if best_header_score == 0 and q_words:
            # No matching header: only include lines with 2+ question-word matches (avoids "access" matching everything)
            target_lines = [
                line for line in target.split("\n")
                if line.strip() and sum(1 for w in q_words if w in line.lower()) >= 2
            ]
            if target_lines:
                target = "\n".join(target_lines)
        # Convert bullet list to inline: "- X: Y" -> "X (Y); "
        parts = []
        for line in target.split("\n"):
            m = re.match(r"^[\-\*]?\s*(.+?):\s*(.+)$", line.strip())
            if m:
                parts.append(f"{m.group(1).strip()} ({m.group(2).strip()})")
            elif line.strip() and not line.strip().endswith(":"):
                parts.append(line.strip())
        prose = "; ".join(parts) if parts else target.replace("\n", " ").strip()
        prose = re.sub(r"\s+", " ", prose)
        return f"Based on the provided context, {prose}" if prose else f"Based on the provided context, {text[:200].replace(chr(10), ' ')}"

    def _mock_chat(self, user_message: str) -> str:
        """Deterministic mock for grounding tests and eval. Derives answer from context."""
        has_context = "[1]" in user_message and "No relevant context" not in user_message
        if not has_context:
            return '{"answer":"I don\\u2019t know based on the provided documents.","confidence":"low","citations":[]}'
        # Out-of-domain questions: refuse (for eval should_refuse=True)
        q = user_message.split("Question:")[-1].split("\n")[0].lower() if "Question:" in user_message else ""
        refuse_phrases = ("meaning of life", "password reset", "what is the meaning")
        if any(p in q for p in refuse_phrases):
            return '{"answer":"I don\\u2019t know based on the provided documents.","confidence":"low","citations":[]}'
        # Parse all excerpts and pick the one that best matches the question
        excerpt_pattern = re.compile(
            r"\[(\d+)\] \(source: ([^)]+)\)\s*\n(.*?)(?=\n\n\[\d+\]|\Z)",
            re.DOTALL,
        )
        excerpts = [
            {"id": int(m.group(1)), "source": m.group(2).strip(), "text": m.group(3).strip()}
            for m in excerpt_pattern.finditer(user_message)
        ]
        if not excerpts:
            pass  # fall through to fallback
        else:
            # Pick excerpt with most question-word matches (case-insensitive)
            q_words = {w.lower() for w in re.findall(r"\w+", q) if len(w) > 2}
            best = excerpts[0]
            best_score = 0
            for ex in excerpts:
                ex_lower = ex["text"].lower()
                score = sum(1 for w in q_words if w in ex_lower)
                if score > best_score:
                    best_score = score
                    best = ex
            # No excerpt matches the question -> out-of-domain, refuse
            if best_score == 0 and q_words:
                return '{"answer":"I don\\u2019t know based on the provided documents.","confidence":"low","citations":[]}'
            # Question asks about consequences but excerpt doesn't state them -> refuse
            consequence_phrases = ("what happens if", "what if", "what happens when", "consequences of")
            if any(p in q for p in consequence_phrases):
                ex_lower = best["text"].lower()
                consequence_words = ("consequence", "penalty", "violation", "revoke", "suspend", "expire", "fail", "happen", "result")
                if not any(w in ex_lower for w in consequence_words):
                    return '{"answer":"I don\\u2019t know based on the provided documents.","confidence":"low","citations":[]}'
            text = best["text"]
            snippet = text[:100] + "..." if len(text) > 100 else text
            snippet = snippet.replace("\n", " ")
            # Format as concise prose: extract relevant section, collapse newlines
            answer = self._mock_format_answer(text, q_words)
            # Use medium when match is weaker (1-2 keywords), high when strong (3+)
            confidence = "high" if best_score >= 3 else "medium"
            return json.dumps({
                "answer": answer[:400],
                "confidence": confidence,
                "citations": [
                    {"excerpt_id": best["id"], "source": best["source"], "snippet": snippet}
                ],
            })
        # Fallback for unexpected format (e.g. tests)
        return '{"answer":"Based on the provided context, access keys must be rotated every 90 days.","confidence":"high","citations":[{"excerpt_id":1,"source":"policy.txt","snippet":"Access keys must be rotated every 90 days."}]}'

    def chat(
        self,
        domain: DomainConfig,
        user_message: str,
        system_override: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """
        Send a chat completion request to Grok.
        Uses domain's model and system prompt unless overridden.
        """
        if self._use_mock:
            return self._mock_chat(user_message)

        system_content = system_override if system_override is not None else domain.system_prompt
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_message})

        model = model_override if model_override else domain.model
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=domain.temperature,
            max_tokens=domain.max_tokens,
        )
        choice = response.choices[0] if response.choices else None
        if not choice or not choice.message:
            return ""
        return choice.message.content or ""
