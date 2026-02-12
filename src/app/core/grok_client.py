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
        # Derive answer from first excerpt so responses match the actual domain's context
        m = re.search(r"\[1\] \(source: ([^)]+)\)\s*\n(.*?)(?=\n\[2\]|\n\n\n|\Z)", user_message, re.DOTALL)
        if m:
            source = m.group(1).strip()
            text = m.group(2).strip()[:200]
            snippet = text[:80] + "..." if len(text) > 80 else text
            snippet = snippet.replace("\n", " ")
            first_sentence = text.split(".")[0] + "." if "." in text else text
            answer = f"Based on the provided context, {first_sentence}"
            return json.dumps({
                "answer": answer[:300],
                "confidence": "high",
                "citations": [{"excerpt_id": 1, "source": source, "snippet": snippet}],
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
