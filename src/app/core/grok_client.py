"""xAI Grok API client (OpenAI-compatible)."""

from typing import Optional

from openai import OpenAI

from .config_loader import DomainConfig


class GrokClient:
    """Client for xAI Grok chat completions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.x.ai/v1",
    ):
        self._client = OpenAI(
            api_key=api_key or "",
            base_url=base_url,
        )

    def chat(
        self,
        domain: DomainConfig,
        user_message: str,
        system_override: Optional[str] = None,
    ) -> str:
        """
        Send a chat completion request to Grok.
        Uses domain's model and system prompt unless overridden.
        """
        system_content = system_override if system_override is not None else domain.system_prompt
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_message})

        response = self._client.chat.completions.create(
            model=domain.model,
            messages=messages,
            temperature=domain.temperature,
            max_tokens=domain.max_tokens,
        )
        choice = response.choices[0] if response.choices else None
        if not choice or not choice.message:
            return ""
        return choice.message.content or ""
