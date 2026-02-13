"""Domain-specific adapters: combine config, RAG context, and Grok."""

import json
from typing import Any, Optional

from src.app.core import DomainConfig, GrokClient, RAGEngine
from src.app.core.grounding import (
    build_grounding_prompts,
    parse_and_validate_response,
    _build_numbered_context,
    is_ambiguous_question,
    CLARIFICATION_RESPONSE,
    REFUSAL_MSG,
    SAFE_REFUSAL_RESPONSE,
)


class DomainAdapter:
    """
    Adapter for a single domain: uses domain config, optional RAG context,
    and Grok client to answer questions.
    When RAG is enabled, applies hallucination mitigation (grounding, citations).
    """

    def __init__(
        self,
        config: DomainConfig,
        grok_client: GrokClient,
        rag_engine: RAGEngine,
    ):
        self.config = config
        self._grok = grok_client
        self._rag = rag_engine

    def ask(
        self,
        question: str,
        include_rag_context: bool = True,
        model_override: Optional[str] = None,
    ) -> str | dict[str, Any]:
        """
        Answer a question in this domain.
        When RAG is enabled: uses numbered excerpts, strict grounding, JSON output,
        and applies hallucination checks. Returns {answer, confidence, citations}.
        When RAG is disabled: returns text or JSON per output.format.
        """
        if include_rag_context and self.config.rag.enabled:
            return self._ask_with_grounding(question, model_override=model_override)
        return self._ask_without_rag(question, model_override=model_override)

    def _ask_with_grounding(
        self, question: str, model_override: Optional[str] = None
    ) -> dict[str, Any]:
        """RAG path: numbered excerpts, grounding prompts, JSON, validation."""
        if is_ambiguous_question(question):
            return CLARIFICATION_RESPONSE.copy()
        excerpts = self._rag.retrieve_with_sources(self.config, question)
        num_excerpts = len(excerpts)

        if num_excerpts == 0:
            return SAFE_REFUSAL_RESPONSE.copy()

        numbered_context = _build_numbered_context(excerpts)
        system_prompt, user_message = build_grounding_prompts(question, numbered_context)

        response_text = self._grok.chat(
            domain=self.config,
            user_message=user_message,
            system_override=system_prompt,
            model_override=model_override,
        )

        return parse_and_validate_response(response_text, num_excerpts)

    def _ask_without_rag(
        self, question: str, model_override: Optional[str] = None
    ) -> str | dict[str, Any]:
        """Non-RAG path: legacy behavior (template-based, text or json)."""
        context = "No relevant context available."
        user_message = self.config.user_prompt_template.format(
            question=question,
            context=context,
        )

        response_text = self._grok.chat(
            domain=self.config,
            user_message=user_message,
            system_override=self.config.system_prompt,
            model_override=model_override,
        )

        if self.config.output.format == "json":
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                return {"error": "Failed to parse JSON response", "raw": response_text}
        return response_text
