"""Domain-specific adapters: combine config, RAG context, and Grok."""

import json
from typing import Any

from src.app.core import DomainConfig, GrokClient, RAGEngine


class DomainAdapter:
    """
    Adapter for a single domain: uses domain config, optional RAG context,
    and Grok client to answer questions.
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

    def ask(self, question: str, include_rag_context: bool = True) -> str | dict[str, Any]:
        """
        Answer a question in this domain.
        If include_rag_context is True and RAG is enabled, augments prompt with RAG context.
        Returns text or JSON based on output.format config.
        """
        # Get RAG context if enabled
        context = ""
        if include_rag_context and self.config.rag.enabled:
            context = self._rag.build_context(self.config, question)
        
        # Build user message from template
        user_message = self.config.user_prompt_template.format(
            question=question,
            context=context if context else "No relevant context available."
        )
        
        # Call Grok
        response_text = self._grok.chat(
            domain=self.config,
            user_message=user_message,
            system_override=self.config.system_prompt,
        )
        
        # Format output based on config
        if self.config.output.format == "json":
            try:
                # Try to parse as JSON
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If parsing fails, return as text with error note
                return {"error": "Failed to parse JSON response", "raw": response_text}
        else:
            return response_text
