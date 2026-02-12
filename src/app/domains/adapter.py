"""Domain-specific adapters: combine config, RAG context, and Grok."""

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

    def ask(self, question: str, include_rag_context: bool = True) -> str:
        """
        Answer a question in this domain.
        If include_rag_context is True and domain has documents, augments prompt with RAG context.
        """
        user_message = question
        system_override = None

        if include_rag_context:
            context = self._rag.build_context(self.config, question)
            if context:
                system_override = (
                    self.config.system_prompt.strip()
                    + "\n\n## Relevant context\n\n"
                    + context
                )

        return self._grok.chat(
            domain=self.config,
            user_message=user_message,
            system_override=system_override,
        )
