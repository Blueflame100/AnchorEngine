"""API routes: GET /domains, POST /ask."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.app.domains.registry import get_domain_registry

router = APIRouter()


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    domain_id: str = Field(..., description="Domain to ask in")
    question: str = Field(..., min_length=1, description="User question")


class AskResponse(BaseModel):
    """Response for POST /ask."""

    domain_id: str
    answer: str


@router.get("/domains")
def list_domains() -> list[dict]:
    """List available domains from configs."""
    registry = get_domain_registry()
    return registry.list_domains()


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Ask a question in a given domain (with optional RAG context)."""
    registry = get_domain_registry()
    adapter = registry.get_adapter(req.domain_id)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Domain not found: {req.domain_id}")
    answer = adapter.ask(req.question)
    return AskResponse(domain_id=req.domain_id, answer=answer)
