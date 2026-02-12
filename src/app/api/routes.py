"""API routes: GET /domains, POST /ask."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import BadRequestError, AuthenticationError, APIError

from src.app.domains.registry import get_domain_registry

router = APIRouter()


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    domain_id: str = Field(..., description="Domain to ask in")
    question: str = Field(..., min_length=1, description="User question")


class AskResponse(BaseModel):
    """Response for POST /ask."""

    domain_id: str
    answer: str | dict  # Can be text or JSON dict based on output.format


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
    
    try:
        answer = adapter.ask(req.question)
    except BadRequestError as e:
        # Model not found, invalid parameters, etc.
        error_msg = str(e)
        if "Model not found" in error_msg:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model configuration: {error_msg}. Check your domain config's 'model' field."
            )
        raise HTTPException(status_code=400, detail=f"Invalid request to Grok API: {error_msg}")
    except AuthenticationError as e:
        raise HTTPException(
            status_code=401,
            detail="Authentication failed. Check your GROK_API_KEY environment variable."
        )
    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Grok API error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )
    
    return AskResponse(domain_id=req.domain_id, answer=answer)
