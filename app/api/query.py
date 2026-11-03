from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.dependencies import AgenticRAGDep, RAGServiceDep
from app.schemas import AgenticQueryOut, QueryIn, QueryOut

router = APIRouter(tags=["query"])


@router.post("/query", response_model=None)
async def run_query(
    payload: QueryIn, rag: RAGServiceDep, agentic: AgenticRAGDep
) -> QueryOut | AgenticQueryOut | EventSourceResponse:
    if payload.mode == "agentic":
        return await agentic.query(payload.question, payload.top_k)
    if payload.stream:
        return EventSourceResponse(rag.query_stream(payload.question, payload.top_k))
    return await rag.query(payload.question, payload.top_k)
