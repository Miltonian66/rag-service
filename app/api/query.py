from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.dependencies import RAGServiceDep
from app.schemas import QueryIn, QueryOut

router = APIRouter(tags=["query"])


@router.post("/query", response_model=None)
async def run_query(payload: QueryIn, rag: RAGServiceDep) -> QueryOut | EventSourceResponse:
    if payload.stream:
        return EventSourceResponse(rag.query_stream(payload.question, payload.top_k))
    return await rag.query(payload.question, payload.top_k)
