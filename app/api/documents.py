from fastapi import APIRouter, status

from app.dependencies import RAGServiceDep
from app.schemas import DocumentIn, DocumentOut

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def ingest_document(payload: DocumentIn, rag: RAGServiceDep) -> DocumentOut:
    count = await rag.ingest(payload.source, payload.text)
    return DocumentOut(source=payload.source, chunks_ingested=count)
