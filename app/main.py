from fastapi import FastAPI

from app.api import documents, health, query

app = FastAPI(
    title="RAG Service",
    version="0.1.0",
    summary="Ingest documents, retrieve with pgvector, answer with cited sources via Claude.",
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
