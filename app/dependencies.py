from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.database import get_session
from app.embeddings import Embedder, FakeEmbedder, LexicalEmbedder, RealEmbedder
from app.graph import AgenticRAG
from app.judge import ClaudeJudge, FakeJudge, Judge
from app.llm import ClaudeGenerator, FakeGenerator, Generator, OpenAIGenerator
from app.rag import RAGService
from app.vectorstore import PgVectorStore, VectorStore

SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_embedder(settings: SettingsDep) -> Embedder:
    if settings.embedding_provider == "fake":
        return FakeEmbedder(dim=settings.embed_dim)
    if settings.embedding_provider == "lexical":
        return LexicalEmbedder(dim=settings.embed_dim)
    return RealEmbedder(settings)


def get_generator(settings: SettingsDep) -> Generator:
    if settings.llm_provider == "fake":
        return FakeGenerator()
    if settings.llm_provider == "openai":
        return OpenAIGenerator(settings)
    return ClaudeGenerator(settings)


def get_judge(settings: SettingsDep) -> Judge:
    if settings.llm_provider == "fake":
        return FakeJudge()
    return ClaudeJudge(settings)


def get_vector_store(session: SessionDep) -> VectorStore:
    return PgVectorStore(session)


def get_rag_service(
    embedder: Annotated[Embedder, Depends(get_embedder)],
    generator: Annotated[Generator, Depends(get_generator)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    settings: SettingsDep,
) -> RAGService:
    return RAGService(embedder, generator, store, settings)


RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]


def get_agentic_rag(
    embedder: Annotated[Embedder, Depends(get_embedder)],
    generator: Annotated[Generator, Depends(get_generator)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    judge: Annotated[Judge, Depends(get_judge)],
    settings: SettingsDep,
) -> AgenticRAG:
    return AgenticRAG(embedder, generator, store, judge, settings)


AgenticRAGDep = Annotated[AgenticRAG, Depends(get_agentic_rag)]
