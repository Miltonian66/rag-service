import pytest
from app.config import Settings
from app.dependencies import (
    get_embedder,
    get_generator,
    get_settings,
    get_vector_store,
)
from app.embeddings import FakeEmbedder
from app.llm import FakeGenerator
from app.main import app
from app.vectorstore import InMemoryVectorStore
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://test:test@localhost/test",
        embedding_provider="fake",
        llm_provider="fake",
        embed_dim=64,
        chunk_size=50,
        chunk_overlap=10,
        top_k=3,
    )


@pytest.fixture
def shared_store() -> InMemoryVectorStore:
    # One store instance shared across the whole request lifecycle of a test,
    # so ingest writes are visible to subsequent query calls.
    return InMemoryVectorStore()


@pytest.fixture
def client(test_settings, shared_store) -> AsyncClient:
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder(dim=test_settings.embed_dim)
    app.dependency_overrides[get_generator] = lambda: FakeGenerator()
    app.dependency_overrides[get_vector_store] = lambda: shared_store

    transport = ASGITransport(app=app)
    yield AsyncClient(transport=transport, base_url="http://test")

    app.dependency_overrides.clear()
