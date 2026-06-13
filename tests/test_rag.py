from app.config import Settings
from app.embeddings import FakeEmbedder
from app.llm import FakeGenerator
from app.rag import RAGService
from app.vectorstore import InMemoryVectorStore


def _service() -> RAGService:
    settings = Settings(embed_dim=64, chunk_size=50, chunk_overlap=10, top_k=3)
    return RAGService(FakeEmbedder(64), FakeGenerator(), InMemoryVectorStore(), settings)


async def test_ingest_returns_chunk_count():
    svc = _service()
    count = await svc.ingest("doc1", "word " * 100)
    assert count > 0


async def test_ingest_empty_text_yields_zero():
    svc = _service()
    assert await svc.ingest("doc1", "   ") == 0


async def test_query_returns_answer_with_citations():
    svc = _service()
    await svc.ingest("guide", "Paris is the capital of France. " * 10)
    result = await svc.query("What is the capital of France?")
    assert result.answer
    assert len(result.citations) > 0
    assert result.citations[0].source == "guide"


async def test_query_no_documents_returns_no_context_answer():
    svc = _service()
    result = await svc.query("anything")
    assert result.citations == []
    assert "No relevant context" in result.answer
