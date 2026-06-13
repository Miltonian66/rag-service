"""RAG orchestrator: ingest pipeline and query pipeline."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from app.chunking import chunk_text
from app.config import Settings
from app.embeddings import Embedder
from app.llm import Generator
from app.schemas import Citation, QueryOut
from app.vectorstore import ChunkRecord, VectorStore


class RAGService:
    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
        store: VectorStore,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._generator = generator
        self._store = store
        self._settings = settings

    async def ingest(self, source: str, text: str) -> int:
        chunks = chunk_text(text, self._settings.chunk_size, self._settings.chunk_overlap)
        if not chunks:
            return 0
        embeddings = await self._embedder.embed(chunks)
        records = [
            ChunkRecord(source=source, chunk_index=i, text=chunk, embedding=emb)
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        return await self._store.add(records)

    async def _retrieve(self, question: str, top_k: int | None):
        k = top_k or self._settings.top_k
        query_vec = await self._embedder.embed_query(question)
        return await self._store.search(query_vec, k)

    async def query(self, question: str, top_k: int | None = None) -> QueryOut:
        contexts = await self._retrieve(question, top_k)
        answer = await self._generator.generate(question, contexts)
        citations = [
            Citation(source=c.source, chunk_index=c.chunk_index, score=c.score, text=c.text)
            for c in contexts
        ]
        return QueryOut(answer=answer, citations=citations)

    async def query_stream(
        self, question: str, top_k: int | None = None
    ) -> AsyncIterator[dict[str, str]]:
        """Yield SSE-shaped events: one ``citations`` event, then ``token`` events."""
        contexts = await self._retrieve(question, top_k)
        citations = [
            Citation(source=c.source, chunk_index=c.chunk_index, score=c.score, text=c.text)
            for c in contexts
        ]
        yield {"event": "citations", "data": json.dumps([c.model_dump() for c in citations])}
        async for token in self._generator.stream(question, contexts):
            yield {"event": "token", "data": token}
        yield {"event": "done", "data": "[DONE]"}
