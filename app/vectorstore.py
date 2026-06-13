"""Vector store abstraction: real pgvector store + in-memory cosine store.

PgVectorStore uses the pgvector ``<=>`` cosine-distance operator over an async
SQLAlchemy session. InMemoryVectorStore replicates the same cosine ranking in
pure Python/NumPy so retrieval is testable without Postgres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk


@dataclass
class ChunkRecord:
    source: str
    chunk_index: int
    text: str
    embedding: list[float]


@dataclass
class SearchResult:
    source: str
    chunk_index: int
    text: str
    score: float


@runtime_checkable
class VectorStore(Protocol):
    async def add(self, chunks: list[ChunkRecord]) -> int: ...

    async def search(self, embedding: list[float], top_k: int) -> list[SearchResult]: ...


class PgVectorStore:
    """Async SQLAlchemy + pgvector implementation."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, chunks: list[ChunkRecord]) -> int:
        rows = [
            Chunk(
                source=c.source,
                chunk_index=c.chunk_index,
                text=c.text,
                embedding=c.embedding,
            )
            for c in chunks
        ]
        self._session.add_all(rows)
        await self._session.commit()
        return len(rows)

    async def search(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        # cosine distance via <=>; score = 1 - distance (higher is more similar).
        distance = Chunk.embedding.cosine_distance(embedding).label("distance")
        stmt = select(Chunk, distance).order_by(distance).limit(top_k)
        result = await self._session.execute(stmt)
        out: list[SearchResult] = []
        for chunk, dist in result.all():
            out.append(
                SearchResult(
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=1.0 - float(dist),
                )
            )
        return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Pure-Python cosine store for tests/offline runs."""

    def __init__(self) -> None:
        self._records: list[ChunkRecord] = []

    async def add(self, chunks: list[ChunkRecord]) -> int:
        self._records.extend(chunks)
        return len(chunks)

    async def search(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        scored = [
            SearchResult(
                source=r.source,
                chunk_index=r.chunk_index,
                text=r.text,
                score=_cosine(embedding, r.embedding),
            )
            for r in self._records
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
