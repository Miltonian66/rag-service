"""Embedding abstraction with a real OpenAI implementation and a deterministic fake.

Anthropic does not ship a first-class embeddings API, so embeddings are produced
by OpenAI's ``text-embedding-3-small`` (1536 dims). The :class:`FakeEmbedder`
derives a deterministic vector from a text hash so retrieval logic can be tested
offline, with no network and no API key — at the same dimensionality.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

from openai import AsyncOpenAI

from app.config import Settings


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class RealEmbedder:
    """OpenAI-backed embedder (text-embedding-3-small, 1536 dims)."""

    name = "openai"

    def __init__(self, settings: Settings) -> None:
        if settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI embedder")
        # AsyncOpenAI reads OPENAI_API_KEY from env by default; pass explicitly so
        # SecretStr-managed config is the single source of truth.
        self._client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._model = settings.embed_model
        self.dim = settings.embed_dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]


class FakeEmbedder:
    """Deterministic, network-free embedder for tests and offline runs.

    Maps each text to a fixed-dimension unit vector derived from its SHA-256
    digest. Identical text -> identical vector; similar text shares no special
    structure, but the mapping is stable, which is all retrieval tests need.
    """

    name = "fake"

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        # Expand a SHA-256 digest into `dim` floats by hashing (text + counter).
        values: list[float] = []
        counter = 0
        while len(values) < self.dim:
            digest = hashlib.sha256(f"{text}:{counter}".encode()).digest()
            for byte in digest:
                values.append((byte / 255.0) * 2.0 - 1.0)  # map 0..255 -> -1..1
                if len(values) == self.dim:
                    break
            counter += 1
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
