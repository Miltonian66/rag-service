"""Embedding abstraction with a real OpenAI implementation and a deterministic fake.

Anthropic does not ship a first-class embeddings API, so embeddings are produced
by OpenAI's ``text-embedding-3-small`` (1536 dims). The :class:`FakeEmbedder`
derives a deterministic vector from a text hash so retrieval logic can be tested
offline, with no network and no API key — at the same dimensionality.
"""

from __future__ import annotations

import hashlib
import math
import re
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


_TOKEN = re.compile(r"[a-z0-9]+")


class LexicalEmbedder:
    """Deterministic bag-of-words embedder (feature hashing) for offline evals.

    Unlike :class:`FakeEmbedder`, texts that share words get similar vectors, so
    retrieval quality can be measured offline. Each lowercase token (plus each
    adjacent token pair) is hashed into one of ``dim`` buckets with a signed
    weight; the vector is L2-normalised.
    """

    name = "lexical"

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN.findall(text.lower())
        return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.dim
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            values[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
