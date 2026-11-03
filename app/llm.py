"""LLM generator abstraction: Claude (primary), OpenAI (optional), and a Fake.

Claude integration follows the official Anthropic Python SDK:
  - ``AsyncAnthropic()`` reads ANTHROPIC_API_KEY from the environment.
  - Non-streaming: ``await client.messages.create(...)``; the response ``content``
    is a list of blocks — join the ``.text`` of blocks whose ``.type == "text"``.
  - Streaming: ``async with client.messages.stream(...) as stream: async for text
    in stream.text_stream: yield text``.
  - Model defaults to ``claude-opus-5-5`` (no date suffix). Thinking is always on
    and ``output_config.effort`` controls how much; ``temperature``/``top_p``,
    ``budget_tokens`` and ``thinking.type="disabled"`` are NOT sent (they 400).
  - A safety-classifier decline is a normal response with ``stop_reason ==
    "refusal"``; it is surfaced as an error rather than an empty answer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

import anthropic
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import Settings
from app.vectorstore import SearchResult

SYSTEM_PROMPT = (
    "You are a precise retrieval-augmented assistant. Answer the user's question "
    "using ONLY the provided context passages. Cite the sources you use inline "
    "with the bracket form [source#chunk_index]. If the context does not contain "
    "the answer, say so plainly. Do not invent facts."
)


def _build_prompt(question: str, contexts: list[SearchResult]) -> str:
    if not contexts:
        blocks = "(no context retrieved)"
    else:
        blocks = "\n\n".join(
            f"[{c.source}#{c.chunk_index}] (score={c.score:.3f})\n{c.text}" for c in contexts
        )
    return f"Context passages:\n{blocks}\n\nQuestion: {question}\n\nAnswer with citations:"


@runtime_checkable
class Generator(Protocol):
    name: str

    async def generate(self, question: str, contexts: list[SearchResult]) -> str: ...

    def stream(self, question: str, contexts: list[SearchResult]) -> AsyncIterator[str]: ...


class ClaudeGenerator:
    """Anthropic Claude generator (default model: claude-opus-5-5)."""

    name = "claude"

    def __init__(self, settings: Settings) -> None:
        # AsyncAnthropic() resolves ANTHROPIC_API_KEY from the environment; pass it
        # explicitly so SecretStr config stays the single source of truth.
        key = settings.anthropic_api_key
        self._client = AsyncAnthropic(api_key=key.get_secret_value()) if key else AsyncAnthropic()
        self._model = settings.llm_model
        self._max_tokens = settings.max_tokens
        self._effort = settings.llm_effort

    async def generate(self, question: str, contexts: list[SearchResult]) -> str:
        prompt = _build_prompt(question, contexts)
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self._effort},
            )
        except (anthropic.APIError, anthropic.RateLimitError) as exc:  # surfaced to API layer
            raise RuntimeError(f"Claude generation failed: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise RuntimeError("Claude declined to answer (stop_reason=refusal)")
        # content is a list of blocks; concatenate the text-type blocks.
        return "".join(b.text for b in resp.content if b.type == "text")

    async def stream(  # type: ignore[override]
        self, question: str, contexts: list[SearchResult]
    ) -> AsyncIterator[str]:
        prompt = _build_prompt(question, contexts)
        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self._effort},
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except (anthropic.APIError, anthropic.RateLimitError) as exc:
            raise RuntimeError(f"Claude streaming failed: {exc}") from exc
        if final.stop_reason == "refusal":
            raise RuntimeError("Claude declined to answer (stop_reason=refusal)")


class OpenAIGenerator:
    """Optional OpenAI chat-completions generator."""

    name = "openai"

    def __init__(self, settings: Settings) -> None:
        if settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI generator")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._model = "gpt-4o-mini"
        self._max_tokens = settings.max_tokens

    async def generate(self, question: str, contexts: list[SearchResult]) -> str:
        prompt = _build_prompt(question, contexts)
        resp = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    async def stream(  # type: ignore[override]
        self, question: str, contexts: list[SearchResult]
    ) -> AsyncIterator[str]:
        prompt = _build_prompt(question, contexts)
        stream = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class FakeGenerator:
    """Deterministic generator for tests: echoes a fixed answer citing the sources."""

    name = "fake"

    async def generate(self, question: str, contexts: list[SearchResult]) -> str:
        cites = " ".join(f"[{c.source}#{c.chunk_index}]" for c in contexts)
        if not contexts:
            return "No relevant context was found to answer the question."
        return f"Based on the retrieved context, here is the answer. Sources: {cites}"

    async def stream(  # type: ignore[override]
        self, question: str, contexts: list[SearchResult]
    ) -> AsyncIterator[str]:
        full = await self.generate(question, contexts)
        for token in full.split(" "):
            yield token + " "
