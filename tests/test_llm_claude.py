"""Unit test for ClaudeGenerator — verifies the service uses the Anthropic SDK
exactly as the API contract requires, with no network call (AsyncAnthropic mocked):
multi text-block assembly, and that none of the parameters that 400 on Opus 4.8
(temperature / top_p / top_k / budget_tokens / thinking) are ever sent.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import Settings
from app.llm import ClaudeGenerator
from app.vectorstore import SearchResult


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


async def test_claude_generate_joins_text_blocks_and_omits_forbidden_params() -> None:
    settings = Settings(llm_provider="claude", anthropic_api_key="sk-ant-test", max_tokens=512)
    # content mixes text blocks with a non-text block, which must be skipped.
    fake_response = SimpleNamespace(
        content=[_text("Hello "), SimpleNamespace(type="thinking", text="dropme"), _text("world")]
    )
    create = AsyncMock(return_value=fake_response)
    mock_client = MagicMock(messages=MagicMock(create=create))

    with patch("app.llm.AsyncAnthropic", return_value=mock_client):
        generator = ClaudeGenerator(settings)
        contexts = [SearchResult(source="doc", chunk_index=0, text="ctx", score=0.9)]
        answer = await generator.generate("question?", contexts)

    # Only text-type blocks are concatenated.
    assert answer == "Hello world"

    # Request matches the documented Claude contract.
    create.assert_awaited_once()
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "claude-opus-4-8"
    assert kwargs["max_tokens"] == 512
    assert kwargs["system"]
    assert kwargs["messages"][0]["role"] == "user"

    # None of the parameters that return 400 on Opus 4.8 are sent.
    for forbidden in ("temperature", "top_p", "top_k", "budget_tokens", "thinking"):
        assert forbidden not in kwargs
