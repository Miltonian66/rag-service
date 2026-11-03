"""ClaudeJudge request contract and fail-soft behaviour (AsyncAnthropic mocked), plus FakeJudge."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
from app.config import Settings
from app.judge import ClaudeJudge, FakeJudge


def _response(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)]
    )


def _judge(create: AsyncMock) -> ClaudeJudge:
    settings = Settings(llm_provider="claude", anthropic_api_key="sk-ant-test")
    with patch(
        "app.judge.AsyncAnthropic", return_value=MagicMock(messages=MagicMock(create=create))
    ):
        return ClaudeJudge(settings)


async def test_grade_uses_structured_output_and_no_sampling_params() -> None:
    create = AsyncMock(return_value=_response('{"relevant": false}'))
    assert await _judge(create).grade("q?", "passage") is False

    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["output_config"]["effort"] == "low"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["output_config"]["format"]["schema"]["required"] == ["relevant"]
    # The passage is wrapped as data, not mixed into the instructions.
    assert "<passage>\npassage\n</passage>" in kwargs["messages"][0]["content"]
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in kwargs


async def test_grade_fails_soft_to_relevant_on_api_error() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    create = AsyncMock(side_effect=anthropic.APIConnectionError(request=request))
    assert await _judge(create).grade("q?", "passage") is True


async def test_grade_fails_soft_on_refusal_and_bad_json() -> None:
    refusal = AsyncMock(return_value=_response("", stop_reason="refusal"))
    assert await _judge(refusal).grade("q?", "p") is True
    garbage = AsyncMock(return_value=_response("not json"))
    assert await _judge(garbage).grade("q?", "p") is True


async def test_rewrite_returns_query_or_falls_back_to_question() -> None:
    ok = AsyncMock(return_value=_response('{"query": "fleet api request limit pro plan"}'))
    assert await _judge(ok).rewrite("limit on pro?") == "fleet api request limit pro plan"
    empty = AsyncMock(return_value=_response('{"query": "  "}'))
    assert await _judge(empty).rewrite("limit on pro?") == "limit on pro?"


async def test_fake_judge_grades_by_overlap_and_strips_stopwords() -> None:
    judge = FakeJudge(min_overlap=0.5)
    assert await judge.grade("refund window days", "Refunds within 30 days, window applies")
    assert not await judge.grade("refund window days", "Express delivery costs 49 euros")
    assert await judge.rewrite("What is the refund window?") == "refund window"
