import pytest
from app.tracing import langfuse_callbacks


def test_tracing_is_noop_without_keys(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert langfuse_callbacks() == []


def test_tracing_returns_handler_with_keys(monkeypatch) -> None:
    langfuse_langchain = pytest.importorskip("langfuse.langchain")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")
    callbacks = langfuse_callbacks()
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], langfuse_langchain.CallbackHandler)
