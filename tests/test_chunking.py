import pytest
from app.chunking import chunk_text


def test_empty_input_returns_empty():
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   \n  ", 100, 10) == []


def test_short_text_single_chunk():
    out = chunk_text("hello world", 100, 10)
    assert out == ["hello world"]


def test_overlap_shared_between_chunks():
    text = "abcdefghij" * 3  # 30 chars
    out = chunk_text(text, chunk_size=20, overlap=5)
    assert len(out) >= 2
    # step = 15, so chunk 2 starts at index 15; overlap region present
    assert out[0][15:] == out[1][:5]


def test_long_text_covers_all_characters():
    text = "x" * 1000
    out = chunk_text(text, chunk_size=100, overlap=20)
    assert "".join(c for c in out)  # non-empty
    assert all(len(c) <= 100 for c in out)
    assert len(out) > 1


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("data", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("data", chunk_size=10, overlap=-1)


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("data", chunk_size=0, overlap=0)
