"""Character-based text chunking with overlap.

Pure, dependency-free, and trivially testable. The chunker slides a fixed-size
window over the input with a configurable overlap so adjacent chunks share
context at their boundaries (important for retrieval recall).
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping character windows.

    Args:
        text: The source text. Empty or whitespace-only input yields ``[]``.
        chunk_size: Maximum characters per chunk. Must be > 0.
        overlap: Characters shared between consecutive chunks. Must satisfy
            ``0 <= overlap < chunk_size``.

    Returns:
        A list of non-empty chunk strings in document order.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    stripped = text.strip()
    if not stripped:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    n = len(stripped)

    while start < n:
        end = start + chunk_size
        piece = stripped[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start += step

    return chunks
