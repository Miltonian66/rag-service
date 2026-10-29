"""Deterministic citation checks: the model writes citations, code verifies them.

Citations use the ``[source#chunk_index]`` form required by the system prompt. A
citation is *supported* only if it points at a passage that was actually given to
the generator; anything else is removed from the answer and reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.vectorstore import SearchResult

CITATION = re.compile(r"\[([^\[\]#]+)#(\d+)\]")


def cited_ids(answer: str) -> list[str]:
    """Return ``source#chunk`` ids in order of first appearance, without duplicates."""
    seen: dict[str, None] = {}
    for source, index in CITATION.findall(answer):
        seen.setdefault(f"{source.strip()}#{int(index)}", None)
    return list(seen)


@dataclass
class CitationCheck:
    answer: str
    supported: list[str]
    unsupported: list[str]


def verify_citations(answer: str, contexts: list[SearchResult]) -> CitationCheck:
    allowed = {f"{c.source}#{c.chunk_index}" for c in contexts}
    ids = cited_ids(answer)
    unsupported = [i for i in ids if i not in allowed]

    def _strip(match: re.Match[str]) -> str:
        cid = f"{match.group(1).strip()}#{int(match.group(2))}"
        return "" if cid in unsupported else match.group(0)

    cleaned = CITATION.sub(_strip, answer) if unsupported else answer
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return CitationCheck(
        answer=cleaned,
        supported=[i for i in ids if i in allowed],
        unsupported=unsupported,
    )
