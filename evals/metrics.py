"""Pure scoring functions for the eval harness — no I/O, no model calls."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.citations import cited_ids

REFUSAL = re.compile(
    r"no relevant context|does not contain|doesn't contain|not (?:mentioned|covered|provided)"
    r"|cannot (?:find|answer)|don't know|no information",
    re.IGNORECASE,
)


@dataclass
class CaseResult:
    id: str
    answerable: bool
    context_sources: list[str]
    cited: list[str]
    hit: bool | None = None
    reciprocal_rank: float | None = None
    context_precision: float | None = None
    citation_precision: float | None = None
    abstained: bool = False
    coverage: float | None = None
    unsupported: int = 0
    steps: list[str] = field(default_factory=list)


def abstained(answer: str, context_sources: list[str]) -> bool:
    return not context_sources or (not cited_ids(answer) and bool(REFUSAL.search(answer)))


def score_case(
    case: dict,
    answer: str,
    context_sources: list[str],
    *,
    unsupported: int = 0,
    steps: list[str] | None = None,
    score_answer: bool = False,
) -> CaseResult:
    expected = set(case["sources"])
    cited_sources = [c.split("#", 1)[0] for c in cited_ids(answer)]
    result = CaseResult(
        id=case["id"],
        answerable=bool(expected),
        context_sources=context_sources,
        cited=cited_sources,
        abstained=abstained(answer, context_sources),
        unsupported=unsupported,
        steps=steps or [],
    )
    if not expected:
        return result
    ranks = [i for i, s in enumerate(context_sources, start=1) if s in expected]
    result.hit = bool(ranks)
    result.reciprocal_rank = 1.0 / ranks[0] if ranks else 0.0
    if context_sources:
        result.context_precision = sum(s in expected for s in context_sources) / len(
            context_sources
        )
    if cited_sources:
        result.citation_precision = sum(s in expected for s in cited_sources) / len(cited_sources)
    if score_answer and case["must_include"]:
        found = [f for f in case["must_include"] if f.lower() in answer.lower()]
        result.coverage = len(found) / len(case["must_include"])
    return result


def _mean(values: list[float | bool | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(results: list[CaseResult]) -> dict[str, float | None]:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    return {
        "hit_rate": _mean([r.hit for r in answerable]),
        "mrr": _mean([r.reciprocal_rank for r in answerable]),
        "context_precision": _mean([r.context_precision for r in answerable]),
        "citation_precision": _mean([r.citation_precision for r in answerable]),
        "answer_coverage": _mean([r.coverage for r in answerable]),
        "abstention_rate": _mean([r.abstained for r in unanswerable]),
        "wrong_abstention_rate": _mean([r.abstained for r in answerable]),
        "unsupported_citations": float(sum(r.unsupported for r in results)),
    }
