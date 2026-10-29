"""Eval harness: ingest a fixed corpus, run a golden question set, score both pipelines.

    uv run python -m evals.run                      # offline: lexical embedder + fakes
    uv run python -m evals.run --live               # real providers from .env
    uv run python -m evals.run --check evals/thresholds.json   # CI regression gate

Offline mode needs no network and no keys. It measures retrieval and the agentic
control flow (grading, rewriting, abstention); answer wording is only scored in
--live mode, where a real model writes the answers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from app.config import Settings
from app.dependencies import get_embedder, get_generator, get_judge
from app.embeddings import LexicalEmbedder
from app.graph import AgenticRAG
from app.judge import FakeJudge
from app.llm import FakeGenerator
from app.rag import RAGService
from app.vectorstore import InMemoryVectorStore

from evals.metrics import score_case, summarize

DATA = Path(__file__).parent / "data"
PIPELINES = ("simple", "agentic")


def load_cases() -> list[dict]:
    lines = (DATA / "golden.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def run(live: bool) -> dict:
    if live:
        settings = Settings(chunk_size=400, chunk_overlap=80, top_k=3)
        embedder = get_embedder(settings)
        generator = get_generator(settings)
        judge = get_judge(settings)
    else:
        settings = Settings(chunk_size=400, chunk_overlap=80, top_k=3, embed_dim=1024)
        embedder, generator, judge = LexicalEmbedder(1024), FakeGenerator(), FakeJudge()

    store = InMemoryVectorStore()
    simple = RAGService(embedder, generator, store, settings)
    agentic = AgenticRAG(embedder, generator, store, judge, settings)
    for doc in sorted((DATA / "corpus").glob("*.md")):
        await simple.ingest(doc.stem, doc.read_text(encoding="utf-8"))

    cases = load_cases()
    report: dict = {"mode": "live" if live else "offline", "cases": len(cases), "pipelines": {}}
    for name in PIPELINES:
        results = []
        for case in cases:
            if name == "simple":
                out = await simple.query(case["question"])
                unsupported, steps = 0, []
            else:
                out = await agentic.query(case["question"])
                unsupported, steps = len(out.unsupported_citations), out.steps
            results.append(
                score_case(
                    case,
                    out.answer,
                    [c.source for c in out.citations],
                    unsupported=unsupported,
                    steps=steps,
                    score_answer=live,
                )
            )
        report["pipelines"][name] = {
            "summary": summarize(results),
            "results": [asdict(r) for r in results],
        }
    return report


def markdown(report: dict) -> str:
    metrics = list(report["pipelines"]["simple"]["summary"])
    rows = [f"| metric | {' | '.join(PIPELINES)} |", f"|---|{'---|' * len(PIPELINES)}"]
    for m in metrics:
        vals = [report["pipelines"][p]["summary"][m] for p in PIPELINES]
        rows.append(f"| {m} | {' | '.join('—' if v is None else f'{v:g}' for v in vals)} |")
    return f"Eval ({report['mode']}, {report['cases']} cases)\n\n" + "\n".join(rows)


def check(report: dict, thresholds: dict) -> list[str]:
    """Return a list of failures. Keys: 'pipeline.metric' -> {'min': x} or {'max': x}."""
    failures = []
    for key, bound in thresholds.items():
        pipeline, metric = key.split(".", 1)
        value = report["pipelines"][pipeline]["summary"][metric]
        if value is None:
            failures.append(f"{key}: no value")
        elif "min" in bound and value < bound["min"]:
            failures.append(f"{key}: {value} < min {bound['min']}")
        elif "max" in bound and value > bound["max"]:
            failures.append(f"{key}: {value} > max {bound['max']}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="use real providers from .env")
    parser.add_argument("--out", type=Path, help="write the full JSON report here")
    parser.add_argument("--check", type=Path, help="thresholds JSON; exit 1 on regression")
    args = parser.parse_args()

    report = asyncio.run(run(args.live))
    print(markdown(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.check:
        failures = check(report, json.loads(args.check.read_text(encoding="utf-8")))
        for f in failures:
            print(f"REGRESSION {f}", file=sys.stderr)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
