"""AgenticRAG control flow: grading, rewrite budget, refusal and citation verification."""

from __future__ import annotations

from app.config import Settings
from app.embeddings import LexicalEmbedder
from app.graph import AgenticRAG
from app.judge import FakeJudge
from app.llm import FakeGenerator
from app.vectorstore import InMemoryVectorStore, SearchResult


class ScriptedJudge(FakeJudge):
    """Grades nothing relevant until the query was rewritten."""

    def __init__(self) -> None:
        super().__init__()
        self.rewrites = 0

    async def grade(self, question: str, passage: str) -> bool:
        return self.rewrites > 0 and await super().grade(question, passage)

    async def rewrite(self, question: str) -> str:
        self.rewrites += 1
        return await super().rewrite(question)


class LyingGenerator(FakeGenerator):
    async def generate(self, question: str, contexts: list[SearchResult]) -> str:
        return await super().generate(question, contexts) + " See also [invented#9]."


async def _agent(judge=None, generator=None, max_rewrites: int = 1) -> AgenticRAG:
    settings = Settings(chunk_size=200, chunk_overlap=20, top_k=2, max_rewrites=max_rewrites)
    store = InMemoryVectorStore()
    agent = AgenticRAG(
        LexicalEmbedder(256), generator or FakeGenerator(), store, judge or FakeJudge(), settings
    )
    from app.rag import RAGService

    ingest = RAGService(LexicalEmbedder(256), FakeGenerator(), store, settings)
    await ingest.ingest("shipping", "Express delivery costs 49 euros and arrives next day.")
    await ingest.ingest("security", "Project tokens expire after 90 days.")
    return agent


async def test_relevant_passages_go_straight_to_generation() -> None:
    out = await (await _agent()).query("How much does express delivery cost?")
    assert out.steps[0].startswith("retrieve") and "rewrite" not in out.steps
    assert [c.source for c in out.citations] == ["shipping"]
    assert out.rewritten_query is None
    assert "[shipping#0]" in out.answer


async def test_rewrite_then_retry_when_nothing_relevant() -> None:
    out = await (await _agent(judge=ScriptedJudge())).query("What does express delivery cost?")
    assert out.steps.count("retrieve:2") == 2 and "rewrite" in out.steps
    assert out.rewritten_query == "express delivery cost"
    assert out.citations


async def test_refuses_after_rewrite_budget_is_spent() -> None:
    out = await (await _agent(max_rewrites=1)).query("Who founded the company?")
    assert out.steps.count("rewrite") == 1
    assert out.citations == []
    assert "No relevant context" in out.answer


async def test_unsupported_citations_are_removed() -> None:
    out = await (await _agent(generator=LyingGenerator())).query("express delivery cost?")
    assert out.unsupported_citations == ["invented#9"]
    assert "invented#9" not in out.answer
