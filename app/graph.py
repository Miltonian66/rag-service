"""Agentic RAG as a LangGraph state machine.

    retrieve ─▶ grade ─┬─(some passage relevant)──────────────▶ generate ─▶ verify ─▶ END
                       ├─(none relevant, rewrites left)─▶ rewrite ─▶ retrieve
                       └─(none relevant, no rewrites left)──▶ generate (no context → refusal)

The judge (an LLM) grades passages and rewrites queries; routing, the rewrite
budget and citation verification are plain code. ``steps`` records the path taken
so every answer can be explained, and the same run can be traced in Langfuse.
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.citations import verify_citations
from app.config import Settings
from app.embeddings import Embedder
from app.judge import Judge
from app.llm import Generator
from app.schemas import AgenticQueryOut, Citation
from app.tracing import langfuse_callbacks
from app.vectorstore import SearchResult, VectorStore


class RAGState(TypedDict, total=False):
    question: str
    query: str
    top_k: int
    rewrites: int
    retrieved: list[SearchResult]
    relevant: list[SearchResult]
    answer: str
    unsupported: list[str]
    steps: Annotated[list[str], operator.add]


class AgenticRAG:
    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
        store: VectorStore,
        judge: Judge,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._generator = generator
        self._store = store
        self._judge = judge
        self._settings = settings
        self.graph = self._build()

    # --- nodes -----------------------------------------------------------------

    async def _retrieve(self, state: RAGState) -> dict[str, Any]:
        vec = await self._embedder.embed_query(state["query"])
        hits = await self._store.search(vec, state["top_k"])
        return {"retrieved": hits, "steps": [f"retrieve:{len(hits)}"]}

    async def _grade(self, state: RAGState) -> dict[str, Any]:
        hits = state["retrieved"]
        verdicts = await asyncio.gather(
            *(self._judge.grade(state["question"], h.text) for h in hits)
        )
        relevant = [h for h, ok in zip(hits, verdicts, strict=True) if ok]
        return {"relevant": relevant, "steps": [f"grade:{len(relevant)}/{len(hits)}"]}

    async def _rewrite(self, state: RAGState) -> dict[str, Any]:
        query = await self._judge.rewrite(state["question"])
        return {"query": query, "rewrites": state["rewrites"] + 1, "steps": ["rewrite"]}

    async def _generate(self, state: RAGState) -> dict[str, Any]:
        answer = await self._generator.generate(state["question"], state["relevant"])
        return {"answer": answer, "steps": ["generate"]}

    async def _verify(self, state: RAGState) -> dict[str, Any]:
        check = verify_citations(state["answer"], state["relevant"])
        return {
            "answer": check.answer,
            "unsupported": check.unsupported,
            "steps": [f"verify:{len(check.unsupported)}_unsupported"],
        }

    def _route_after_grade(self, state: RAGState) -> str:
        if state["relevant"]:
            return "generate"
        if state["rewrites"] < self._settings.max_rewrites:
            return "rewrite"
        return "generate"

    def _build(self):
        g = StateGraph(RAGState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("rewrite", self._rewrite)
        g.add_node("generate", self._generate)
        g.add_node("verify", self._verify)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade", self._route_after_grade, {"rewrite": "rewrite", "generate": "generate"}
        )
        g.add_edge("rewrite", "retrieve")
        g.add_edge("generate", "verify")
        g.add_edge("verify", END)
        return g.compile()

    # --- entry point -------------------------------------------------------------

    async def query(self, question: str, top_k: int | None = None) -> AgenticQueryOut:
        initial: RAGState = {
            "question": question,
            "query": question,
            "top_k": top_k or self._settings.top_k,
            "rewrites": 0,
            "steps": [],
        }
        config = {"callbacks": langfuse_callbacks(), "run_name": "agentic-rag"}
        final = await self.graph.ainvoke(initial, config=config)
        return AgenticQueryOut(
            answer=final["answer"],
            citations=[
                Citation(source=c.source, chunk_index=c.chunk_index, score=c.score, text=c.text)
                for c in final["relevant"]
            ],
            rewritten_query=final["query"] if final["rewrites"] else None,
            unsupported_citations=final["unsupported"],
            steps=final["steps"],
        )
