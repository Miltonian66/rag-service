# RAG Service

> Production-grade Retrieval-Augmented Generation API — ingest documents, retrieve with pgvector, and answer with inline-cited sources via Claude. Two pipelines (simple and agentic LangGraph), an offline eval harness that gates CI, and optional Langfuse tracing.

[![CI](https://github.com/Miltonian66/rag-service/actions/workflows/ci.yml/badge.svg)](https://github.com/Miltonian66/rag-service/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A compact but realistic RAG backend: async FastAPI, async SQLAlchemy 2.0 over
PostgreSQL + pgvector, cosine retrieval, and grounded answer generation with
Claude — including SSE streaming. The whole thing is built around three
provider abstractions (Embedder / Generator / VectorStore), each with a real and
a fake implementation, so the test suite runs green offline with **no Postgres
and no API keys**.

## Highlights

- **Async end-to-end** — FastAPI + async SQLAlchemy 2.0 + asyncpg, no blocking calls in request paths.
- **pgvector cosine retrieval** — top-k nearest neighbours via the `<=>` operator with an `ivfflat` index.
- **Grounded generation with citations** — Claude Opus 5.5 answers strictly from retrieved passages and cites `[source#chunk_index]`; effort is configurable and a safety-classifier refusal surfaces as an error, never as an empty answer.
- **SSE streaming** — `/query?stream=true` streams a `citations` event followed by incremental `token` events.
- **Provider abstractions (Real + Fake)** — `Embedder`, `Generator`, `VectorStore`. Tests swap in `FakeEmbedder` / `FakeGenerator` / `InMemoryVectorStore` via FastAPI `dependency_overrides`, giving a fully offline, network-free, Postgres-free CI.
- **Agentic mode on LangGraph** — `mode: "agentic"` runs retrieve → grade → (rewrite → retrieve) → generate → verify. Claude grades each passage and rewrites the query; routing, the rewrite budget and citation checks are plain code. Every answer carries the `steps` it took.
- **Citations verified in code** — any `[source#chunk]` the model cites that was not in its context is stripped and reported in `unsupported_citations`.
- **Fail-soft judge** — on an API error, refusal or malformed JSON the grader keeps the passage and the rewriter keeps the original question; the request never fails because the judge did.
- **Eval harness as a CI gate** — 27 golden questions (4 deliberately unanswerable) over a bundled corpus; hit rate, MRR, context and citation precision, abstention. Runs offline in CI against `evals/thresholds.json`; `--live` re-runs it on real models.
- **Langfuse tracing (optional)** — set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` and every agentic run is traced node by node; otherwise a no-op.
- **Async Alembic migrations** — initial migration creates the `vector` extension, the `chunks` table, and the cosine index.
- **One-command Docker Compose** — `db` (pgvector) + `api` (runs `alembic upgrade head`, then uvicorn).

## Architecture

```
Ingest:  text ─▶ chunk_text() ─▶ Embedder.embed() ─▶ VectorStore.add() ─▶ Postgres/pgvector

Query:   question ─▶ Embedder.embed_query() ─▶ VectorStore.search(<=>, top-k)
                   ─▶ contexts ─▶ Generator.generate()/.stream() ─▶ answer + citations (+ SSE)

Agentic (LangGraph, app/graph.py):
         retrieve ─▶ grade ─┬─ some relevant ─────────────────────▶ generate ─▶ verify ─▶ answer
                            ├─ none relevant, rewrites left ─▶ rewrite ─▶ retrieve
                            └─ none relevant, budget spent ──▶ generate (no context → "not found")
```

The judge (`app/judge.py`) answers two questions with structured JSON output:
*is this passage relevant?* and *how should this question be phrased for search?*
It runs on `claude-opus-5-5` at low effort by default (`JUDGE_MODEL`,
`JUDGE_EFFORT`); the passage is wrapped as data so instructions inside documents
are not followed.

The orchestration lives in `RAGService` (`app/rag.py`); each stage is a provider
behind a `Protocol`, resolved by FastAPI dependencies (`app/dependencies.py`).

## Tech stack

Python 3.11 · FastAPI (async) · LangGraph · Langfuse (optional) · async SQLAlchemy 2.0 + asyncpg · PostgreSQL +
pgvector · Alembic · Pydantic v2 + pydantic-settings · anthropic · openai · uv ·
ruff · pytest.

## Quick start

### Docker Compose (everything wired)

```bash
cp .env.example .env          # fill in OPENAI_API_KEY / ANTHROPIC_API_KEY
docker compose up --build     # starts db, runs alembic upgrade head, then uvicorn
```

The API is then on <http://localhost:8000> (`/docs` for the OpenAPI UI).

### Local development

```bash
uv sync --extra dev
cp .env.example .env          # set DATABASE_URL + keys (or use the fake providers)
uv run alembic upgrade head   # against a running pgvector Postgres
uv run uvicorn app.main:app --reload
```

To run locally with zero external dependencies, set
`EMBEDDING_PROVIDER=fake` and `LLM_PROVIDER=fake` in `.env`.

## API endpoints

| Method | Path         | Body                            | Response                          |
| ------ | ------------ | ------------------------------- | --------------------------------- |
| POST   | `/documents` | `{source, text}`                | `{source, chunks_ingested}`       |
| POST   | `/query`     | `{question, top_k?, stream?, mode?}` | `{answer, citations[]}` or SSE; agentic adds `steps`, `rewritten_query`, `unsupported_citations` |
| GET    | `/health`    | —                               | `{status, embedder, generator}`   |

When `stream=true`, `/query` returns `text/event-stream`: one `citations` event
(JSON array), then `token` events, then a `done` event with `[DONE]`.

## Design decisions & trade-offs

- **pgvector instead of a dedicated vector DB.** One transactional database for
  both chunks and embeddings keeps the infrastructure minimal and lets retrieval
  ride on plain SQL. An `ivfflat` cosine index is more than enough at this scale;
  a separate vector store (Pinecone, Qdrant, Milvus) would add operational weight
  this project doesn't need.
- **Provider abstraction (Embedder / Generator / VectorStore).** Each capability
  is a `Protocol` with a real and a fake implementation. This is what makes CI
  green without a network or a database: tests override the DI factories with
  `FakeEmbedder` / `FakeGenerator` / `InMemoryVectorStore`. It also makes swapping
  backends (e.g. OpenAI ↔ Claude for generation) a one-line change.
- **OpenAI for embeddings while Claude generates.** Anthropic does not ship a
  first-class embeddings API, so embeddings come from OpenAI's
  `text-embedding-3-small` (1536 dims) — a deliberate split, not an oversight.
  Generation uses Claude (`claude-opus-5-5`, `LLM_EFFORT=low` by default).
- **Chunk size 1000 / overlap 200.** A pragmatic balance: chunks large enough to
  carry self-contained context, with 20% overlap so answers that straddle a
  boundary still retrieve cleanly. Both are configurable via env.
- **The model judges meaning, code owns the contract.** Relevance and query
  phrasing are semantic, so the LLM decides them; how many retries, what counts
  as a valid citation and when to refuse are deterministic and unit-tested.
- **Offline evals measure the pipeline, live evals measure the models.** In CI
  the embedder is a bag-of-words hasher (`LexicalEmbedder`) and the judge is a
  word-overlap fake, so the gate checks retrieval and control flow for free and
  deterministically. It cannot see paraphrases; semantic grading quality is
  measured with `evals.run --live`, which costs API calls and is run by hand.
- **Not done (intentionally out of scope for a showcase):** authentication,
  a cross-encoder reranker (the grader acts as a relevance filter instead), and
  LLM-as-judge scoring of answer faithfulness in live evals.

## Project structure

```
rag-service/
├── app/
│   ├── config.py          # Settings (pydantic-settings), SecretStr, get_settings()
│   ├── database.py        # async engine, sessionmaker, get_session() DI, Base
│   ├── models.py          # SQLAlchemy 2.0 ORM: Chunk with Vector(EMBED_DIM)
│   ├── schemas.py         # Pydantic v2 request/response models
│   ├── chunking.py        # chunk_text() — pure character chunking with overlap
│   ├── embeddings.py      # Embedder Protocol + RealEmbedder(OpenAI) + FakeEmbedder
│   ├── llm.py             # Generator Protocol + Claude/OpenAI/Fake generators
│   ├── vectorstore.py     # VectorStore Protocol + PgVectorStore + InMemoryVectorStore
│   ├── rag.py             # RAGService — ingest()/query()/query_stream()
│   ├── graph.py           # AgenticRAG — LangGraph retrieve/grade/rewrite/generate/verify
│   ├── judge.py           # Judge Protocol + ClaudeJudge (structured output) + FakeJudge
│   ├── citations.py       # deterministic [source#chunk] verification
│   ├── tracing.py         # optional Langfuse callbacks
│   ├── dependencies.py    # DI factories (overridden in tests)
│   ├── main.py            # FastAPI app + routers
│   └── api/               # documents / query / health routers
├── alembic/               # async env.py + initial migration (vector ext + index)
├── evals/                 # golden set, corpus, metrics, runner, CI thresholds
├── tests/                 # offline suite (Fake/InMemory via dependency_overrides)
├── docker-compose.yml     # db (pgvector) + api (alembic + uvicorn)
├── Dockerfile             # multi-stage, non-root, healthcheck
└── pyproject.toml         # uv project, deps, ruff, pytest config
```

## Evaluation

```bash
uv run python -m evals.run                                  # offline report
uv run python -m evals.run --check evals/thresholds.json    # CI gate, exit 1 on regression
uv run python -m evals.run --live --out evals/live.json     # real providers from .env
```

Offline baseline (27 cases, top-k 3):

| metric | simple | agentic |
|---|---|---|
| hit rate | 1.00 | 0.96 |
| MRR | 0.84 | 0.94 |
| context precision | 0.46 | 0.83 |
| abstention on unanswerable | 0.00 | 0.25 |
| wrong abstention on answerable | 0.00 | 0.00 |
| unsupported citations | 0 | 0 |

The agentic pipeline trades a little recall for much cleaner context and learns
to say "not found". With the strict word-overlap fake (threshold 0.5) it also
refuses 35% of answerable questions — paraphrases like "yearly" vs "annual" — which
is exactly the gap the Claude judge closes in live mode.

## Testing

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The suite runs fully offline — no Postgres, no live API keys. Pure-logic tests
(chunking, embeddings, vector store, RAG service) construct the fakes directly;
API tests drive the app through `httpx.ASGITransport` with the DI factories
overridden to `FakeEmbedder` / `FakeGenerator` / `InMemoryVectorStore`.
Overriding `get_vector_store` is what cuts the entire
`get_session → asyncpg → Postgres` chain out of the request path.

## License

MIT © 2026 Konstantin Fatykov
