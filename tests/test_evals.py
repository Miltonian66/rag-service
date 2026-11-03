from app.embeddings import LexicalEmbedder
from app.vectorstore import _cosine
from evals.metrics import score_case, summarize
from evals.run import check


async def test_lexical_embedder_ranks_shared_words_higher() -> None:
    emb = LexicalEmbedder(512)
    q, near, far = await emb.embed(["refund window", "the refund window is 30 days", "49 euros"])
    assert _cosine(q, near) > _cosine(q, far)
    assert await emb.embed_query("refund window") == q


def test_score_case_retrieval_and_citations() -> None:
    case = {"id": "x", "sources": ["refunds"], "must_include": ["30 days"]}
    r = score_case(
        case, "Within 30 days [refunds#0] [pricing#1]", ["pricing", "refunds"], score_answer=True
    )
    assert r.hit and r.reciprocal_rank == 0.5
    assert r.context_precision == 0.5 and r.citation_precision == 0.5
    assert r.coverage == 1.0 and not r.abstained


def test_unanswerable_case_counts_abstention() -> None:
    case = {"id": "none", "sources": [], "must_include": []}
    yes = score_case(case, "No relevant context was found to answer the question.", [])
    no = score_case(case, "Sure [kb#0]", ["kb"])
    summary = summarize([yes, no])
    assert summary["abstention_rate"] == 0.5 and summary["hit_rate"] is None


def test_check_reports_regressions() -> None:
    report = {"pipelines": {"agentic": {"summary": {"hit_rate": 0.5, "unsupported_citations": 2}}}}
    failures = check(
        report, {"agentic.hit_rate": {"min": 0.9}, "agentic.unsupported_citations": {"max": 0}}
    )
    assert len(failures) == 2
