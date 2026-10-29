from app.citations import cited_ids, verify_citations
from app.vectorstore import SearchResult


def _ctx(source: str, index: int) -> SearchResult:
    return SearchResult(source=source, chunk_index=index, text="t", score=1.0)


def test_cited_ids_dedupes_in_order() -> None:
    assert cited_ids("A [kb#1] and [kb#0], again [kb#1].") == ["kb#1", "kb#0"]


def test_verify_strips_citations_to_passages_not_given() -> None:
    check = verify_citations("Fact [kb#0]. Invented [other#7].", [_ctx("kb", 0)])
    assert check.supported == ["kb#0"]
    assert check.unsupported == ["other#7"]
    assert check.answer == "Fact [kb#0]. Invented ."


def test_verify_keeps_answer_untouched_when_all_supported() -> None:
    answer = "Fact [kb#0]."
    assert verify_citations(answer, [_ctx("kb", 0)]).answer == answer
