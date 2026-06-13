async def test_query_happy_path(client):
    async with client as c:
        await c.post(
            "/documents",
            json={"source": "kb", "text": "RAG combines retrieval and generation. " * 10},
        )
        resp = await c.post("/query", json={"question": "What is RAG?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert len(body["citations"]) > 0
    assert body["citations"][0]["source"] == "kb"


async def test_query_empty_question_rejected(client):
    async with client as c:
        resp = await c.post("/query", json={"question": ""})
    assert resp.status_code == 422


async def test_query_no_documents(client):
    async with client as c:
        resp = await c.post("/query", json={"question": "anything?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"] == []


async def test_query_stream_sse(client):
    async with client as c:
        await c.post(
            "/documents",
            json={"source": "kb", "text": "streaming context here. " * 10},
        )
        resp = await c.post("/query", json={"question": "go?", "stream": True})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        text = resp.text
    assert "event: citations" in text
    assert "event: token" in text
    assert "[DONE]" in text
