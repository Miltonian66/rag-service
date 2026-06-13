async def test_ingest_endpoint(client):
    async with client as c:
        resp = await c.post("/documents", json={"source": "doc1", "text": "hello " * 100})
    assert resp.status_code == 201
    body = resp.json()
    assert body["source"] == "doc1"
    assert body["chunks_ingested"] > 0


async def test_ingest_rejects_empty_text(client):
    async with client as c:
        resp = await c.post("/documents", json={"source": "doc1", "text": ""})
    assert resp.status_code == 422  # pydantic min_length validation
