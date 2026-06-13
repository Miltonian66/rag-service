async def test_health(client):
    async with client as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["embedder"] == "fake"
    assert body["generator"] == "fake"
