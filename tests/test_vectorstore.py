from app.embeddings import FakeEmbedder
from app.vectorstore import ChunkRecord, InMemoryVectorStore


async def test_topk_ranking_by_cosine():
    emb = FakeEmbedder(dim=64)
    store = InMemoryVectorStore()
    texts = ["the cat sat", "dogs run fast", "quantum physics"]
    vecs = await emb.embed(texts)
    await store.add(
        [
            ChunkRecord(source="doc", chunk_index=i, text=t, embedding=v)
            for i, (t, v) in enumerate(zip(texts, vecs, strict=True))
        ]
    )
    # Query identical to first chunk -> it must rank #1 with score ~1.0
    query_vec = await emb.embed_query("the cat sat")
    results = await store.search(query_vec, top_k=2)
    assert len(results) == 2
    assert results[0].text == "the cat sat"
    assert results[0].score > 0.99
    assert results[0].score >= results[1].score


async def test_topk_limits_results():
    store = InMemoryVectorStore()
    await store.add(
        [
            ChunkRecord(source="d", chunk_index=i, text=str(i), embedding=[float(i)] * 4)
            for i in range(10)
        ]
    )
    results = await store.search([1.0, 1.0, 1.0, 1.0], top_k=3)
    assert len(results) == 3


async def test_empty_store_returns_empty():
    store = InMemoryVectorStore()
    assert await store.search([0.1, 0.2], top_k=5) == []
