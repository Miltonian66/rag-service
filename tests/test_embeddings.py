from app.embeddings import FakeEmbedder


async def test_fake_embedder_dimension():
    emb = FakeEmbedder(dim=64)
    vectors = await emb.embed(["a", "b"])
    assert len(vectors) == 2
    assert all(len(v) == 64 for v in vectors)


async def test_fake_embedder_deterministic():
    emb = FakeEmbedder(dim=64)
    v1 = await emb.embed_query("same text")
    v2 = await emb.embed_query("same text")
    assert v1 == v2


async def test_fake_embedder_distinguishes_text():
    emb = FakeEmbedder(dim=64)
    assert await emb.embed_query("alpha") != await emb.embed_query("beta")


async def test_fake_embedder_empty_list():
    emb = FakeEmbedder(dim=64)
    assert await emb.embed([]) == []
