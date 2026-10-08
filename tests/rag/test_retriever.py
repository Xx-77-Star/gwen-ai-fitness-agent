import pytest

from app.rag.models import DocumentChunk
from app.rag.retriever import KnowledgeRetriever
from tests.rag.fakes import FakeEmbeddingProvider, FakeVectorStore, search_result


def test_retriever_embeds_query_and_returns_top_k_chunks() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore(
        [
            search_result("力量训练。", source="training/muscle.md", score=0.95),
            search_result("蛋白质摄入。", source="nutrition/protein.md", score=0.82),
            search_result("睡眠恢复。", source="recovery/sleep.md", score=0.70),
        ]
    )
    retriever = KnowledgeRetriever(embedding_provider, vector_store, top_k=2)

    chunks = retriever.retrieve("如何安排力量训练？")

    assert embedding_provider.text_calls == ["如何安排力量训练？"]
    assert vector_store.calls[0][1] == 2
    assert [chunk.content for chunk in chunks] == ["力量训练。", "蛋白质摄入。"]
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)


def test_retriever_top_k_can_be_overridden() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore(
        [
            search_result("第一篇。", source="training/one.md", score=0.9),
            search_result("第二篇。", source="training/two.md", score=0.8),
        ]
    )
    retriever = KnowledgeRetriever(embedding_provider, vector_store)

    results = retriever.retrieve("训练", top_k=1)
    assert results == [
        search_result("第一篇。", source="training/one.md", score=0.9).chunk
    ]
    assert retriever.top_k == 3
    assert vector_store.calls[0][1] == 1


def test_retriever_rejects_empty_query_and_invalid_top_k() -> None:
    retriever = KnowledgeRetriever(FakeEmbeddingProvider(), FakeVectorStore([]))

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.retrieve(" ")
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.retrieve("训练", top_k=0)
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        KnowledgeRetriever(FakeEmbeddingProvider(), FakeVectorStore([]), top_k=0)


def test_retriever_uses_empty_vector_store_result() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore([])

    assert KnowledgeRetriever(embedding_provider, vector_store).retrieve("训练") == []
    assert embedding_provider.text_calls == ["训练"]