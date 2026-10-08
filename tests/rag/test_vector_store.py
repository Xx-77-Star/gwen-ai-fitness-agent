from pathlib import Path

import pytest

from app.rag.models import DocumentChunk
from app.rag.vector_store import VectorStore
from tests.rag.fakes import FakeEmbeddingProvider


def chunk(content: str, *, category: str) -> DocumentChunk:
    return DocumentChunk(
        id=f"{category}-{content}",
        content=content,
        metadata={"source": f"{category}/fake.md", "category": category},
    )


def test_add_chunks_generates_vectors_for_chunk_content() -> None:
    embedding_provider = FakeEmbeddingProvider()
    store = VectorStore(embedding_provider)
    chunks = [chunk("力量训练。", category="training"), chunk("睡眠恢复。", category="recovery")]

    store.add_chunks(chunks)

    assert embedding_provider.document_calls == [["力量训练。", "睡眠恢复。"]]
    assert store.dimension == 4
    assert len(store) == 2


def test_similarity_search_uses_query_embedding_and_returns_best_chunks() -> None:
    embedding_provider = FakeEmbeddingProvider()
    store = VectorStore(embedding_provider)
    store.add_chunks(
        [
            chunk("蛋白质摄入。", category="nutrition"),
            chunk("力量训练。", category="training"),
            chunk("训练计划。", category="training"),
        ]
    )

    results = store.similarity_search("训练", limit=2)

    assert embedding_provider.text_calls == ["训练"]
    assert [result.chunk.metadata["source"] for result in results] == [
        "training/fake.md",
        "training/fake.md",
    ]
    assert results[0].score >= results[1].score


def test_index_save_and_load_preserves_search_results(tmp_path: Path) -> None:
    embedding_provider = FakeEmbeddingProvider()
    store = VectorStore(embedding_provider)
    chunks = [chunk("睡眠恢复。", category="recovery"), chunk("蛋白质摄入。", category="nutrition")]
    store.add_chunks(chunks)

    store.save(tmp_path)
    restored = VectorStore.load(tmp_path, embedding_provider)

    assert (tmp_path / "index.faiss").is_file()
    assert (tmp_path / "chunks.json").is_file()
    assert len(restored) == 2
    assert restored.dimension == 4
    assert restored.similarity_search("睡眠恢复。", limit=1)[0].chunk == chunks[0]


def test_empty_store_has_no_search_results(tmp_path: Path) -> None:
    store = VectorStore(FakeEmbeddingProvider())

    assert store.similarity_search("训练") == []
    with pytest.raises(ValueError, match="empty vector store"):
        store.save(tmp_path)


def test_similarity_search_validates_query_and_limit() -> None:
    store = VectorStore(FakeEmbeddingProvider())

    with pytest.raises(ValueError, match="Query must not be empty"):
        store.similarity_search(" ")
    with pytest.raises(ValueError, match="limit must be at least 1"):
        store.similarity_search("训练", limit=0)