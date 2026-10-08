from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.rag.embedding import BailianEmbeddingProvider, EmbeddingProvider


class FakeEmbeddings:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.request: dict[str, object] | None = None

    def create(self, **kwargs) -> SimpleNamespace:
        self.request = kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=vector) for vector in self.vectors]
        )


def test_bailian_provider_embeds_documents_with_configured_credentials() -> None:
    embeddings = FakeEmbeddings([[1.0, 2.0], [3.0, 4.0]])
    fake_client = SimpleNamespace(embeddings=embeddings)
    settings = Settings(LLM_API_KEY="fake-key", LLM_BASE_URL="https://fake.example/v1")
    provider: EmbeddingProvider = BailianEmbeddingProvider(settings, client=fake_client)

    vectors = provider.embed_documents(["力量训练", "蛋白质摄入"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert embeddings.request == {
        "model": "text-embedding-v4",
        "input": ["力量训练", "蛋白质摄入"],
    }


def test_bailian_provider_embeds_single_text() -> None:
    embeddings = FakeEmbeddings([[0.1, 0.2]])
    provider = BailianEmbeddingProvider(
        Settings(LLM_API_KEY="fake-key"),
        client=SimpleNamespace(embeddings=embeddings),
    )

    assert provider.embed_text("睡眠恢复") == [0.1, 0.2]


def test_bailian_provider_rejects_empty_documents() -> None:
    provider = BailianEmbeddingProvider(
        Settings(LLM_API_KEY="fake-key"),
        client=SimpleNamespace(embeddings=FakeEmbeddings([])),
    )

    assert provider.embed_documents([]) == []
    with pytest.raises(ValueError, match="must not be empty"):
        provider.embed_documents([""])