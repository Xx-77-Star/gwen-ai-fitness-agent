"""Embedding provider contracts and the Bailian embedding adapter."""

from collections.abc import Sequence
from typing import Any, Protocol

from openai import OpenAI

from app.config.settings import Settings, get_settings


class EmbeddingProvider(Protocol):
    """Provider-agnostic contract for generating text embeddings."""

    def embed_text(self, text: str) -> list[float]:
        """Return one embedding vector for a text input."""
        ...

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector for each document input, preserving order."""
        ...


class BailianEmbeddingProvider:
    """Generate embeddings through Alibaba Cloud Bailian's OpenAI-compatible API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: str = "text-embedding-v4",
        client: Any = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model = model
        self._client = client or OpenAI(
            api_key=self._settings.llm_api_key.get_secret_value(),
            base_url=self._settings.llm_base_url,
            timeout=self._settings.llm_timeout_seconds,
        )

    def embed_text(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        inputs = list(documents)
        if not inputs:
            return []
        if any(not document.strip() for document in inputs):
            raise ValueError("Embedding documents must not be empty")

        try:
            response = self._client.embeddings.create(model=self._model, input=inputs)
        except Exception as exc:
            raise RuntimeError("Bailian embedding request failed") from exc

        raw_vectors = getattr(response, "data", None)
        if raw_vectors is None or len(raw_vectors) != len(inputs):
            raise RuntimeError("Bailian returned an invalid embedding response")

        vectors = [self._normalize_vector(item.embedding) for item in raw_vectors]
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise RuntimeError("Bailian returned inconsistent embedding dimensions")
        return vectors

    @staticmethod
    def _normalize_vector(vector: Any) -> list[float]:
        try:
            normalized = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Bailian returned an invalid embedding vector") from exc
        if not normalized:
            raise RuntimeError("Bailian returned an empty embedding vector")
        return normalized