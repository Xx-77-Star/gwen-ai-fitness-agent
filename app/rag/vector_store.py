"""Persistent FAISS vector storage for FitLife knowledge chunks."""

import json
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np

from app.rag.embedding import EmbeddingProvider
from app.rag.models import DocumentChunk, VectorSearchResult

_INDEX_FILENAME = "index.faiss"
_METADATA_FILENAME = "chunks.json"


class VectorStore:
    """Build and search a normalized cosine-similarity FAISS index."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embedding_provider = embedding_provider
        self._index: faiss.IndexFlatIP | None = None
        self._dimension: int | None = None
        self._chunks: list[DocumentChunk] = []

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def __len__(self) -> int:
        return len(self._chunks)

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            return

        vectors = self._embedding_provider.embed_documents(
            [chunk.content for chunk in chunk_list]
        )
        if len(vectors) != len(chunk_list):
            raise ValueError("Embedding provider returned the wrong number of vectors")

        matrix = self._vector_matrix(vectors)
        dimension = matrix.shape[1]
        if self._dimension is None:
            self._index = faiss.IndexFlatIP(dimension)
            self._dimension = dimension
        elif dimension != self._dimension:
            raise ValueError("Embedding vectors must have a consistent dimension")

        faiss.normalize_L2(matrix)
        self._index.add(matrix)  # type: ignore[union-attr]
        self._chunks.extend(chunk.model_copy(deep=True) for chunk in chunk_list)

    def similarity_search(
        self,
        query: str | Sequence[float],
        *,
        limit: int = 5,
    ) -> list[VectorSearchResult]:
        if isinstance(query, str) and not query.strip():
            raise ValueError("Query must not be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not self._index or not self._chunks:
            return []

        query_embedding = (
            self._embedding_provider.embed_text(query)
            if isinstance(query, str)
            else query
        )
        query_vector = self._vector_matrix([query_embedding])
        if query_vector.shape[1] != self._dimension:
            raise ValueError("Query vector dimension does not match the index")

        faiss.normalize_L2(query_vector)
        result_count = min(limit, len(self._chunks))
        scores, indexes = self._index.search(query_vector, result_count)
        results = []
        for score, index in zip(scores[0], indexes[0], strict=True):
            if int(index) < 0:
                continue
            results.append(
                VectorSearchResult(
                    chunk=self._chunks[int(index)].model_copy(deep=True),
                    score=float(score),
                )
            )
        return results

    def save(self, directory: Path | str) -> None:
        if self._index is None:
            raise ValueError("Cannot save an empty vector store")

        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(output_directory / _INDEX_FILENAME))
        metadata = {
            "dimension": self._dimension,
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks],
        }
        (output_directory / _METADATA_FILENAME).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: Path | str,
        embedding_provider: EmbeddingProvider,
    ) -> "VectorStore":
        input_directory = Path(directory)
        index_path = input_directory / _INDEX_FILENAME
        metadata_path = input_directory / _METADATA_FILENAME
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(input_directory)

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            chunks = [DocumentChunk.model_validate(item) for item in metadata["chunks"]]
            dimension = int(metadata["dimension"])
            index = faiss.read_index(str(index_path))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError("Vector store metadata is invalid") from exc

        if index.d != dimension or index.ntotal != len(chunks):
            raise ValueError("Vector store index and metadata do not match")

        store = cls(embedding_provider)
        store._index = index
        store._dimension = dimension
        store._chunks = chunks
        return store

    @classmethod
    def _vector_matrix(cls, vectors: Sequence[Sequence[float]]) -> np.ndarray:
        rows = [[float(value) for value in vector] for vector in vectors]
        if not rows or not rows[0]:
            raise ValueError("Embedding vectors must not be empty")
        if any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("Embedding vectors must have a consistent dimension")
        return np.asarray(rows, dtype=np.float32)