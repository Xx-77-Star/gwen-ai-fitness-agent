"""Data models shared by the FitLife RAG ingestion pipeline."""

from pydantic import BaseModel, Field


class KnowledgeDocument(BaseModel):
    """A Markdown knowledge document before it is split into retrieval chunks."""

    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A natural paragraph prepared for later indexing and retrieval."""

    id: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class VectorSearchResult(BaseModel):
    """A retrieved chunk and its normalized cosine similarity score."""

    chunk: DocumentChunk
    score: float