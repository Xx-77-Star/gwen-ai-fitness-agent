"""Split Markdown knowledge documents into natural-paragraph retrieval chunks."""

import re
from uuid import NAMESPACE_URL, uuid5

from app.rag.models import DocumentChunk, KnowledgeDocument

_PARAGRAPH_SEPARATOR = re.compile(r"\n[ \t]*\n+")


class Chunker:
    """Create deterministic chunks from blank-line-separated Markdown paragraphs."""

    def chunk(self, document: KnowledgeDocument) -> list[DocumentChunk]:
        return self.chunk_text(document.content, document.metadata)

    def chunk_text(
        self,
        content: str,
        metadata: dict[str, str],
    ) -> list[DocumentChunk]:
        normalized_content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized_content:
            return []

        chunks = []
        for index, paragraph in enumerate(_PARAGRAPH_SEPARATOR.split(normalized_content)):
            chunk_content = paragraph.strip()
            if not chunk_content:
                continue
            source = metadata.get("source", "unknown")
            chunk_id = str(uuid5(NAMESPACE_URL, f"fitlife://{source}#paragraph-{index}"))
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    content=chunk_content,
                    metadata=dict(metadata),
                )
            )
        return chunks