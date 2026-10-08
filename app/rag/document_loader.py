"""Load local Markdown knowledge documents into RAG-ready document models."""

from collections.abc import Iterable
from pathlib import Path

from app.rag.models import KnowledgeDocument

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})


class DocumentLoader:
    """Read Markdown files and attach stable source/category metadata."""

    def is_markdown(self, path: Path | str) -> bool:
        return Path(path).suffix.lower() in MARKDOWN_SUFFIXES

    def load(
        self,
        path: Path | str,
        *,
        category: str | None = None,
        source: str | None = None,
    ) -> KnowledgeDocument:
        file_path = Path(path)
        if not self.is_markdown(file_path):
            raise ValueError(f"Unsupported knowledge document: {file_path}")
        if not file_path.is_file():
            raise FileNotFoundError(file_path)

        resolved_path = file_path.resolve()
        metadata = {
            "source": source or str(resolved_path),
            "category": category or self._category_for_file(resolved_path),
        }
        return KnowledgeDocument(content=file_path.read_text(encoding="utf-8"), metadata=metadata)

    def load_many(self, paths: Iterable[Path | str]) -> list[KnowledgeDocument]:
        return [self.load(path) for path in paths if self.is_markdown(path)]

    def load_directory(self, directory: Path | str) -> list[KnowledgeDocument]:
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(root)

        documents = []
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file() or not self.is_markdown(file_path):
                continue
            relative_path = file_path.relative_to(root)
            category = (
                relative_path.parts[0]
                if len(relative_path.parts) > 1
                else self._category_for_file(file_path)
            )
            documents.append(
                self.load(file_path, category=category, source=relative_path.as_posix())
            )
        return documents

    @staticmethod
    def _category_for_file(path: Path) -> str:
        return path.parent.name or "general"