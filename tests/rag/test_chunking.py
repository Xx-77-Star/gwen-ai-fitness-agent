from pathlib import Path

from app.rag.chunking import Chunker
from app.rag.document_loader import DocumentLoader
from app.rag.models import KnowledgeDocument


def test_chunker_splits_document_into_natural_paragraphs() -> None:
    document = KnowledgeDocument(
        content="# 标题\n\n第一段。\n\n第二段。\n\n第三段包含\n两行。",
        metadata={"source": "training/fake.md", "category": "training"},
    )

    chunks = Chunker().chunk(document)

    assert [chunk.content for chunk in chunks] == [
        "# 标题",
        "第一段。",
        "第二段。",
        "第三段包含\n两行。",
    ]
    assert len(chunks) == 4


def test_chunks_keep_document_metadata() -> None:
    metadata = {"source": "nutrition/fake-protein.md", "category": "nutrition"}

    chunks = Chunker().chunk_text("第一段。\n\n第二段。", metadata)

    assert [chunk.metadata for chunk in chunks] == [metadata, metadata]
    assert chunks[0].metadata is not metadata


def test_empty_document_produces_no_chunks() -> None:
    empty_document = KnowledgeDocument(
        content=" \n\n  ",
        metadata={"source": "recovery/empty.md", "category": "recovery"},
    )

    assert Chunker().chunk(empty_document) == []


def test_chunk_ids_are_unique_and_deterministic() -> None:
    document = KnowledgeDocument(
        content="第一段。\n\n第二段。",
        metadata={"source": "training/stable.md", "category": "training"},
    )

    first_result = Chunker().chunk(document)
    second_result = Chunker().chunk(document)

    assert [chunk.id for chunk in first_result] == [chunk.id for chunk in second_result]
    assert len({chunk.id for chunk in first_result}) == 2


def test_loader_and_chunker_handle_multiple_fake_files(tmp_path: Path) -> None:
    first_path = tmp_path / "training" / "first.md"
    second_path = tmp_path / "recovery" / "second.md"
    ignored_path = tmp_path / "recovery" / "ignored.txt"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_text("第一篇标题\n\n第一篇正文。", encoding="utf-8")
    second_path.write_text("第二篇正文。", encoding="utf-8")
    ignored_path.write_text("非 Markdown 内容。", encoding="utf-8")

    documents = DocumentLoader().load_directory(tmp_path)
    chunks = [chunk for document in documents for chunk in Chunker().chunk(document)]

    assert len(documents) == 2
    assert len(chunks) == 3
    assert {chunk.metadata["category"] for chunk in chunks} == {"training", "recovery"}