from pathlib import Path

import pytest

from app.rag.document_loader import DocumentLoader


def test_load_markdown_document(tmp_path: Path) -> None:
    document_path = tmp_path / "training" / "muscle_gain.md"
    document_path.parent.mkdir()
    document_path.write_text("# 增肌\n\n渐进增加训练刺激。", encoding="utf-8")

    document = DocumentLoader().load(document_path)

    assert document.content == "# 增肌\n\n渐进增加训练刺激。"
    assert document.metadata == {
        "source": str(document_path.resolve()),
        "category": "training",
    }


def test_load_many_reads_multiple_markdown_files(tmp_path: Path) -> None:
    first_path = tmp_path / "training" / "first.md"
    second_path = tmp_path / "nutrition" / "second.md"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_text("第一篇内容", encoding="utf-8")
    second_path.write_text("第二篇内容", encoding="utf-8")

    documents = DocumentLoader().load_many([first_path, second_path])

    assert [document.content for document in documents] == ["第一篇内容", "第二篇内容"]


def test_load_directory_preserves_category_metadata_and_order(tmp_path: Path) -> None:
    training_path = tmp_path / "training" / "strength.md"
    nutrition_path = tmp_path / "nutrition" / "protein.md"
    training_path.parent.mkdir()
    nutrition_path.parent.mkdir()
    training_path.write_text("力量知识", encoding="utf-8")
    nutrition_path.write_text("蛋白质知识", encoding="utf-8")

    documents = DocumentLoader().load_directory(tmp_path)

    assert [document.metadata["source"] for document in documents] == [
        "nutrition/protein.md",
        "training/strength.md",
    ]
    assert [document.metadata["category"] for document in documents] == ["nutrition", "training"]


def test_load_directory_ignores_non_markdown_files(tmp_path: Path) -> None:
    markdown_path = tmp_path / "training" / "plan.md"
    text_path = tmp_path / "training" / "notes.txt"
    json_path = tmp_path / "nutrition" / "data.json"
    markdown_path.parent.mkdir()
    json_path.parent.mkdir()
    markdown_path.write_text("有效文档", encoding="utf-8")
    text_path.write_text("必须过滤", encoding="utf-8")
    json_path.write_text('{"content": "必须过滤"}', encoding="utf-8")

    documents = DocumentLoader().load_directory(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata["source"] == "training/plan.md"


def test_load_rejects_non_markdown_file(tmp_path: Path) -> None:
    text_path = tmp_path / "training" / "notes.txt"
    text_path.parent.mkdir()
    text_path.write_text("不是 Markdown", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported knowledge document"):
        DocumentLoader().load(text_path)