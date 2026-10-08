from typing import Any

from app.agent.nodes.rag_retrieve import rag_retrieve_node
from app.agent.nodes.response import response_node
from app.rag.retriever import KnowledgeRetriever
from app.rag.service import KnowledgeRetrievalService
from tests.rag.fakes import FakeEmbeddingProvider, FakeVectorStore, search_result


class RecordingLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def knowledge_service() -> KnowledgeRetrievalService:
    return KnowledgeRetrievalService(
        KnowledgeRetriever(
            FakeEmbeddingProvider(),
            FakeVectorStore(
                [
                    search_result(
                        "蛋白质应分配到多餐。",
                        source="nutrition/protein.md",
                        score=0.93,
                    ),
                    search_result("训练后保证睡眠。", source="recovery/sleep.md", score=0.81),
                    search_result("循序增加训练量。", source="training/muscle.md", score=0.77),
                ]
            ),
            top_k=2,
        )
    )


def test_rag_retrieve_node_replaces_context_with_top_k_metadata() -> None:
    result = rag_retrieve_node(
        {
            "input": "如何通过蛋白质和睡眠促进恢复？",
            "rag_context": [{"content": "旧内容", "source": "old", "score": 0.1}],
        },
        knowledge_service=knowledge_service(),
    )

    assert result["rag_context"] == [
        {"content": "蛋白质应分配到多餐。", "source": "nutrition/protein.md", "score": 0.93},
        {"content": "训练后保证睡眠。", "source": "recovery/sleep.md", "score": 0.81},
    ]


def test_rag_retrieve_node_returns_empty_context_for_empty_query() -> None:
    result = rag_retrieve_node({"input": "  "}, knowledge_service=knowledge_service())

    assert result == {"rag_context": []}


async def test_rag_context_is_injected_into_response_prompt() -> None:
    llm = RecordingLLM("结合蛋白质和睡眠建议安排恢复。")
    state: dict[str, Any] = {
        "input": "如何恢复？",
        "messages": [{"role": "user", "content": "如何恢复？"}],
        "conversation_history": [],
        "rag_context": [
            {
                "content": "蛋白质应分配到多餐。",
                "source": "nutrition/protein.md",
                "score": 0.93,
            }
        ],
        "answer_draft": None,
        "tool_results": [],
    }

    result = await response_node(state, llm_client=llm)

    prompt = llm.calls[0]["system_prompt"]
    assert "Knowledge Context:" in prompt
    assert "蛋白质应分配到多餐。" in prompt
    assert "Source: nutrition/protein.md" in prompt
    assert "score: 0.9300" in prompt
    assert "检索知识是优先参考依据" in prompt
    assert result["response"] == "结合蛋白质和睡眠建议安排恢复。"


def test_context_formatting_preserves_sources_and_handles_empty_context() -> None:
    formatted = KnowledgeRetrievalService.format_context(
        [{"content": "恢复知识。", "source": "recovery/sleep.md", "score": 0.5}]
    )

    assert formatted.startswith("Knowledge Context:")
    assert "Source: recovery/sleep.md" in formatted
    assert "恢复知识。" in formatted
    assert KnowledgeRetrievalService.format_context([]) == ""