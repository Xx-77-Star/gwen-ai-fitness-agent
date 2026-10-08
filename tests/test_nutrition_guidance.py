"""Nutrition Guidance: intent, memory linkage, RAG knowledge and prompt scope."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.agent.graph import create_agent_graph
from app.agent.nodes.intent import classify_intent
from app.agent.nodes.memory_prompt import build_memory_prompt
from app.agent.nodes.memory_retrieval import memory_retrieve_node
from app.agent.nodes.response import build_system_prompt
from app.memory.models import MemoryContext, UserMemory
from app.rag.chunking import Chunker
from app.rag.document_loader import DocumentLoader
from app.rag.models import VectorSearchResult
from app.rag.retriever import KnowledgeRetriever
from app.rag.service import KnowledgeRetrievalService

KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1] / "knowledge"


# ---------------------------------------------------------------------------
# 1. Intent: nutrition_advice
# ---------------------------------------------------------------------------
def test_nutrition_advice_intent_for_muscle_gain_question() -> None:
    assert classify_intent("我的目标是增肌，我应该怎么吃") == "nutrition_advice"


def test_nutrition_advice_intent_for_fat_loss_question() -> None:
    assert classify_intent("我想减脂期间怎么安排饮食") == "nutrition_advice"


def test_nutrition_advice_intent_for_timing_questions() -> None:
    assert classify_intent("训练后应该补充什么") == "nutrition_advice"
    assert classify_intent("今天训练前吃什么比较好") == "nutrition_advice"


def test_existing_intents_are_unchanged() -> None:
    assert classify_intent("帮我制定今天的训练计划") == "training"
    assert classify_intent("今天应该怎么训练？") == "training"
    assert classify_intent("晚餐应该吃什么？") == "nutrition"
    assert classify_intent("最近睡眠不好怎么恢复？") == "recovery"
    assert classify_intent("帮我安排这一周的生活") == "lifestyle"
    assert classify_intent("你好") == "general"


# ---------------------------------------------------------------------------
# 2. Memory linkage: nutrition advice reads fitness_goal
# ---------------------------------------------------------------------------
class _MemoryService:
    """Minimal MemoryServiceInterface stub backed by real MemoryContext models."""

    def __init__(self, facts):
        self._facts = facts

    def retrieve(self, user_id, *, short_term):
        del short_term
        return MemoryContext(user_id=user_id, short_term=[], long_term=list(self._facts))

    def remember(self, user_id, memory_key, memory_value):
        del user_id, memory_key, memory_value
        raise NotImplementedError


def _fact(key: str, value: str) -> UserMemory:
    now = datetime.now(UTC)
    return UserMemory(
        id=1,
        user_id="user-001",
        memory_key=key,
        memory_value=value,
        created_at=now,
        updated_at=now,
    )


def test_nutrition_advice_reads_fitness_goal_memory() -> None:
    """memory_retrieve -> memory_prompt must surface the long-term fitness_goal."""
    memory = _MemoryService([_fact("fitness_goal", "增肌")])
    state = {
        "user_id": "user-001",
        "conversation_id": "",
        "input": "我应该怎么吃",
        "messages": [{"role": "user", "content": "我应该怎么吃"}],
    }

    result = memory_retrieve_node(state, memory_service=memory)
    prompt = build_memory_prompt(result)

    assert result["user_memories"][0]["memory_key"] == "fitness_goal"
    assert "fitness_goal" in prompt
    assert "增肌" in prompt


def test_nutrition_advice_supports_three_fitness_goals() -> None:
    """fitness_goal values 增肌 / 减脂 / 保持 are all injectable into prompts."""
    for goal in ("增肌", "减脂", "保持"):
        prompt = build_memory_prompt(
            {
                "user_memories": [{"memory_key": "fitness_goal", "memory_value": goal}],
                "memory_context": {"long_term": [], "short_term": []},
            }
        )
        assert goal in prompt


# ---------------------------------------------------------------------------
# 3. RAG: nutrition knowledge is retrievable
# ---------------------------------------------------------------------------
class _KeywordEmbedding:
    """Deterministic keyword embedding, no provider or network access."""

    DIMENSIONS = ("增肌", "减脂", "蛋白质", "训练前", "训练后")

    def embed_text(self, text: str) -> list[float]:
        return [float(word in text) for word in self.DIMENSIONS]

    def embed_documents(self, documents):
        return [self.embed_text(document) for document in documents]


class _VectorStore:
    """Tiny keyword-overlap store over real knowledge chunks."""

    def __init__(self, chunks):
        self.chunks = list(chunks)

    def similarity_search(self, query, *, limit: int = 5):
        embed = _KeywordEmbedding()
        scored = []
        for chunk in self.chunks:
            score = sum(
                left * right
                for left, right in zip(query, embed.embed_text(chunk.content), strict=True)
            )
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            VectorSearchResult(chunk=chunk.model_copy(deep=True), score=float(score))
            for score, chunk in scored[:limit]
            if score > 0
        ]


def _nutrition_service() -> KnowledgeRetrievalService:
    documents = DocumentLoader().load_directory(KNOWLEDGE_ROOT / "nutrition")
    chunks = [chunk for doc in documents for chunk in Chunker().chunk(doc)]
    return KnowledgeRetrievalService(
        KnowledgeRetriever(_KeywordEmbedding(), _VectorStore(chunks), top_k=3)
    )


def test_nutrition_knowledge_base_contains_required_documents() -> None:
    names = {path.name for path in (KNOWLEDGE_ROOT / "nutrition").glob("*.md")}
    assert {"muscle_gain.md", "fat_loss.md", "protein.md", "pre_post_workout.md"} <= names


def test_nutrition_knowledge_is_general_scope_not_medical() -> None:
    for path in sorted((KNOWLEDGE_ROOT / "nutrition").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        assert "不构成医疗建议" in text or "专业医疗建议" in text, path.name
        assert "疾病饮食方案" not in text, path.name


def test_rag_retrieves_nutrition_knowledge_for_muscle_gain_question() -> None:
    context = _nutrition_service().retrieve_context("增肌应该怎么吃，蛋白质要多少？")
    assert context, "nutrition RAG must return chunks"
    assert any("增肌" in item["content"] or "蛋白质" in item["content"] for item in context)
    assert all(item["source"].endswith(".md") for item in context)


def test_rag_retrieves_nutrition_knowledge_for_fat_loss_question() -> None:
    context = _nutrition_service().retrieve_context("减脂期间怎么安排饮食")
    assert context
    assert any("fat_loss" in item["source"] or "减脂" in item["content"] for item in context)


def test_rag_retrieves_pre_post_workout_knowledge() -> None:
    context = _nutrition_service().retrieve_context("训练前吃什么比较好，训练后应该补充什么")
    assert context
    assert any("pre_post_workout" in item["source"] for item in context)


# ---------------------------------------------------------------------------
# 5. End-to-end: intent + memory + RAG on the same nutrition request
# ---------------------------------------------------------------------------
class _RecordingLLM:
    def __init__(self):
        self.system_prompts = []

    async def complete(self, *, system_prompt=None, messages=None, **kwargs):
        del messages, kwargs
        self.system_prompts.append(system_prompt or "")
        return "增肌期间以足量蛋白质和适量热量盈余为主，训练前后安排含碳水的正餐。"


class _ProfileRepo:
    def get_profile(self, user_id):
        del user_id
        return None


@pytest.mark.asyncio
async def test_nutrition_question_runs_intent_memory_and_rag_together() -> None:
    """"我的目标是增肌，我应该怎么吃" must flow through the whole pipeline."""
    llm = _RecordingLLM()
    memory = _MemoryService([_fact("fitness_goal", "增肌")])
    graph = create_agent_graph(
        llm,
        _ProfileRepo(),
        memory_service=memory,
        knowledge_service=_nutrition_service(),
    )

    result = await graph.ainvoke(
        {
            "user_id": "user-001",
            "conversation_id": "",
            "input": "我的目标是增肌，我应该怎么吃",
            "conversation_history": [],
            "rag_context": [],
            "tool_calls": [],
            "tool_rounds": 0,
            "tool_call_count": 0,
            "answer_draft": None,
            "stop_reason": "completed",
            "messages": [],
            "tool_messages": [],
            "tool_results": [],
        }
    )

    assert result["intent"] == "nutrition_advice"
    assert result["memory_context"].long_term[0].memory_key == "fitness_goal"
    assert result["memory_context"].long_term[0].memory_value == "增肌"
    assert result["rag_context"], "nutrition RAG context must be present"
    prompt = llm.system_prompts[-1]
    assert "增肌" in prompt
    assert "饮食建议" in prompt


# ---------------------------------------------------------------------------
# 4. Prompt: capability + non-medical disclaimer
# ---------------------------------------------------------------------------
def test_system_prompt_mentions_nutrition_capability_and_disclaimer() -> None:
    prompt = build_system_prompt(None, None)
    assert "饮食建议" in prompt
    assert "训练目标" in prompt
    assert "不替代专业医疗建议" in prompt
    assert "不是营养师" in prompt


def test_nutrition_prompt_does_not_claim_medical_or_prescription_scope() -> None:
    prompt = build_system_prompt(None, None)
    for forbidden in ("诊断", "治疗方案", "处方剂量", "疾病饮食方案"):
        assert forbidden not in prompt
