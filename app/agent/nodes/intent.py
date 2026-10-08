import re

from app.agent.state import AgentState, Intent

# Explicit diet-planning phrasing -> nutrition_advice (checked before the
# generic nutrition bucket so "晚餐应该吃什么？" keeps its existing intent).
_NUTRITION_ADVICE_PATTERN = re.compile(
    r"应该怎么吃|怎么吃|怎么安排饮食|饮食安排|饮食计划|吃什么好|吃什么比较好|"
    r"吃什么才|训练前.{0,8}吃|训练后.{0,8}(?:吃|补充)|补充什么|该补充|吃多少|"
    r"饮食.{0,6}(?:建议|搭配|结构)",
    re.I,
)

_INTENT_PATTERNS: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    (
        "training",
        re.compile(
            r"训练|锻炼|健身|增肌|减脂|力量|有氧|跑步|卧推|深蹲|硬拉|动作|课程",
            re.I,
        ),
    ),
    ("nutrition", re.compile(r"饮食|吃什么|营养|蛋白质|热量|餐|食谱|补剂|喝水|碳水|脂肪", re.I)),
    ("recovery", re.compile(r"睡眠|恢复|休息|疲劳|酸痛|疼痛|受伤|拉伸|放松|压力", re.I)),
    ("lifestyle", re.compile(r"习惯|日程|安排|时间|打卡|目标|生活|坚持|计划表", re.I)),
)


def classify_intent(message: str) -> Intent:
    """Classify user intent with deterministic Phase 1 heuristics.

    This node deliberately does not call an LLM. A future Intent model can replace
    this function without changing the graph topology or State contract.
    """
    normalized = message.strip()
    if _NUTRITION_ADVICE_PATTERN.search(normalized):
        return "nutrition_advice"
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(normalized):
            return intent
    return "general"


def intent_node(state: AgentState) -> dict[str, object]:
    """Normalize the incoming message and classify its business intent."""
    message = state["input"].strip()
    history = [
        *state.get("messages", state.get("conversation_history", [])),
        {"role": "user", "content": message},
    ]
    return {
        "intent": classify_intent(message),
        "messages": history,
        "conversation_history": history,
    }
