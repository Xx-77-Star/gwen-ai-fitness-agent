import pytest

from app.tools.registry import ToolRegistry
from app.tools.training_tools import build_training_tool_registry


@pytest.fixture
def training_tools() -> ToolRegistry:
    return build_training_tool_registry()


@pytest.mark.parametrize(
    ("question", "expected_tool"),
    [
        ("查询我最近训练情况", "get_training_summary"),
        ("我上一次训练是什么", "get_last_training_record"),
        ("列出我的训练记录", "list_training_records"),
    ],
)
def test_training_tool_descriptions_prefer_expected_tool(
    training_tools: ToolRegistry,
    question: str,
    expected_tool: str,
) -> None:
    registry = training_tools
    payloads = registry.schema_payloads()
    assert expected_tool in registry.names()

    function_descriptions = {
        payload["function"]["name"]: payload["function"]["description"] for payload in payloads
    }
    if expected_tool == "get_training_summary":
        assert "最近训练情况" in function_descriptions[expected_tool]
        assert "最近练得怎么样" in function_descriptions[expected_tool]
        assert "训练总结" in function_descriptions[expected_tool]
        assert "总训练次数" in function_descriptions[expected_tool]
        assert "总训练时长" in function_descriptions[expected_tool]
        assert "类型分布" in function_descriptions[expected_tool]
        assert "不要用于只查看最近一次" in function_descriptions[expected_tool]
    elif expected_tool == "get_last_training_record":
        assert "最近一次单独训练记录" in function_descriptions[expected_tool]
        assert "我上一次练了什么" in function_descriptions[expected_tool]
        assert "最后一次训练是哪天" in function_descriptions[expected_tool]
        assert "不要用于训练趋势" in function_descriptions[expected_tool]
    else:
        assert "训练记录列表" in function_descriptions[expected_tool]
        assert "训练记录明细" in function_descriptions[expected_tool]
        assert "不要用于只看最近一次" in function_descriptions[expected_tool]


def test_tool_registry_contains_only_the_three_training_tools() -> None:
    registry = build_training_tool_registry()

    assert registry.names() == (
        "get_last_training_record",
        "list_training_records",
        "get_training_summary",
    )
