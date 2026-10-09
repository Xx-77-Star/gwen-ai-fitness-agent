from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.database.training_repository import TrainingRecordRepository
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition

MAX_RECORD_LIMIT = 50


class LastTrainingInput(BaseModel):
    """Input for get_last_training_record."""

    model_config = ConfigDict(extra="forbid")


class ListTrainingInput(BaseModel):
    """Input for list_training_records."""

    model_config = ConfigDict(extra="forbid")

    start_date: date
    end_date: date
    limit: int = Field(default=20, ge=1, le=MAX_RECORD_LIMIT)


class SummaryInput(BaseModel):
    """Input for get_training_summary."""

    model_config = ConfigDict(extra="forbid")

    start_date: date
    end_date: date


def _validate_range(start_date: date, end_date: date) -> None:
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")


def get_last_training_record(
    arguments: dict[str, Any],
    user_id: str,
    repository: TrainingRecordRepository,
) -> dict[str, Any]:
    record = repository.get_latest_by_user(user_id)
    return {"record": record.model_dump(mode="json") if record else None}


def list_training_records(
    arguments: dict[str, Any],
    user_id: str,
    repository: TrainingRecordRepository,
) -> dict[str, Any]:
    payload = ListTrainingInput.model_validate(arguments)
    _validate_range(payload.start_date, payload.end_date)
    records = repository.list_by_user(
        user_id,
        start_date=payload.start_date,
        end_date=payload.end_date,
        limit=payload.limit,
    )
    return {
        "records": [record.model_dump(mode="json") for record in records],
        "returned_count": len(records),
        "start_date": payload.start_date.isoformat(),
        "end_date": payload.end_date.isoformat(),
    }


def get_training_summary(
    arguments: dict[str, Any],
    user_id: str,
    repository: TrainingRecordRepository,
) -> dict[str, Any]:
    payload = SummaryInput.model_validate(arguments)
    _validate_range(payload.start_date, payload.end_date)
    summary = repository.summarize_by_user(
        user_id,
        start_date=payload.start_date,
        end_date=payload.end_date,
    )
    return {"summary": summary.model_dump(mode="json")}


def build_training_tool_registry() -> ToolRegistry:
    """Create a registry containing the supported read-only training tools."""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_last_training_record",
            description="只用于用户明确询问最近一次单独训练记录，例如我上一次练了什么、最后一次训练是哪天、最近一次训练详情。不要用于训练趋势、训练总结或训练记录列表。",
            input_model=LastTrainingInput,
            handler=get_last_training_record,
        )
    )
    registry.register(
        ToolDefinition(
            name="list_training_records",
            description="用于用户明确要求查看训练记录列表或某段时间内的训练记录明细，例如列出我的训练记录、看看这周有哪些训练。不要用于只看最近一次或统计趋势。",
            input_model=ListTrainingInput,
            handler=list_training_records,
        )
    )
    registry.register(
        ToolDefinition(
            name="get_training_summary",
            description="优先用于最近训练情况、最近练得怎么样、训练总结、训练趋势、总训练次数、总训练时长或训练类型分布。返回指定日期范围内的训练次数、训练天数、总训练时长和类型分布。不要用于只查看最近一次单独训练或列出记录明细。",
            input_model=SummaryInput,
            handler=get_training_summary,
        )
    )
    return registry
