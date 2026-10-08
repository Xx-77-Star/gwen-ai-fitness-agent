from typing import Annotated, Any, Literal, TypedDict

from app.memory.models import MemoryContext, MemoryTurn
from app.schemas.user import UserProfileResponse

Intent = Literal[
    "training",
    "nutrition",
    "nutrition_advice",
    "recovery",
    "lifestyle",
    "general",
]
FitnessLevel = Literal["beginner", "intermediate", "advanced", "unknown"]
ToolStatus = Literal["success", "error", "pending", "denied"]
StopReason = Literal["completed", "tool_call", "tool_limit", "error", "cancelled", "unknown"]


class UserContext(TypedDict, total=False):
    """User profile fields that later Memory/User modules can populate."""

    user_id: str
    display_name: str
    fitness_level: FitnessLevel
    goals: list[str]
    preferences: dict[str, Any]


class ConversationTurn(TypedDict, total=False):
    """A transport-neutral conversation item used by the workflow and LLM gateway."""

    role: Literal["user", "assistant", "system"]
    content: str


class RAGContext(TypedDict, total=False):
    """Source-aware knowledge context retrieved for one user query."""

    source: str
    content: str
    score: float


class ToolCall(TypedDict, total=False):
    """Tool call emitted by the LLM before local execution."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]


class ToolResult(TypedDict, total=False):
    """Result of one tool call after local execution."""

    call_id: str
    tool_name: str
    status: ToolStatus
    result: Any
    error: dict[str, Any] | str | None


class AgentMessage(TypedDict, total=False):
    """Tool-aware message record used by the Tool Calling lifecycle.

    ``tool_calls`` is populated on assistant messages and ``tool_call_id``
    correlates a tool result with the originating assistant call. The older
    ``content`` contract remains supported for normal conversation turns.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: Any
    name: str | None
    tool_calls: list[ToolCall]
    tool_call_id: str


def append_list(left: list[Any], right: list[Any]) -> list[Any]:
    """Reducer for lifecycle lists that should retain every accumulated item."""
    return [*left, *right]


def replace_list(left: list[Any], right: list[Any]) -> list[Any]:
    """Reducer for messages whose latest explicit value replaces history."""
    return right if right is not None else left


class AgentState(TypedDict, total=False):
    """State shared by every node in the LangGraph workflow."""

    input: str
    user_id: str
    user: UserContext
    profile: UserProfileResponse | None
    profile_loaded: bool
    fitness_level: FitnessLevel
    conversation_id: str
    conversation_history: list[MemoryTurn]
    messages: Annotated[list[MemoryTurn], replace_list]
    rag_context: Annotated[list[RAGContext], replace_list]
    tool_calls: Annotated[list[ToolCall], replace_list]
    tool_results: Annotated[list[ToolResult], append_list]
    tool_rounds: int
    tool_call_count: int
    answer_draft: str | None
    stop_reason: StopReason
    tool_messages: Annotated[list[AgentMessage], replace_list]
    memory_context: MemoryContext | None
    memory_loaded: bool
    memory_turns: list[MemoryTurn]
    user_memories: list[dict[str, Any]]
    memory_extractions: list[dict[str, Any]]
    intent: Intent
    response: str
    metadata: dict[str, Any]
