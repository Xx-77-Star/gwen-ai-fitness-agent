from functools import partial

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.agent.nodes.conversation_persistence import conversation_persist_node
from app.agent.nodes.intent import intent_node
from app.agent.nodes.memory_retrieval import memory_retrieve_node
from app.agent.nodes.memory_update import memory_update_node
from app.agent.nodes.profile_loader import load_profile_node
from app.agent.nodes.rag_retrieve import rag_retrieve_node
from app.agent.nodes.response import response_node
from app.agent.nodes.tool_decision import tool_decision_node
from app.agent.nodes.tool_execution import tool_execution_node
from app.agent.state import AgentState
from app.config.settings import get_settings
from app.database.session import get_session_factory
from app.database.training_repository import SQLTrainingRecordRepository
from app.database.workout_repository import SQLWorkoutCheckInRepository
from app.llm.client import LLMClient, OpenAICompatibleLLMClient
from app.memory.conversation_repository import ConversationRepository, SQLConversationRepository
from app.memory.interface import MemoryServiceInterface
from app.memory.profile_repository import ProfileRepository, SQLProfileRepository
from app.memory.repository import SQLUserMemoryRepository
from app.memory.service import MemoryService
from app.rag.retriever import KnowledgeRetriever
from app.rag.service import KnowledgeRetrievalService
from app.tools.executor import ToolExecutor
from app.tools.geocoding_tools import OpenMeteoGeocodingClient, register_geocoding_tool
from app.tools.recommendation_tools import register_recommendation_tool
from app.tools.registry import ToolRegistry
from app.tools.time_tools import register_time_tool
from app.tools.training_tools import build_training_tool_registry
from app.tools.weather_tools import OpenMeteoWeatherClient, register_weather_tool
from app.tools.workout_tools import register_workout_tools

MAX_TOOL_ROUNDS = 2
MAX_TOTAL_TOOL_CALLS = 4
DEFAULT_CONVERSATION_WINDOW = 20


def create_agent_graph(
    llm_client: LLMClient,
    profile_repository: ProfileRepository,
    *,
    tool_registry: ToolRegistry | None = None,
    training_repository: object | None = None,
    memory_service: MemoryServiceInterface | None = None,
    conversation_repository: ConversationRepository | None = None,
    weather_client: object | None = None,
    geocoding_client: object | None = None,
    time_provider: object | None = None,
    workout_repository: object | None = None,
    max_tool_calls: int = MAX_TOTAL_TOOL_CALLS,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    conversation_window: int = DEFAULT_CONVERSATION_WINDOW,
    knowledge_service: KnowledgeRetrievalService | None = None,
) -> object:
    """Build the request-scoped LangGraph workflow with bounded tool and memory loops."""
    workflow = StateGraph(AgentState)
    registry = tool_registry or _build_default_tool_registry()
    weather_client = weather_client or OpenMeteoWeatherClient()
    geocoding_client = geocoding_client or OpenMeteoGeocodingClient()
    tool_executor = ToolExecutor(registry)
    memory = memory_service or MemoryService(_NullMemoryRepository())

    workflow.add_node(
        "load_profile",
        partial(load_profile_node, profile_repository=profile_repository),
    )
    workflow.add_node(
        "memory_retrieve",
        partial(
            memory_retrieve_node,
            memory_service=memory,
            conversation_repository=conversation_repository,
            conversation_window=conversation_window,
        ),
    )
    workflow.add_node("intent", intent_node)
    workflow.add_node(
        "rag_retrieve",
        partial(
            rag_retrieve_node,
            knowledge_service=knowledge_service or KnowledgeRetrievalService(
                KnowledgeRetriever(_NullEmbeddingProvider(), _EmptyVectorStore())
            ),
        ),
    )
    workflow.add_node(
        "tool_decision",
        partial(tool_decision_node, llm_client=llm_client, tool_registry=registry),
    )
    workflow.add_node(
        "tool_execution",
        partial(
            _execute_tools_with_state_user_id,
            tool_executor=tool_executor,
            tool_registry=tool_registry or registry,
            repository=training_repository,
            weather_client=weather_client,
            geocoding_client=geocoding_client,
            time_provider=time_provider,
            workout_repository=workout_repository,
            max_tool_calls=max_tool_calls,
        ),
    )
    workflow.add_node("response", partial(response_node, llm_client=llm_client))
    workflow.add_node("memory_update", partial(memory_update_node, memory_service=memory))
    workflow.add_node(
        "conversation_persistence",
        partial(
            conversation_persist_node,
            conversation_repository=conversation_repository or _NullConversationRepository(),
        ),
    )

    workflow.add_edge(START, "load_profile")
    workflow.add_edge("load_profile", "memory_retrieve")
    workflow.add_edge("memory_retrieve", "rag_retrieve")
    workflow.add_edge("rag_retrieve", "intent")
    workflow.add_edge("intent", "tool_decision")
    workflow.add_conditional_edges(
        "tool_decision",
        route_tool_decision,
        {"tool_execution": "tool_execution", "response": "response"},
    )
    workflow.add_conditional_edges(
        "tool_execution",
        make_tool_execution_router(max_tool_rounds, max_tool_calls),
        {"tool_decision": "tool_decision", "response": "response"},
    )
    workflow.add_edge("response", "memory_update")
    workflow.add_edge("memory_update", "conversation_persistence")
    workflow.add_edge("conversation_persistence", END)
    return workflow.compile()


def route_tool_decision(state: AgentState) -> str:
    """Route one decision to execution or direct response generation."""
    return "tool_execution" if state.get("tool_calls") else "response"


def make_tool_execution_router(max_tool_rounds: int, max_total_tool_calls: int):
    """Build a router enforcing bounded tool rounds and total calls."""

    def route_tool_execution(state: AgentState) -> str:
        if state.get("stop_reason") == "tool_limit":
            return "response"
        if int(state.get("tool_rounds", 0)) >= max_tool_rounds:
            return "response"
        if int(state.get("tool_call_count", 0)) >= max_total_tool_calls:
            return "response"
        return "tool_decision"

    return route_tool_execution


def _execute_tools_with_state_user_id(
    state: AgentState,
    *,
    tool_executor: ToolExecutor,
    tool_registry: ToolRegistry,
    repository: object,
    weather_client: object,
    geocoding_client: object,
    time_provider: object | None,
    workout_repository: object | None,
    max_tool_calls: int,
) -> object:
    return tool_execution_node(
        state,
        tool_executor=tool_executor,
        tool_registry=tool_registry,
        runtime_context={
            "user_id": state.get("user_id", ""),
            "repository": repository,
            "weather_client": weather_client,
            "geocoding_client": geocoding_client,
            "time_provider": time_provider,
            "workout_repository": workout_repository,
        },
        max_tool_calls=max_tool_calls,
    )


def get_agent_graph_with_repository(
    session: Session,
    *,
    conversation_repository: ConversationRepository | None = None,
) -> object:
    """Build a fresh graph for one request using its SQLAlchemy session."""
    settings = get_settings()
    return create_agent_graph(
        llm_client=OpenAICompatibleLLMClient(settings),
        profile_repository=SQLProfileRepository(session),
        memory_service=MemoryService(SQLUserMemoryRepository(session)),
        conversation_repository=conversation_repository or SQLConversationRepository(session),
        training_repository=SQLTrainingRecordRepository(session),
        workout_repository=SQLWorkoutCheckInRepository(session),
    )


def get_agent_graph() -> object:
    """Build a fresh graph without a cached shared database session."""
    session = get_session_factory()()
    return get_agent_graph_with_repository(session)


def _build_default_tool_registry() -> ToolRegistry:
    registry = build_training_tool_registry()
    register_weather_tool(registry, OpenMeteoWeatherClient())
    register_geocoding_tool(registry, OpenMeteoGeocodingClient())
    register_recommendation_tool(registry)
    register_time_tool(registry)
    register_workout_tools(registry)
    return registry


class _NullEmbeddingProvider:
    """Default no-op embeddings for graphs without configured knowledge retrieval."""

    def embed_text(self, text: str) -> list[float]:
        del text
        return []

    def embed_documents(self, documents):
        del documents
        return []


class _EmptyVectorStore:
    """Default empty vector store for graphs without a knowledge index."""

    def similarity_search(self, query, *, limit=5):
        del query, limit
        return []


class _NullMemoryRepository:
    """Default no-op repository for graphs built without persistent memory."""

    def list_by_user(self, user_id: str):
        return []

    def upsert(self, user_id: str, memory_key: str, memory_value: str):
        raise RuntimeError("Memory persistence is not configured")


class _NullConversationRepository:
    """Default no-op conversation repository for direct graph calls."""

    def create_conversation(self, user_id: str, conversation_id: str | None = None) -> str:
        del user_id, conversation_id
        raise RuntimeError("Conversation persistence is not configured")

    def append_turn(self, conversation_id: str, user_id: str, role, content):
        del conversation_id, user_id, role, content
        raise RuntimeError("Conversation persistence is not configured")

    def list_recent_turns(self, conversation_id: str, user_id: str, *, limit: int):
        del conversation_id, user_id, limit
        return []
