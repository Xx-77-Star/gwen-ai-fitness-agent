from app.agent.state import AgentState
from app.memory.profile_repository import ProfileRepository


def load_profile_node(
    state: AgentState,
    *,
    profile_repository: ProfileRepository,
) -> dict[str, object]:
    """Load a user profile into Agent State before intent classification."""
    user_id = state.get("user_id", "").strip()
    profile = profile_repository.get_profile(user_id) if user_id else None
    return {
        "profile": profile,
        "profile_loaded": profile is not None,
    }
