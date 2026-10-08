import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest


def test_chat_request_rejects_empty_message() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(user_id="user-001", message="")


def test_chat_request_requires_user_id() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="今天应该怎么训练？")