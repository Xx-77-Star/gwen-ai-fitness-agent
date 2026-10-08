"""Live Agent node-event bus for the user-facing execution status card.

This module sits in the observability layer only. It never touches Agent,
LangGraph, Memory, RAG or database logic: it just relays node start/end
lifecycle events to one SSE stream per chat run so the frontend can show real
workflow stages instead of a fixed timer.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import anyio


@dataclass(frozen=True)
class NodeEvent:
    """One node lifecycle event streamed to the execution status card."""

    node_name: str
    phase: str  # "start" | "end"
    error: bool = False

    def payload(self) -> dict[str, Any]:
        return {"node_name": self.node_name, "phase": self.phase, "error": self.error}


class _Subscription:
    """One SSE listener for one chat run."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[NodeEvent | None] = asyncio.Queue()
        self.closed = False

    def push(self, event: NodeEvent | None) -> None:
        if self.closed:
            return
        with contextlib.suppress(asyncio.QueueFull):  # bounded by maxsize
            self.queue.put_nowait(event)

    def close(self) -> None:
        # Enqueue the terminator BEFORE marking closed: push() ignores events
        # once closed, so the ordering matters or the consumer never wakes up.
        self.push(None)
        self.closed = True


class AgentRunBus:
    """In-process pub/sub for node lifecycle events, keyed by conversation_id."""

    def __init__(self) -> None:
        self._subscriptions: dict[str, set[_Subscription]] = defaultdict(set)
        self._buffers: dict[str, list[NodeEvent]] = defaultdict(list)
        self._owners: dict[str, str] = {}
        self._finished: set[str] = set()
        self._lock = anyio.Lock()

    async def register(self, conversation_id: str, user_id: str) -> None:
        """Register the run owner and prepare a replay buffer."""
        async with self._lock:
            self._owners[conversation_id] = user_id
            self._buffers.setdefault(conversation_id, [])

    async def owner(self, conversation_id: str) -> str | None:
        async with self._lock:
            return self._owners.get(conversation_id)

    async def publish(self, conversation_id: str, event: NodeEvent) -> None:
        async with self._lock:
            self._buffers[conversation_id].append(event)
            subscribers = list(self._subscriptions.get(conversation_id, ()))
        for subscriber in subscribers:
            subscriber.push(event)

    async def finish(self, conversation_id: str) -> None:
        async with self._lock:
            self._finished.add(conversation_id)
            subscribers = list(self._subscriptions.get(conversation_id, ()))
        for subscriber in subscribers:
            subscriber.close()

    async def subscribe(self, conversation_id: str) -> _Subscription:
        """Return a subscription that replays buffered events then streams live."""
        subscription = _Subscription()
        async with self._lock:
            self._subscriptions[conversation_id].add(subscription)
            replay = list(self._buffers.get(conversation_id, ()))
            finished = conversation_id in self._finished
        for event in replay:
            subscription.push(event)
        if finished:
            subscription.close()
        return subscription

    async def unsubscribe(self, conversation_id: str, subscription: _Subscription) -> None:
        subscription.close()
        async with self._lock:
            self._subscriptions[conversation_id].discard(subscription)
            if not self._subscriptions[conversation_id]:
                self._subscriptions.pop(conversation_id, None)
                self._buffers.pop(conversation_id, None)
                self._owners.pop(conversation_id, None)
                self._finished.discard(conversation_id)


_RUN_BUS = AgentRunBus()


def get_agent_run_bus() -> AgentRunBus:
    """Process-wide bus used by the chat route and its SSE stream."""
    return _RUN_BUS
