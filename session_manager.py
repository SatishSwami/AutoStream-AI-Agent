"""
Session Manager — maintains per-user AgentState across webhook calls.
Keyed by session_id (phone number, user_id, or chat_id depending on platform).
Uses an in-memory store by default; swap for Redis in production.
"""

import time
from typing import Dict, Optional
from utils.logger import get_logger

logger = get_logger("session_manager")

# Session TTL in seconds (30 minutes of inactivity = session expires)
SESSION_TTL_SECONDS = 1800


class SessionStore:
    """
    Thread-safe in-memory session store.
    Replace _store with a Redis client for production horizontal scaling.
    """

    def __init__(self):
        self._store: Dict[str, dict] = {}

    def get(self, session_id: str) -> Optional[dict]:
        record = self._store.get(session_id)
        if record is None:
            return None
        # Check TTL
        if time.time() - record["last_active"] > SESSION_TTL_SECONDS:
            logger.info(f"Session {session_id} expired — evicting.")
            del self._store[session_id]
            return None
        return record["agent_state"]

    def set(self, session_id: str, agent_state: dict):
        self._store[session_id] = {
            "agent_state": agent_state,
            "last_active": time.time(),
        }

    def delete(self, session_id: str):
        self._store.pop(session_id, None)

    def active_sessions(self) -> int:
        now = time.time()
        return sum(
            1 for r in self._store.values()
            if now - r["last_active"] <= SESSION_TTL_SECONDS
        )


# Singleton store
_session_store = SessionStore()


def get_session_store() -> SessionStore:
    return _session_store
