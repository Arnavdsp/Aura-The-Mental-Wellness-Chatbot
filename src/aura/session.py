"""Session and attachment storage.

An in-memory, TTL'd, LRU-bounded store. That is the right default for a wellness
product: conversations are sensitive, so nothing touches disk and everything
expires. The class is deliberately narrow (``get``/``create``/``delete``) so a
Redis or Postgres implementation can replace it without touching callers.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from aura.config import Settings
from aura.logging import get_logger
from aura.memory import ConversationMemory
from aura.schemas import Attachment, SessionSummary

log = get_logger(__name__)


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(12)}"


@dataclass
class StoredAttachment:
    """Upload bytes held only for as long as the conversation needs them."""

    id: str
    attachment: Attachment
    data: bytes
    created_at: float = field(default_factory=time.time)


class SessionStore:
    """TTL + LRU store for conversations and their pending attachments."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: OrderedDict[str, ConversationMemory] = OrderedDict()
        self._touched: dict[str, float] = {}
        self._attachments: OrderedDict[str, StoredAttachment] = OrderedDict()
        self._audio: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self._lock = asyncio.Lock()

    # -- sessions --------------------------------------------------------

    async def get_or_create(self, session_id: str | None) -> ConversationMemory:
        async with self._lock:
            self._evict_expired()
            if session_id and session_id in self._sessions:
                self._sessions.move_to_end(session_id)
                self._touched[session_id] = time.time()
                return self._sessions[session_id]

            identifier = session_id or new_id("ses")
            memory = ConversationMemory(session_id=identifier)
            self._sessions[identifier] = memory
            self._touched[identifier] = time.time()
            self._evict_overflow()
            log.info("created session %s", identifier)
            return memory

    async def get(self, session_id: str) -> ConversationMemory | None:
        async with self._lock:
            self._evict_expired()
            memory = self._sessions.get(session_id)
            if memory is not None:
                self._sessions.move_to_end(session_id)
                self._touched[session_id] = time.time()
            return memory

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            self._touched.pop(session_id, None)
            return self._sessions.pop(session_id, None) is not None

    async def summaries(self) -> list[SessionSummary]:
        async with self._lock:
            self._evict_expired()
            return [_summarise(memory) for memory in self._sessions.values()]

    # -- attachments -----------------------------------------------------

    async def put_attachment(self, attachment: Attachment, data: bytes) -> str:
        async with self._lock:
            identifier = new_id("att")
            attachment.uri = f"/api/attachments/{identifier}"
            self._attachments[identifier] = StoredAttachment(identifier, attachment, data)
            while len(self._attachments) > 500:
                self._attachments.popitem(last=False)
            return identifier

    async def get_attachment(self, attachment_id: str) -> StoredAttachment | None:
        async with self._lock:
            return self._attachments.get(attachment_id)

    async def pop_attachments(self, ids: list[str]) -> list[StoredAttachment]:
        """Claim attachments for a turn; each upload is consumed exactly once."""
        async with self._lock:
            return [
                stored
                for stored in (self._attachments.pop(identifier, None) for identifier in ids)
                if stored is not None
            ]

    # -- generated audio -------------------------------------------------

    async def put_audio(self, data: bytes, media_type: str) -> str:
        async with self._lock:
            identifier = new_id("aud")
            self._audio[identifier] = (data, media_type)
            while len(self._audio) > 200:
                self._audio.popitem(last=False)
            return identifier

    async def get_audio(self, audio_id: str) -> tuple[bytes, str] | None:
        async with self._lock:
            return self._audio.get(audio_id)

    # -- eviction --------------------------------------------------------

    def _evict_expired(self) -> None:
        ttl = self._settings.session_ttl_seconds
        if ttl <= 0:
            return
        cutoff = time.time() - ttl
        stale = [key for key, seen in self._touched.items() if seen < cutoff]
        for key in stale:
            self._sessions.pop(key, None)
            self._touched.pop(key, None)
        if stale:
            log.info("evicted %d expired session(s)", len(stale))

    def _evict_overflow(self) -> None:
        while len(self._sessions) > self._settings.max_sessions:
            key, _ = self._sessions.popitem(last=False)
            self._touched.pop(key, None)


def _summarise(memory: ConversationMemory) -> SessionSummary:
    return SessionSummary(
        session_id=memory.session_id,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        turn_count=len(memory.turns),
        topics=memory.graph.dominant(5),
        mood_trend=memory.mood_trend[-20:],
        highest_risk=memory.highest_risk,
    )
