from __future__ import annotations

import time

from aura.config import Settings
from aura.schemas import Attachment, Modality
from aura.session import SessionStore


async def test_get_or_create_is_idempotent(store: SessionStore) -> None:
    first = await store.get_or_create(None)
    again = await store.get_or_create(first.session_id)
    assert first is again


async def test_unknown_ids_are_created_not_rejected(store: SessionStore) -> None:
    memory = await store.get_or_create("ses_client_supplied")
    assert memory.session_id == "ses_client_supplied"


async def test_expired_sessions_are_evicted() -> None:
    store = SessionStore(Settings(engine="echo", session_ttl_seconds=1))
    memory = await store.get_or_create(None)
    store._touched[memory.session_id] = time.time() - 5
    assert await store.get(memory.session_id) is None


async def test_lru_eviction_bounds_memory() -> None:
    store = SessionStore(Settings(engine="echo", max_sessions=3))
    ids = [(await store.get_or_create(None)).session_id for _ in range(5)]
    assert await store.get(ids[0]) is None
    assert await store.get(ids[-1]) is not None


async def test_attachments_are_popped_exactly_once(store: SessionStore) -> None:
    attachment_id = await store.put_attachment(
        Attachment(kind=Modality.IMAGE, media_type="image/png"), b"bytes"
    )
    assert len(await store.pop_attachments([attachment_id])) == 1
    assert await store.pop_attachments([attachment_id]) == []


async def test_attachment_uri_is_assigned_on_store(store: SessionStore) -> None:
    attachment = Attachment(kind=Modality.IMAGE, media_type="image/png")
    attachment_id = await store.put_attachment(attachment, b"bytes")
    assert attachment.uri == f"/api/attachments/{attachment_id}"


async def test_audio_round_trip(store: SessionStore) -> None:
    audio_id = await store.put_audio(b"RIFF", "audio/wav")
    assert await store.get_audio(audio_id) == (b"RIFF", "audio/wav")


async def test_delete_reports_whether_anything_was_removed(store: SessionStore) -> None:
    memory = await store.get_or_create(None)
    assert await store.delete(memory.session_id) is True
    assert await store.delete(memory.session_id) is False


async def test_summaries_reflect_stored_sessions(store: SessionStore) -> None:
    await store.get_or_create("ses_a")
    await store.get_or_create("ses_b")
    assert {s.session_id for s in await store.summaries()} == {"ses_a", "ses_b"}
