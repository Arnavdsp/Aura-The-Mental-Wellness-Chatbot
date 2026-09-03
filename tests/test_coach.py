"""Orchestration: prompt assembly, memory commits, and modality handling."""

from __future__ import annotations

import pytest

from aura.coach import Coach, _postprocess
from aura.prompts import build_system_prompt, suggestions_for
from aura.schemas import AffectSignal, ChatRequest, RiskLevel


@pytest.fixture
def coach(settings, engine, store) -> Coach:
    return Coach(settings, engine, store)


async def test_prepare_builds_a_prompt_carrying_affect(coach) -> None:
    prepared = await coach.prepare(ChatRequest(message="I'm completely overwhelmed"))
    assert "overwhelmed" in prepared.generation.system_prompt
    assert prepared.affect.label == "overwhelmed"
    assert prepared.short_circuit is None


async def test_history_grows_and_stays_bounded(coach, settings) -> None:
    session_id = None
    for index in range(settings.max_turns_in_context + 6):
        request = ChatRequest(session_id=session_id, message=f"message number {index}")
        prepared, _, _ = await coach.respond(request)
        session_id = prepared.memory.session_id
    prepared = await coach.prepare(ChatRequest(session_id=session_id, message="one more"))
    assert len(prepared.generation.history) <= settings.max_turns_in_context


async def test_both_turns_are_committed_to_memory(coach) -> None:
    prepared, reply, _ = await coach.respond(ChatRequest(message="I feel low today"))
    roles = [turn.role.value for turn in prepared.memory.turns]
    assert roles == ["user", "assistant"]
    assert prepared.memory.turns[-1].id == reply.id
    assert reply.latency_ms is not None


async def test_crisis_short_circuits_before_the_engine_runs(coach) -> None:
    prepared, reply, _ = await coach.respond(ChatRequest(message="I want to die"))
    assert prepared.safety.risk is RiskLevel.CRISIS
    assert reply.meta["short_circuited"] is True
    assert "helpline" in reply.text.lower() or "988" in reply.text


async def test_streaming_and_buffered_paths_agree_on_metadata(coach) -> None:
    events = [event async for event in coach.stream(ChatRequest(message="I'm anxious"))]
    names = [event["event"] for event in events]
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert events[0]["data"]["affect"]["label"] == "anxious"


async def test_image_only_turn_is_accepted(coach, store, png_bytes) -> None:
    from aura.schemas import Attachment, Modality

    attachment_id = await store.put_attachment(
        Attachment(kind=Modality.IMAGE, media_type="image/png"), png_bytes
    )
    prepared = await coach.prepare(
        ChatRequest(message="", attachment_ids=[attachment_id])
    )
    assert prepared.generation.images
    assert "image" in prepared.generation.system_prompt.lower()


async def test_disabled_vision_is_reported_as_a_notice(settings, engine, store, png_bytes) -> None:
    from aura.schemas import Attachment, Modality

    settings.enable_vision = False
    coach = Coach(settings, engine, store)
    attachment_id = await store.put_attachment(
        Attachment(kind=Modality.IMAGE, media_type="image/png"), png_bytes
    )
    prepared = await coach.prepare(ChatRequest(message="hi", attachment_ids=[attachment_id]))
    assert not prepared.generation.images
    assert any("disabled" in notice for notice in prepared.notices)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<start_of_turn>model\nHello there<end_of_turn>", "Hello there"),
        ("Aura: I'm listening", "I'm listening"),
        ("   ", "I'm here, but I lost my words for a second. Could you say that again?"),
    ],
)
def test_postprocess_strips_template_leakage(raw: str, expected: str) -> None:
    assert _postprocess(raw) == expected


def test_system_prompt_states_the_coaching_stance() -> None:
    prompt = build_system_prompt(None, AffectSignal())
    assert "not a therapist" in prompt
    assert "one open question" in prompt


def test_system_prompt_escalates_tone_on_elevated_risk() -> None:
    prompt = build_system_prompt(None, AffectSignal(), risk=RiskLevel.ELEVATED)
    assert "significant distress" in prompt


def test_suggestions_change_with_risk_and_affect() -> None:
    crisis = suggestions_for(AffectSignal(label="sad"), RiskLevel.CRISIS)
    assert any("someone" in item for item in crisis)
    assert suggestions_for(AffectSignal(label="joy"), RiskLevel.NONE) != crisis
