"""The safety screen is the highest-stakes code here, so it gets the most tests."""

from __future__ import annotations

import pytest

from aura.safety import crisis_message, resources_for, screen
from aura.schemas import RiskLevel


@pytest.mark.parametrize(
    "text",
    [
        "I want to die",
        "I've been thinking about killing myself",
        "I don't want to be here anymore",
        "Everyone would be better off without me",
        "I have a suicide plan",
        "I've been cutting myself again",
        "I took an overdose last night",
    ],
)
def test_crisis_language_escalates(text: str) -> None:
    assessment = screen(text)
    assert assessment.risk is RiskLevel.CRISIS
    assert assessment.should_short_circuit
    assert assessment.resources


@pytest.mark.parametrize(
    "text",
    [
        "I feel completely hopeless about my career",
        "I can't take this anymore, work is relentless",
        "My partner hits me when he drinks",
    ],
)
def test_elevated_language_surfaces_resources_without_short_circuiting(text: str) -> None:
    assessment = screen(text)
    assert assessment.risk is RiskLevel.ELEVATED
    assert assessment.resources
    assert not assessment.should_short_circuit


@pytest.mark.parametrize(
    "text",
    [
        "I had a good day today",
        "Work is busy but manageable",
        "I'm going to kill this presentation tomorrow",
        "This deadline is murder on my schedule",
        "",
    ],
)
def test_ordinary_language_is_not_flagged(text: str) -> None:
    assert screen(text).risk is RiskLevel.NONE


@pytest.mark.parametrize(
    "text",
    [
        "I used to want to die but I'm past that now",
        "Years ago I thought about killing myself; I've recovered from that",
        "My friend said he wanted to die and I didn't know what to say",
    ],
)
def test_past_and_third_party_accounts_are_softened(text: str) -> None:
    """Recounting recovery must not trigger the full crisis interrupt."""
    assessment = screen(text)
    assert assessment.risk is RiskLevel.ELEVATED
    assert not assessment.should_short_circuit
    assert "softened" in assessment.rationale


def test_screen_can_be_disabled() -> None:
    assert screen("I want to die", enabled=False).risk is RiskLevel.NONE


def test_regional_resources_always_include_international_fallback() -> None:
    india = resources_for("IN")
    assert any(r.region == "IN" for r in india)
    assert any(r.region == "INTL" for r in india)
    assert resources_for("ZZ") == resources_for("INTL")


def test_crisis_message_leads_with_contact_details() -> None:
    message = crisis_message(screen("I want to end my life", region="US"))
    assert "988" in message
    assert "emergency services" in message.lower()
    # It must still acknowledge the person, not just dump a phone number.
    assert "glad you told me" in message.lower()
