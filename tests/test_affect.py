from __future__ import annotations

import pytest

from aura.affect import analyse_text, describe, fuse
from aura.schemas import AffectSignal


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I'm completely overwhelmed, it's all too much", "overwhelmed"),
        ("I'm so anxious I can't stop worrying", "anxious"),
        ("I feel sad and empty lately", "sad"),
        ("I'm exhausted and can't sleep", "tired"),
        ("I'm grateful and proud of how today went", "joy"),
        ("I feel lonely, like nobody sees me", "lonely"),
    ],
)
def test_dominant_label(text: str, expected: str) -> None:
    assert analyse_text(text).label == expected


def test_valence_sign_tracks_sentiment() -> None:
    assert analyse_text("I feel hopeless and alone").valence < 0
    assert analyse_text("I'm happy and grateful today").valence > 0


def test_arousal_separates_panic_from_flatness() -> None:
    panicky = analyse_text("I'm panicking, my heart is racing, so anxious")
    flat = analyse_text("I'm just tired and drained")
    assert panicky.arousal > flat.arousal


def test_intensifiers_raise_confidence() -> None:
    mild = analyse_text("I'm anxious")
    strong = analyse_text("I'm really extremely anxious")
    assert strong.confidence > mild.confidence


def test_negation_suppresses_the_negated_feeling() -> None:
    """"not anxious" must not be read as anxious."""
    assert analyse_text("I'm not anxious at all today").label != "anxious"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A negation in an earlier clause must not cancel a later feeling word.
        ("My partner says I am never present any more. I feel alone in it.", "lonely"),
        ("I am not happy, I feel empty", "sad"),
        ("I'm not sad, just tired", "tired"),
    ],
)
def test_negation_does_not_leak_across_clauses(text: str, expected: str) -> None:
    assert analyse_text(text).label == expected


def test_empty_and_neutral_text_yield_low_confidence() -> None:
    assert analyse_text("").source == "none"
    assert analyse_text("The meeting is at three o'clock").confidence < 0.3


def test_fusion_prefers_the_more_confident_channel() -> None:
    text_signal = AffectSignal(label="calm", valence=0.5, arousal=0.2,
                               confidence=0.3, source="text")
    speech_signal = AffectSignal(label="sad", valence=-0.7, arousal=0.25,
                                 confidence=0.9, source="speech")
    fused = fuse(text_signal, speech_signal)
    assert fused.source == "fused"
    assert fused.label == "sad"          # the voice wins when it is far surer
    assert fused.valence < 0


def test_fusion_degrades_to_whichever_channel_exists() -> None:
    text_signal = analyse_text("I feel low")
    assert fuse(text_signal, AffectSignal(source="none")) is text_signal
    assert fuse(AffectSignal(source="none"), text_signal) is text_signal


def test_describe_is_honest_about_uncertainty() -> None:
    assert describe(AffectSignal(source="none")) == "unclear"
    assert describe(analyse_text("I am extremely overwhelmed")) != "unclear"
