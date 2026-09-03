"""Affect estimation — reading tone from text, and optionally from voice.

Two independent estimators feed one fused signal:

* a lexicon estimator over the transcript, which always runs and costs nothing;
* an optional wav2vec2 speech-emotion classifier over the raw audio, which
  catches the case where the words say "I'm fine" and the voice does not.

Fusion is confidence-weighted, and the speech channel is trusted slightly more
when both are confident, because prosody is harder to mask than word choice.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from aura.logging import get_logger
from aura.schemas import AffectSignal

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

log = get_logger(__name__)

# label -> (valence, arousal); a compact circumplex placement.
EMOTION_SPACE: dict[str, tuple[float, float]] = {
    "joy": (0.8, 0.6),
    "calm": (0.5, 0.15),
    "hopeful": (0.55, 0.45),
    "neutral": (0.0, 0.25),
    "tired": (-0.25, 0.1),
    "sad": (-0.7, 0.25),
    "lonely": (-0.6, 0.3),
    "anxious": (-0.5, 0.8),
    "overwhelmed": (-0.6, 0.85),
    "angry": (-0.65, 0.85),
    "fearful": (-0.7, 0.8),
    "disgust": (-0.6, 0.5),
    "surprise": (0.15, 0.7),
}

_LEXICON: dict[str, tuple[str, float]] = {}


def _register(label: str, weight: float, *words: str) -> None:
    for word in words:
        _LEXICON[word] = (label, weight)


_register("anxious", 1.0, "anxious", "anxiety", "nervous", "worried", "worry",
          "panic", "panicking", "on edge", "restless", "dread", "uneasy")
_register("overwhelmed", 1.2, "overwhelmed", "overwhelming", "too much",
          "drowning", "swamped", "burned out", "burnt out", "burnout",
          "can't keep up", "spread thin")
_register("sad", 1.1, "sad", "down", "depressed", "depression", "crying",
          "tearful", "miserable", "grief", "grieving", "heartbroken", "empty",
          "numb", "low")
_register("lonely", 1.1, "lonely", "alone", "isolated", "no one", "nobody",
          "disconnected", "left out")
_register("angry", 1.0, "angry", "furious", "resentful", "frustrated",
          "irritated", "fed up", "pissed", "rage")
_register("fearful", 1.0, "scared", "afraid", "terrified", "frightened", "fear")
_register("tired", 0.9, "tired", "exhausted", "drained", "no energy", "fatigued",
          "sleepless", "insomnia", "can't sleep", "cannot sleep",
          "barely slept", "wiped out")
_register("hopeful", 1.0, "hopeful", "hope", "optimistic", "looking forward",
          "excited", "motivated", "ready")
_register("joy", 1.1, "happy", "glad", "grateful", "thankful", "proud",
          "relieved", "joyful", "good day", "went well")
_register("calm", 0.9, "calm", "peaceful", "settled", "steady", "grounded",
          "at ease", "better today")

_INTENSIFIERS = re.compile(
    r"\b(?:really|so|very|extremely|completely|totally|absolutely|utterly|"
    r"incredibly|beyond)\b", re.IGNORECASE)
_NEGATORS = re.compile(r"\b(?:not|no longer|never|hardly|barely|isn'?t|aren'?t|don'?t)\b",
                       re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z']+")

# Longest phrases first so "burned out" beats "out".
_PHRASES: list[str] = sorted((k for k in _LEXICON if " " in k), key=len, reverse=True)


def analyse_text(text: str) -> AffectSignal:
    """Estimate affect from a transcript using a weighted emotion lexicon."""
    if not text or not text.strip():
        return AffectSignal(source="none")

    lowered = text.lower()
    scores: dict[str, float] = {}
    hits = 0

    def bump(label: str, weight: float, span_start: int) -> None:
        nonlocal hits
        # Look back only within the current clause: in "never present. I feel
        # alone" the "alone" must not be cancelled by the previous clause's
        # "never", and nor must "empty" in "I am not happy, I feel empty".
        clause_start = max(
            (lowered.rfind(mark, 0, span_start) for mark in ".!?;,\n"), default=-1
        )
        window = lowered[max(clause_start + 1, span_start - 30):span_start]
        if _INTENSIFIERS.search(window):
            weight *= 1.4
        if _NEGATORS.search(window):
            weight *= -0.6
        scores[label] = scores.get(label, 0.0) + weight
        hits += 1

    consumed = lowered
    for phrase in _PHRASES:
        start = consumed.find(phrase)
        while start != -1:
            label, weight = _LEXICON[phrase]
            bump(label, weight, start)
            consumed = consumed[:start] + " " * len(phrase) + consumed[start + len(phrase):]
            start = consumed.find(phrase)

    for match in _WORD_RE.finditer(consumed):
        entry = _LEXICON.get(match.group())
        if entry:
            bump(entry[0], entry[1], match.start())

    positives = {label: value for label, value in scores.items() if value > 0}
    if not positives:
        return AffectSignal(label="neutral", source="text", confidence=0.2 if hits else 0.1)

    total = sum(positives.values())
    normalised = {label: value / total for label, value in positives.items()}
    label = max(normalised, key=normalised.__getitem__)

    valence = sum(EMOTION_SPACE.get(k, (0.0, 0.3))[0] * w for k, w in normalised.items())
    arousal = sum(EMOTION_SPACE.get(k, (0.0, 0.3))[1] * w for k, w in normalised.items())
    # Confidence saturates with evidence rather than growing without bound.
    confidence = min(0.9, 0.35 + 0.18 * math.log1p(total))

    return AffectSignal(
        label=label,
        valence=round(max(-1.0, min(1.0, valence)), 3),
        arousal=round(max(0.0, min(1.0, arousal)), 3),
        confidence=round(confidence, 3),
        source="text",
        scores={k: round(v, 3) for k, v in sorted(
            normalised.items(), key=lambda kv: -kv[1])[:5]},
    )


_SPEECH_LABEL_MAP = {
    "angry": "angry", "calm": "calm", "disgust": "disgust", "fearful": "fearful",
    "happy": "joy", "neutral": "neutral", "sad": "sad", "surprised": "surprise",
    "surprise": "surprise",
}


@lru_cache(maxsize=1)
def _speech_classifier(model_id: str) -> Any | None:
    try:
        from transformers import pipeline  # type: ignore[import-not-found]
    except ImportError:
        log.warning("transformers unavailable; speech emotion disabled")
        return None
    try:
        return pipeline("audio-classification", model=model_id, top_k=None)
    except Exception as exc:  # pragma: no cover - depends on network/weights
        log.warning("could not load speech emotion model %s: %s", model_id, exc)
        return None


def analyse_speech(
    waveform: np.ndarray, sample_rate: int, *, model_id: str
) -> AffectSignal:
    """Estimate affect from prosody. Returns a null signal if unavailable."""
    classifier = _speech_classifier(model_id)
    if classifier is None or waveform is None or len(waveform) == 0:
        return AffectSignal(source="none")
    try:
        raw = classifier({"array": waveform, "sampling_rate": sample_rate})
    except Exception as exc:  # pragma: no cover
        log.warning("speech emotion inference failed: %s", exc)
        return AffectSignal(source="none")

    scores = {
        _SPEECH_LABEL_MAP.get(str(item["label"]).lower(), "neutral"): float(item["score"])
        for item in raw
    }
    if not scores:
        return AffectSignal(source="none")
    label = max(scores, key=scores.__getitem__)
    valence, arousal = EMOTION_SPACE.get(label, (0.0, 0.3))
    return AffectSignal(
        label=label,
        valence=valence,
        arousal=arousal,
        confidence=round(scores[label], 3),
        source="speech",
        scores={k: round(v, 3) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:5]},
    )


def fuse(text_signal: AffectSignal, speech_signal: AffectSignal) -> AffectSignal:
    """Confidence-weighted fusion, with a mild prior toward the voice channel."""
    if speech_signal.source == "none":
        return text_signal
    if text_signal.source == "none":
        return speech_signal

    speech_weight = speech_signal.confidence * 1.15
    text_weight = text_signal.confidence
    total = speech_weight + text_weight
    if total <= 0:
        return text_signal

    speech_share = speech_weight / total
    text_share = text_weight / total
    valence = speech_signal.valence * speech_share + text_signal.valence * text_share
    arousal = speech_signal.arousal * speech_share + text_signal.arousal * text_share
    dominant = speech_signal if speech_share >= text_share else text_signal

    merged: dict[str, float] = dict(text_signal.scores)
    for key, value in speech_signal.scores.items():
        merged[key] = round(merged.get(key, 0.0) * text_share + value * speech_share, 3)

    return AffectSignal(
        label=dominant.label,
        valence=round(valence, 3),
        arousal=round(arousal, 3),
        confidence=round(max(text_signal.confidence, speech_signal.confidence), 3),
        source="fused",
        scores=dict(sorted(merged.items(), key=lambda kv: -kv[1])[:5]),
    )


def describe(signal: AffectSignal) -> str:
    """A short natural-language gloss, used to steer the model's tone."""
    if signal.source == "none" or signal.confidence < 0.25:
        return "unclear"
    intensity = "strongly" if signal.confidence > 0.65 else "somewhat"
    energy = "high-energy" if signal.arousal > 0.6 else (
        "low-energy" if signal.arousal < 0.3 else "moderate-energy")
    return f"{intensity} {signal.label} ({energy})"
