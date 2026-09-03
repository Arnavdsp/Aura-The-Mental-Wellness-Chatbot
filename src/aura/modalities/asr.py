"""Speech to text.

Preference order matches the project's intent: use Gemma 3n's own audio encoder
when it is available (one model for everything), fall back to Whisper when it is
not, and degrade to a clear error rather than silently dropping the user's words.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from aura.config import Settings
from aura.engine.base import CoachEngine
from aura.logging import get_logger
from aura.modalities.audio_io import AudioDecodeError, decode, duration_seconds

log = get_logger(__name__)


class TranscriptionError(RuntimeError):
    pass


@dataclass
class Transcription:
    text: str
    backend: str
    duration_seconds: float = 0.0


@lru_cache(maxsize=2)
def _whisper(model_id: str) -> Any | None:
    try:
        from transformers import pipeline  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return pipeline("automatic-speech-recognition", model=model_id)
    except Exception as exc:  # pragma: no cover - depends on weights
        log.warning("failed to load ASR model %s: %s", model_id, exc)
        return None


class Transcriber:
    """Turns uploaded audio into text, and keeps the waveform for affect analysis."""

    def __init__(self, settings: Settings, engine: CoachEngine) -> None:
        self._settings = settings
        self._engine = engine

    @property
    def available(self) -> bool:
        if not self._settings.enable_audio_input:
            return False
        if self._settings.asr_backend == "none":
            return False
        if self._settings.asr_backend == "gemma":
            return self._engine.capabilities.get("audio_in", False)
        return True

    async def transcribe(self, data: bytes, media_type: str) -> tuple[Transcription, Any]:
        """Return the transcript and the decoded waveform (or ``None``)."""
        waveform = None
        rate = self._settings.audio_sample_rate
        try:
            waveform, rate = decode(data, media_type, rate)
        except AudioDecodeError as exc:
            # Gemma can often still ingest the container directly.
            log.info("local decode failed (%s); passing raw bytes to the model", exc)

        seconds = duration_seconds(waveform, rate) if waveform is not None else 0.0

        if self._settings.asr_backend == "gemma":
            try:
                text = await self._engine.transcribe(data, media_type)
                return Transcription(text.strip(), "gemma", seconds), waveform
            except NotImplementedError:
                log.info("engine has no native ASR; falling back to whisper")
            except Exception as exc:  # pragma: no cover
                log.warning("native ASR failed (%s); falling back to whisper", exc)

        if waveform is None:
            raise TranscriptionError(
                "Could not decode that audio. Try a WAV file, or install ffmpeg."
            )

        model = _whisper(self._settings.asr_model_id)
        if model is None:
            raise TranscriptionError(
                "No speech-to-text backend is available. Install the ml extra or "
                "type your message instead."
            )

        def run() -> str:
            result = model({"raw": waveform, "sampling_rate": rate})
            return str(result.get("text", "")).strip()

        text = await asyncio.to_thread(run)
        if not text:
            raise TranscriptionError("I couldn't make out any speech in that recording.")
        return Transcription(text, "whisper", seconds), waveform
