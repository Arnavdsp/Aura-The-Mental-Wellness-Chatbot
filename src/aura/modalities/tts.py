"""Text to speech.

SpeechT5 is the default because it is small, permissively licensed and runs on
CPU. Piper is supported for hosts that have it installed and want lower latency.
Synthesis is best-effort by design: a failure here must never cost the user their
written reply, so callers treat ``None`` as "no audio this turn".
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from aura.config import Settings
from aura.logging import get_logger
from aura.modalities.audio_io import encode_wav

log = get_logger(__name__)

# SpeechT5 degrades on long inputs, so we synthesise sentence by sentence and
# concatenate — which also gives natural pauses at the joins.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_MAX_CHARS_PER_CHUNK = 220


@dataclass
class Speech:
    audio: bytes
    media_type: str = "audio/wav"
    sample_rate: int = 16_000
    backend: str = "none"


def strip_markup(text: str) -> str:
    """Remove markdown so the voice does not read asterisks aloud."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_#>]+", " ", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def _chunk(text: str) -> list[str]:
    chunks: list[str] = []
    for sentence in _SENTENCE_RE.split(text):
        sentence = sentence.strip()
        while len(sentence) > _MAX_CHARS_PER_CHUNK:
            cut = sentence.rfind(" ", 0, _MAX_CHARS_PER_CHUNK)
            cut = cut if cut > 0 else _MAX_CHARS_PER_CHUNK
            chunks.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


@lru_cache(maxsize=1)
def _speecht5_importable() -> bool:
    """Whether the SpeechT5 stack is installed, without loading any weights."""
    from importlib.util import find_spec

    return all(find_spec(name) is not None for name in ("torch", "transformers", "datasets"))


@lru_cache(maxsize=1)
def _speecht5(model_id: str, vocoder_id: str, speaker_index: int) -> Any | None:
    try:
        import torch  # type: ignore[import-not-found]
        from datasets import load_dataset
        from transformers import (
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )
    except ImportError:
        log.info("SpeechT5 dependencies unavailable; audio output disabled")
        return None
    try:
        processor = SpeechT5Processor.from_pretrained(model_id)
        model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
        vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id)
        embeddings = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )
        speaker = torch.tensor(
            embeddings[speaker_index % len(embeddings)]["xvector"]
        ).unsqueeze(0)
        model.eval()
        return processor, model, vocoder, speaker
    except Exception as exc:  # pragma: no cover - depends on weights/network
        log.warning("could not initialise SpeechT5: %s", exc)
        return None


class Synthesizer:
    """Renders coach replies to speech."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def available(self) -> bool:
        if not self._settings.enable_audio_output:
            return False
        if self._settings.tts_backend == "none":
            return False
        if self._settings.tts_backend == "piper":
            return shutil.which("piper") is not None
        return _speecht5_importable()

    async def synthesize(self, text: str) -> Speech | None:
        clean = strip_markup(text)
        if not clean or not self.available:
            return None
        try:
            if self._settings.tts_backend == "piper":
                return await asyncio.to_thread(self._piper, clean)
            return await asyncio.to_thread(self._speecht5, clean)
        except Exception as exc:  # never let a voice failure lose the reply
            log.warning("speech synthesis failed: %s", exc)
            return None

    def _speecht5(self, text: str) -> Speech | None:
        import numpy as np
        import torch  # type: ignore[import-not-found]

        bundle = _speecht5(
            self._settings.tts_model_id,
            self._settings.tts_vocoder_id,
            self._settings.tts_speaker_index,
        )
        if bundle is None:
            return None
        processor, model, vocoder, speaker = bundle

        pieces: list[Any] = []
        for chunk in _chunk(text):
            inputs = processor(text=chunk, return_tensors="pt")
            with torch.no_grad():
                audio = model.generate_speech(
                    inputs["input_ids"], speaker, vocoder=vocoder
                )
            pieces.append(audio.cpu().numpy())
            pieces.append(np.zeros(int(0.16 * 16_000), dtype="float32"))

        if not pieces:
            return None
        waveform = np.concatenate(pieces)
        return Speech(encode_wav(waveform, 16_000), sample_rate=16_000, backend="speecht5")

    def _piper(self, text: str) -> Speech | None:
        binary = shutil.which("piper")
        if binary is None:
            return None
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "speech.wav"
            result = subprocess.run(
                [binary, "--output_file", str(output)],
                input=text.encode(),
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0 or not output.exists():
                log.warning("piper failed: %s", result.stderr.decode("utf-8", "replace")[:200])
                return None
            return Speech(output.read_bytes(), sample_rate=22_050, backend="piper")
