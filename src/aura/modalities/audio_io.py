"""Audio decoding and encoding.

Browsers hand us whatever their MediaRecorder produced (usually WebM/Opus), and
models want mono float32 PCM at a fixed rate. This module is the only place that
knows how to get from one to the other, with a pure-Python WAV path so plain
WAV uploads work with no system dependencies at all.
"""

from __future__ import annotations

import io
import shutil
import struct
import subprocess
import wave
from typing import TYPE_CHECKING

from aura.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    pass

log = get_logger(__name__)

WAV_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}


class AudioDecodeError(RuntimeError):
    """Raised when audio bytes cannot be turned into PCM."""


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def decode(data: bytes, media_type: str, target_rate: int = 16_000):
    """Decode arbitrary audio bytes to mono float32 in ``[-1, 1]``.

    Returns ``(waveform, sample_rate)``. Requires numpy; requires ffmpeg for
    anything that is not already a PCM WAV.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise AudioDecodeError("numpy is required to decode audio") from exc

    base_type = (media_type or "").split(";")[0].strip().lower()

    if base_type in WAV_TYPES or data[:4] == b"RIFF":
        waveform, rate = _decode_wav(data)
    elif ffmpeg_available():
        waveform, rate = _decode_via_ffmpeg(data, target_rate)
    else:
        raise AudioDecodeError(
            f"cannot decode {base_type or 'unknown audio'} — install ffmpeg, or "
            "have the client send WAV"
        )

    if rate != target_rate:
        waveform = _resample(waveform, rate, target_rate)
        rate = target_rate
    return np.asarray(waveform, dtype=np.float32), rate


def _decode_wav(data: bytes):
    import numpy as np

    with wave.open(io.BytesIO(data), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise AudioDecodeError(f"unsupported WAV sample width: {width} bytes")

    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if width == 1:  # 8-bit WAV is unsigned and centred on 128
        samples = (samples - 128.0) / 128.0
    else:
        samples /= float(2 ** (8 * width - 1))

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def _decode_via_ffmpeg(data: bytes, target_rate: int):
    import numpy as np

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(target_rate), "pipe:1",
    ]
    try:
        result = subprocess.run(command, input=data, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError("ffmpeg timed out while decoding audio") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[:300]
        raise AudioDecodeError(f"ffmpeg failed to decode audio: {detail}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy(), target_rate


def _resample(waveform, source_rate: int, target_rate: int):
    """Linear resampling — adequate for speech at 16 kHz."""
    import numpy as np

    if source_rate == target_rate or len(waveform) == 0:
        return waveform
    duration = len(waveform) / source_rate
    target_length = max(1, round(duration * target_rate))
    source_index = np.linspace(0, len(waveform) - 1, num=target_length)
    return np.interp(source_index, np.arange(len(waveform)), waveform).astype("float32")


def encode_wav(waveform, sample_rate: int) -> bytes:
    """Encode mono float32 samples as a 16-bit PCM WAV."""
    try:
        import numpy as np

        clipped = np.clip(np.asarray(waveform, dtype="float32"), -1.0, 1.0)
        pcm = (clipped * 32767.0).astype("<i2").tobytes()
    except ImportError:  # pragma: no cover - numpy-free fallback
        pcm = b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767)) for value in waveform
        )

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def duration_seconds(waveform, sample_rate: int) -> float:
    return round(len(waveform) / float(sample_rate or 1), 2)
