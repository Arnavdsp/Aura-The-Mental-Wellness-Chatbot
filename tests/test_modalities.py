"""Audio and image handling, without any model weights."""

from __future__ import annotations

import io

import numpy as np
import pytest

from aura.modalities.audio_io import (
    AudioDecodeError,
    decode,
    duration_seconds,
    encode_wav,
)
from aura.modalities.tts import _chunk, strip_markup
from aura.modalities.vision import ImageError, prepare


def _tone(seconds: float = 0.5, rate: int = 22_050) -> np.ndarray:
    t = np.linspace(0, seconds, int(rate * seconds), dtype="float32")
    return (0.4 * np.sin(2 * np.pi * 440 * t)).astype("float32")


def test_wav_round_trip_preserves_amplitude() -> None:
    waveform = _tone()
    decoded, rate = decode(encode_wav(waveform, 22_050), "audio/wav", 22_050)
    assert rate == 22_050
    assert len(decoded) == len(waveform)
    assert np.max(np.abs(decoded - waveform)) < 1e-3


def test_decode_resamples_to_the_target_rate() -> None:
    decoded, rate = decode(encode_wav(_tone(1.0, 22_050), 22_050), "audio/wav", 16_000)
    assert rate == 16_000
    assert abs(duration_seconds(decoded, rate) - 1.0) < 0.02


def test_stereo_is_mixed_down_to_mono() -> None:
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.zeros(3200, dtype="<i2").tobytes())
    decoded, _ = decode(buffer.getvalue(), "audio/wav", 16_000)
    assert decoded.ndim == 1
    assert len(decoded) == 1600


def test_wav_is_detected_by_magic_bytes_not_just_media_type() -> None:
    decoded, _ = decode(encode_wav(_tone(), 22_050), "application/octet-stream", 22_050)
    assert len(decoded) > 0


def test_undecodable_audio_raises_a_clear_error() -> None:
    with pytest.raises(AudioDecodeError):
        decode(b"definitely not audio", "audio/x-unknown-format", 16_000)


def test_encode_clips_out_of_range_samples() -> None:
    loud = np.array([2.0, -2.0, 0.0], dtype="float32")
    decoded, _ = decode(encode_wav(loud, 16_000), "audio/wav", 16_000)
    assert np.max(np.abs(decoded)) <= 1.0


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        ("**Bold** text", "Bold text"),
        ("See [the site](https://x.com)", "See the site"),
        ("`code` here", "code here"),
        ("- item one\n- item two", "item one item two"),
        ("# Heading\n\nBody", "Heading Body"),
    ],
)
def test_strip_markup_keeps_only_speakable_text(markup: str, expected: str) -> None:
    assert strip_markup(markup) == expected


def test_tts_chunking_splits_on_sentences_and_bounds_length() -> None:
    assert _chunk("One. Two! Three?") == ["One.", "Two!", "Three?"]
    long_sentence = " ".join(["word"] * 200)
    assert all(len(chunk) <= 220 for chunk in _chunk(long_sentence))


def test_image_is_downscaled_and_re_encoded(png_bytes: bytes) -> None:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (3000, 1500), (10, 20, 30)).save(buffer, "JPEG")
    prepared = prepare(buffer.getvalue(), "image/jpeg")
    assert prepared.media_type == "image/png"   # re-encoded, so EXIF is gone
    assert max(prepared.width, prepared.height) <= 896


def test_small_images_are_not_upscaled(png_bytes: bytes) -> None:
    prepared = prepare(png_bytes, "image/png")
    assert (prepared.width, prepared.height) == (64, 48)


def test_non_image_bytes_are_rejected() -> None:
    with pytest.raises(ImageError):
        prepare(b"<html>not an image</html>", "image/png")


def test_unsupported_image_type_is_rejected(png_bytes: bytes) -> None:
    with pytest.raises(ImageError, match="unsupported"):
        prepare(png_bytes, "image/tiff")
