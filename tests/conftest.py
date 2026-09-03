"""Shared fixtures. Everything runs on the echo engine — no GPU, no downloads."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aura.api.app import create_app
from aura.config import Settings
from aura.engine.echo import EchoEngine
from aura.session import SessionStore


@pytest.fixture
def settings() -> Settings:
    return Settings(
        engine="echo",
        environment="development",
        enable_vision=True,
        enable_audio_input=True,
        enable_audio_output=False,
        tts_backend="none",
        crisis_region="US",
    )


@pytest.fixture
def store(settings: Settings) -> SessionStore:
    return SessionStore(settings)


@pytest.fixture
def engine(settings: Settings) -> EchoEngine:
    return EchoEngine(settings)


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def png_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (90, 120, 100)).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    import numpy as np

    from aura.modalities.audio_io import encode_wav

    t = np.linspace(0, 0.5, 8000, dtype="float32")
    return encode_wav(0.3 * np.sin(2 * np.pi * 220 * t), 16_000)
