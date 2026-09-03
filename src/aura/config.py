"""Runtime configuration.

Every knob is settable through the environment with an ``AURA_`` prefix, so the
same image runs on a laptop (echo engine, no GPU) and on a GPU host (Gemma 3n)
without a code change. See ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

EngineName = Literal["gemma", "echo", "auto"]


class Settings(BaseSettings):
    """Application settings, loaded from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="AURA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- service ---------------------------------------------------------
    app_name: str = "Aura"
    environment: Literal["development", "staging", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # -- engine ----------------------------------------------------------
    engine: EngineName = "auto"
    model_id: str = "unsloth/gemma-3n-E2B-it"
    adapter_path: str | None = None
    load_in_4bit: bool = True
    max_seq_length: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 64
    repetition_penalty: float = 1.05
    device: str | None = None  # None -> auto ("cuda" if available else "cpu")

    # -- modalities ------------------------------------------------------
    enable_vision: bool = True
    enable_audio_input: bool = True
    enable_audio_output: bool = True
    tts_backend: Literal["piper", "speecht5", "none"] = "speecht5"
    tts_model_id: str = "microsoft/speecht5_tts"
    tts_vocoder_id: str = "microsoft/speecht5_hifigan"
    tts_speaker_index: int = 7306
    asr_backend: Literal["gemma", "whisper", "none"] = "gemma"
    asr_model_id: str = "openai/whisper-small"
    audio_sample_rate: int = 16_000
    max_upload_bytes: int = 25 * 1024 * 1024

    # -- affect ----------------------------------------------------------
    speech_emotion_model_id: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    enable_speech_emotion: bool = False

    # -- session / memory ------------------------------------------------
    session_ttl_seconds: int = 60 * 60 * 6
    max_turns_in_context: int = 12
    max_sessions: int = 2_000

    # -- safety ----------------------------------------------------------
    safety_enabled: bool = True
    crisis_region: str = "INTL"

    # -- paths -----------------------------------------------------------
    static_dir: Path = REPO_ROOT / "web"
    data_dir: Path = REPO_ROOT / "var"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
