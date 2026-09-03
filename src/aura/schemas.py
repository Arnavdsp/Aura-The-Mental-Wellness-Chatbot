"""Wire contracts shared by the API, the engines and the web client."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Modality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"


class RiskLevel(str, Enum):
    """How urgently a message needs a human, not a model."""

    NONE = "none"
    LOW = "low"
    ELEVATED = "elevated"
    CRISIS = "crisis"

    @property
    def rank(self) -> int:
        return {"none": 0, "low": 1, "elevated": 2, "crisis": 3}[self.value]


class Attachment(BaseModel):
    """A non-text part of a message, stored as a reference rather than bytes."""

    kind: Modality
    media_type: str = "application/octet-stream"
    uri: str | None = None
    filename: str | None = None
    size_bytes: int | None = None
    transcript: str | None = Field(
        default=None, description="Transcribed text, for audio attachments."
    )
    caption: str | None = Field(
        default=None, description="Model-generated description, for images."
    )


class AffectSignal(BaseModel):
    """What we believe the person is feeling, and how sure we are."""

    label: str = "neutral"
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Literal["text", "speech", "fused", "none"] = "none"
    scores: dict[str, float] = Field(default_factory=dict)


class SafetyAssessment(BaseModel):
    risk: RiskLevel = RiskLevel.NONE
    matched_categories: list[str] = Field(default_factory=list)
    rationale: str = ""
    resources: list[CrisisResource] = Field(default_factory=list)
    should_short_circuit: bool = False


class CrisisResource(BaseModel):
    name: str
    contact: str
    region: str = "INTL"
    url: str | None = None
    note: str | None = None


class Turn(BaseModel):
    """One message in a conversation."""

    id: str
    role: Role
    text: str = ""
    attachments: list[Attachment] = Field(default_factory=list)
    affect: AffectSignal | None = None
    safety: SafetyAssessment | None = None
    created_at: datetime = Field(default_factory=_now)
    latency_ms: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """A turn submitted by the client."""

    session_id: str | None = None
    message: str = ""
    speak: bool = Field(default=False, description="Also synthesise a spoken reply.")
    voice: str | None = None
    attachment_ids: list[str] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def _trim(cls, value: str) -> str:
        return value.strip()


class ChatResponse(BaseModel):
    session_id: str
    turn: Turn
    reply: Turn
    audio_url: str | None = None
    affect: AffectSignal
    safety: SafetyAssessment
    suggestions: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    attachment_id: str
    attachment: Attachment


class SessionSummary(BaseModel):
    session_id: str
    created_at: datetime
    updated_at: datetime
    turn_count: int
    topics: list[str] = Field(default_factory=list)
    mood_trend: list[float] = Field(default_factory=list)
    highest_risk: RiskLevel = RiskLevel.NONE


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str
    engine: str
    engine_ready: bool
    capabilities: dict[str, bool]
    uptime_seconds: float


SafetyAssessment.model_rebuild()
