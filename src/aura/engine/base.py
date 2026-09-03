"""The engine contract every backend implements."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from aura.schemas import Attachment


@dataclass
class GenerationRequest:
    """One generation call: a system prompt, a history, and this turn's media."""

    system_prompt: str
    history: list[dict[str, str]] = field(default_factory=list)
    user_text: str = ""
    images: list[bytes] = field(default_factory=list)
    audio: list[bytes] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)
    max_new_tokens: int | None = None
    temperature: float | None = None


class CoachEngine(abc.ABC):
    """A text-and-media in, text out conversational backend."""

    name: str = "base"

    @property
    @abc.abstractmethod
    def ready(self) -> bool:
        """True once weights are loaded and the engine can serve traffic."""

    @property
    def capabilities(self) -> dict[str, bool]:
        return {"text": True, "vision": False, "audio_in": False, "streaming": True}

    @abc.abstractmethod
    async def warmup(self) -> None:
        """Load weights. Called once at startup; safe to call again."""

    @abc.abstractmethod
    async def generate(self, request: GenerationRequest) -> str:
        """Produce a complete reply."""

    @abc.abstractmethod
    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Produce a reply as incremental token chunks."""

    async def transcribe(self, audio: bytes, media_type: str) -> str:
        """Transcribe audio, when the engine supports it natively."""
        raise NotImplementedError

    async def shutdown(self) -> None:
        """Release resources. Default is a no-op."""
        return None
