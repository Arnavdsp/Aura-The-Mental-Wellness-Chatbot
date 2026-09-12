"""The orchestrator.

One turn of conversation, in order:

1. Resolve attachments — transcribe audio, normalise images.
2. Screen for risk *before* generating anything.
3. Estimate affect from text and, when we have a waveform, from voice.
4. Build the system prompt from stance + affect + conversation memory.
5. Generate (or short-circuit to crisis resources).
6. Optionally synthesise speech.

Both the buffered and streaming paths run through the same preparation so they
cannot drift apart; only step 5 differs.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from aura import affect as affect_module
from aura.config import Settings
from aura.engine.base import CoachEngine, GenerationRequest
from aura.logging import get_logger
from aura.memory import ConversationMemory
from aura.modalities.asr import Transcriber, TranscriptionError
from aura.modalities.tts import Synthesizer
from aura.modalities.vision import ImageError
from aura.modalities.vision import prepare as prepare_image
from aura.prompts import build_system_prompt, suggestions_for
from aura.safety import crisis_message, screen
from aura.schemas import (
    AffectSignal,
    Attachment,
    ChatRequest,
    InlineAttachment,
    Modality,
    Role,
    SafetyAssessment,
    Turn,
)
from aura.session import SessionStore, StoredAttachment, new_id

log = get_logger(__name__)


@dataclass
class PreparedTurn:
    """Everything resolved about a user turn before generation runs."""

    memory: ConversationMemory
    user_turn: Turn
    generation: GenerationRequest
    affect: AffectSignal
    safety: SafetyAssessment
    notices: list[str] = field(default_factory=list)

    @property
    def short_circuit(self) -> str | None:
        """The reply to send instead of generating, when risk demands it."""
        if self.safety.should_short_circuit:
            return crisis_message(self.safety)
        return None



def _inline_to_stored(item: InlineAttachment) -> StoredAttachment:
    """Adapt an inline upload to the same shape a staged upload has."""
    return StoredAttachment(
        id=new_id("att"),
        attachment=Attachment(
            kind=item.kind,
            media_type=item.media_type,
            filename=item.filename,
            size_bytes=len(item.data),
        ),
        data=item.data,
    )


class Coach:
    """Stateless per-request orchestration over the engine and session store."""

    def __init__(self, settings: Settings, engine: CoachEngine, store: SessionStore) -> None:
        self.settings = settings
        self.engine = engine
        self.store = store
        self.transcriber = Transcriber(settings, engine)
        self.synthesizer = Synthesizer(settings)

    # -- preparation -----------------------------------------------------

    async def prepare(self, request: ChatRequest) -> PreparedTurn:
        memory = await self.store.get_or_create(request.session_id)
        stored = [_inline_to_stored(item) for item in request.attachments]
        stored += await self.store.pop_attachments(request.attachment_ids)

        attachments: list[Attachment] = []
        images: list[bytes] = []
        audio_blobs: list[bytes] = []
        notices: list[str] = []
        modalities: set[str] = set()
        waveform: Any = None
        transcripts: list[str] = []

        for item in stored:
            if item.attachment.kind is Modality.IMAGE:
                self._attach_image(item, attachments, images, notices, modalities)
            elif item.attachment.kind is Modality.AUDIO:
                waveform = await self._attach_audio(
                    item, attachments, audio_blobs, transcripts, notices, modalities
                ) or waveform

        spoken = " ".join(transcripts).strip()
        combined = " ".join(part for part in (request.message, spoken) if part).strip()

        safety = screen(
            combined,
            region=self.settings.crisis_region,
            enabled=self.settings.safety_enabled,
        )
        signal = self._estimate_affect(combined, waveform)

        user_turn = Turn(
            id=new_id("turn"),
            role=Role.USER,
            text=request.message,
            attachments=attachments,
            affect=signal,
            safety=safety,
        )

        generation = GenerationRequest(
            system_prompt=build_system_prompt(
                memory, signal, modalities=modalities, risk=safety.risk
            ),
            history=self._history(memory),
            user_text=combined or request.message,
            images=images,
            audio=audio_blobs,
            attachments=attachments,
        )

        return PreparedTurn(memory, user_turn, generation, signal, safety, notices)

    def _attach_image(
        self,
        item: StoredAttachment,
        attachments: list[Attachment],
        images: list[bytes],
        notices: list[str],
        modalities: set[str],
    ) -> None:
        if not self.settings.enable_vision:
            notices.append("Image input is disabled on this server.")
            return
        try:
            prepared = prepare_image(item.data, item.attachment.media_type)
        except ImageError as exc:
            notices.append(f"I couldn't read that image: {exc}")
            return
        images.append(prepared.data)
        item.attachment.media_type = prepared.media_type
        item.attachment.size_bytes = len(prepared.data)
        attachments.append(item.attachment)
        modalities.add("image")

    async def _attach_audio(
        self,
        item: StoredAttachment,
        attachments: list[Attachment],
        audio_blobs: list[bytes],
        transcripts: list[str],
        notices: list[str],
        modalities: set[str],
    ) -> Any:
        if not self.settings.enable_audio_input:
            notices.append("Voice input is disabled on this server.")
            return None
        modalities.add("audio")
        try:
            transcription, waveform = await self.transcriber.transcribe(
                item.data, item.attachment.media_type
            )
        except TranscriptionError as exc:
            notices.append(str(exc))
            attachments.append(item.attachment)
            return None

        item.attachment.transcript = transcription.text
        attachments.append(item.attachment)
        if transcription.text:
            transcripts.append(transcription.text)
        # Only hand raw audio to the model if it did not already transcribe it.
        if transcription.backend != "gemma":
            audio_blobs.append(item.data)
        return waveform

    def _estimate_affect(self, text: str, waveform: Any) -> AffectSignal:
        text_signal = affect_module.analyse_text(text)
        if waveform is None or not self.settings.enable_speech_emotion:
            return text_signal
        speech_signal = affect_module.analyse_speech(
            waveform,
            self.settings.audio_sample_rate,
            model_id=self.settings.speech_emotion_model_id,
        )
        return affect_module.fuse(text_signal, speech_signal)

    def _history(self, memory: ConversationMemory) -> list[dict[str, str]]:
        history: list[dict[str, str]] = []
        for turn in memory.recent(self.settings.max_turns_in_context):
            content = turn.text or " ".join(
                filter(None, (a.transcript or a.caption for a in turn.attachments))
            )
            if content:
                history.append({"role": turn.role.value, "content": content})
        return history

    # -- generation ------------------------------------------------------

    async def respond(self, request: ChatRequest) -> tuple[PreparedTurn, Turn, str | None]:
        """Buffered turn. Returns the prepared turn, the reply and any audio id."""
        prepared = await self.prepare(request)
        started = time.perf_counter()

        text = prepared.short_circuit
        if text is None:
            text = await self.engine.generate(prepared.generation)
            text = _postprocess(text)

        reply = self._finalise(prepared, text, started)
        audio_id = await self._maybe_speak(text, request.speak)
        return prepared, reply, audio_id

    async def stream(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        """Streaming turn, yielding SSE-shaped events."""
        prepared = await self.prepare(request)
        started = time.perf_counter()

        yield {
            "event": "meta",
            "data": {
                "session_id": prepared.memory.session_id,
                "affect": prepared.affect.model_dump(mode="json"),
                "safety": prepared.safety.model_dump(mode="json"),
                "notices": prepared.notices,
                "user_turn": prepared.user_turn.model_dump(mode="json"),
            },
        }

        chunks: list[str] = []
        short_circuit = prepared.short_circuit
        if short_circuit is not None:
            chunks.append(short_circuit)
            yield {"event": "token", "data": {"text": short_circuit}}
        else:
            try:
                async for chunk in self.engine.stream(prepared.generation):
                    chunks.append(chunk)
                    yield {"event": "token", "data": {"text": chunk}}
            except Exception as exc:  # pragma: no cover - engine dependent
                log.exception("streaming generation failed")
                yield {"event": "error", "data": {"message": str(exc)}}
                return

        text = _postprocess("".join(chunks))
        reply = self._finalise(prepared, text, started)
        audio_id = await self._maybe_speak(text, request.speak)

        yield {
            "event": "done",
            "data": {
                "reply": reply.model_dump(mode="json"),
                "audio_url": f"/api/audio/{audio_id}" if audio_id else None,
                "suggestions": suggestions_for(prepared.affect, prepared.safety.risk),
            },
        }

    def _finalise(self, prepared: PreparedTurn, text: str, started: float) -> Turn:
        """Commit both turns to memory once the reply is complete."""
        reply = Turn(
            id=new_id("turn"),
            role=Role.ASSISTANT,
            text=text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            meta={
                "engine": self.engine.name,
                "short_circuited": prepared.safety.should_short_circuit,
            },
        )
        prepared.memory.add(prepared.user_turn)
        prepared.memory.add(reply)
        return reply

    async def _maybe_speak(self, text: str, requested: bool) -> str | None:
        if not requested:
            return None
        speech = await self.synthesizer.synthesize(text)
        if speech is None:
            return None
        return await self.store.put_audio(speech.audio, speech.media_type)


def _postprocess(text: str) -> str:
    """Trim template leakage and role prefixes some checkpoints emit."""
    cleaned = text.strip()
    for marker in ("<end_of_turn>", "<eos>", "<start_of_turn>model", "<start_of_turn>"):
        cleaned = cleaned.replace(marker, "")
    for prefix in ("Aura:", "Assistant:", "model\n", "model:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):]
    return cleaned.strip() or (
        "I'm here, but I lost my words for a second. Could you say that again?"
    )
