"""Chat, upload and media endpoints."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse

from aura.api.deps import AppState, get_coach, get_settings_dep, get_state
from aura.coach import Coach
from aura.config import Settings
from aura.logging import get_logger
from aura.prompts import suggestions_for
from aura.schemas import Attachment, ChatRequest, ChatResponse, Modality, UploadResponse

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

_AUDIO_PREFIX = "audio/"
_IMAGE_PREFIX = "image/"


def _validate(request: ChatRequest, settings: Settings) -> None:
    """Reject empty turns and oversized inline uploads before any work starts."""
    if not request.message and not request.attachment_ids and not request.attachments:
        raise HTTPException(status_code=422, detail="Send a message or an attachment.")
    limit = settings.max_upload_bytes
    if any(len(item.data) > limit for item in request.attachments):
        raise HTTPException(
            status_code=413,
            detail=f"That file is larger than the {limit // (1024 * 1024)} MB limit.",
        )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    coach: Annotated[Coach, Depends(get_coach)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> ChatResponse:
    """One buffered turn. Use ``/api/chat/stream`` for a live typing effect."""
    _validate(request, settings)

    prepared, reply, audio_id = await coach.respond(request)
    return ChatResponse(
        session_id=prepared.memory.session_id,
        turn=prepared.user_turn,
        reply=reply,
        audio_url=f"/api/audio/{audio_id}" if audio_id else None,
        affect=prepared.affect,
        safety=prepared.safety,
        suggestions=suggestions_for(prepared.affect, prepared.safety.risk),
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    coach: Annotated[Coach, Depends(get_coach)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> StreamingResponse:
    """Server-sent events: ``meta``, then ``token``\\*, then ``done``."""
    _validate(request, settings)

    async def publish() -> AsyncIterator[str]:
        try:
            async for event in coach.stream(request):
                payload = json.dumps(event["data"], default=str)
                yield f"event: {event['event']}\ndata: {payload}\n\n"
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("stream failed")
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

    return StreamingResponse(
        publish(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # don't let nginx buffer the stream
        },
    )


@router.post("/uploads", response_model=UploadResponse)
async def upload(
    state: Annotated[AppState, Depends(get_state)],
    file: Annotated[UploadFile, File()],
    kind: Annotated[str | None, Form()] = None,
) -> UploadResponse:
    """Stage an image or audio clip, returning an id to attach to a chat turn."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="That file was empty.")
    limit = state.settings.max_upload_bytes
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"That file is larger than the {limit // (1024 * 1024)} MB limit.",
        )

    media_type = file.content_type or "application/octet-stream"
    if kind == "audio" or media_type.startswith(_AUDIO_PREFIX) or media_type == "video/webm":
        modality = Modality.AUDIO
    elif kind == "image" or media_type.startswith(_IMAGE_PREFIX):
        modality = Modality.IMAGE
    else:
        raise HTTPException(
            status_code=415, detail=f"Unsupported file type: {media_type}"
        )

    attachment = Attachment(
        kind=modality,
        media_type=media_type,
        filename=file.filename,
        size_bytes=len(data),
    )
    attachment_id = await state.store.put_attachment(attachment, data)
    return UploadResponse(attachment_id=attachment_id, attachment=attachment)


@router.get("/attachments/{attachment_id}")
async def get_attachment(
    attachment_id: str, state: Annotated[AppState, Depends(get_state)]
) -> Response:
    stored = await state.store.get_attachment(attachment_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="That attachment has expired.")
    return Response(
        content=stored.data,
        media_type=stored.attachment.media_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/audio/{audio_id}")
async def get_audio(
    audio_id: str, state: Annotated[AppState, Depends(get_state)]
) -> Response:
    entry = await state.store.get_audio(audio_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="That audio has expired.")
    data, media_type = entry
    return Response(
        content=data,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )
