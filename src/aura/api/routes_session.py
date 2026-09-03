"""Session inspection and lifecycle."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from aura.api.deps import AppState, get_state
from aura.safety import resources_for
from aura.schemas import CrisisResource, SessionSummary

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionSummary])
async def list_sessions(state: Annotated[AppState, Depends(get_state)]):
    return await state.store.summaries()


@router.get("/{session_id}")
async def get_session(
    session_id: str, state: Annotated[AppState, Depends(get_state)]
) -> dict[str, Any]:
    """Full transcript plus the derived insights the UI's side panel renders."""
    memory = await state.store.get(session_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="No such session.")
    return {
        "session_id": memory.session_id,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "turns": [turn.model_dump(mode="json") for turn in memory.turns],
        "insights": {
            "graph": memory.graph.as_dict(),
            "mood_trend": memory.mood_trend,
            "mood_direction": memory.mood_direction(),
            "highest_risk": memory.highest_risk.value,
            "context_note": memory.context_note(),
        },
    }


@router.delete("/{session_id}")
async def delete_session(
    session_id: str, state: Annotated[AppState, Depends(get_state)]
) -> dict[str, bool]:
    """Erase a conversation. Nothing is retained after this returns."""
    return {"deleted": await state.store.delete(session_id)}


resources_router = APIRouter(prefix="/api", tags=["safety"])


@resources_router.get("/resources", response_model=list[CrisisResource])
async def crisis_resources(
    state: Annotated[AppState, Depends(get_state)], region: str | None = None
):
    return resources_for(region or state.settings.crisis_region)
