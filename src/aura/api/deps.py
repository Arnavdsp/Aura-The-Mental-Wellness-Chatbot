"""Shared application state and FastAPI dependencies."""

from __future__ import annotations

import time
from dataclasses import dataclass

from fastapi import Request

from aura.coach import Coach
from aura.config import Settings
from aura.engine.base import CoachEngine
from aura.session import SessionStore


@dataclass
class AppState:
    settings: Settings
    engine: CoachEngine
    store: SessionStore
    coach: Coach
    started_at: float

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self.started_at, 2)


def get_state(request: Request) -> AppState:
    return request.app.state.aura


def get_coach(request: Request) -> Coach:
    return request.app.state.aura.coach


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.aura.settings


def get_store(request: Request) -> SessionStore:
    return request.app.state.aura.store
