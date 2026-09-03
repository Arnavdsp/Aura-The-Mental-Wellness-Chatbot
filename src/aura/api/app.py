"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aura import __version__
from aura.api.deps import AppState, get_state
from aura.api.routes_chat import router as chat_router
from aura.api.routes_session import resources_router
from aura.api.routes_session import router as session_router
from aura.coach import Coach
from aura.config import Settings, get_settings
from aura.engine.registry import build_engine
from aura.logging import configure_logging, get_logger, new_request_id
from aura.schemas import HealthResponse
from aura.session import SessionStore

log = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = build_engine(settings)
        store = SessionStore(settings)
        app.state.aura = AppState(
            settings=settings,
            engine=engine,
            store=store,
            coach=Coach(settings, engine, store),
            started_at=time.time(),
        )
        log.info(
            "starting %s v%s with the %s engine",
            settings.app_name, __version__, engine.name,
        )
        # Warm up off the critical path so the first request isn't a cold start,
        # but never block startup (or the health check) on model download.
        warmup = asyncio.create_task(_warmup(engine))
        try:
            yield
        finally:
            warmup.cancel()
            await engine.shutdown()
            log.info("shutdown complete")

    app = FastAPI(
        title="Aura — Multimodal Wellness Coach",
        description=(
            "A Gemma 3n powered wellness coach that listens in text, voice and "
            "images, and replies in text and speech. Not a medical device."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or new_request_id()
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": (
                    "Something went wrong on my end. Your conversation is safe — "
                    "please try that again."
                )
            },
        )

    app.include_router(chat_router)
    app.include_router(session_router)
    app.include_router(resources_router)

    @app.get("/api/health", response_model=HealthResponse, tags=["ops"])
    async def health(state: Annotated[AppState, Depends(get_state)]) -> HealthResponse:
        capabilities = dict(state.engine.capabilities)
        capabilities["audio_out"] = state.coach.synthesizer.available
        capabilities["asr"] = state.coach.transcriber.available or capabilities.get(
            "audio_in", False
        )
        return HealthResponse(
            status="ok" if state.engine.ready else "degraded",
            version=__version__,
            engine=state.engine.name,
            engine_ready=state.engine.ready,
            capabilities=capabilities,
            uptime_seconds=state.uptime_seconds,
        )

    _mount_web(app, settings)
    return app


async def _warmup(engine) -> None:
    try:
        await engine.warmup()
    except asyncio.CancelledError:  # pragma: no cover
        raise
    except Exception:  # pragma: no cover - the health endpoint reports this
        log.exception("engine warmup failed; serving in a degraded state")


def _mount_web(app: FastAPI, settings: Settings) -> None:
    static_dir = settings.static_dir
    if not static_dir.is_dir():
        log.warning("static directory %s not found; UI disabled", static_dir)
        return

    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")


app = create_app()
