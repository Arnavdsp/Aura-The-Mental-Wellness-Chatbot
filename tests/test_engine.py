"""The echo engine and engine selection."""

from __future__ import annotations

from aura.config import Settings
from aura.engine.base import GenerationRequest
from aura.engine.echo import EchoEngine
from aura.engine.registry import build_engine


def _request(text: str) -> GenerationRequest:
    return GenerationRequest(system_prompt="be a coach", user_text=text)


async def test_echo_reflects_and_asks_a_question(engine: EchoEngine) -> None:
    reply = await engine.generate(_request("I'm burned out at work and can't sleep"))
    assert reply.endswith("?")
    assert len(reply.split()) < 90


async def test_echo_is_deterministic_for_a_fixed_seed(settings) -> None:
    prompt = _request("I feel anxious about everything")
    first = await EchoEngine(settings).generate(prompt)
    second = await EchoEngine(settings).generate(prompt)
    assert first == second


async def test_echo_adapts_to_the_detected_mood(engine: EchoEngine) -> None:
    sad = await engine.generate(_request("I feel so sad and empty"))
    glad = await engine.generate(_request("I'm grateful and proud today"))
    assert sad != glad


async def test_echo_handles_empty_and_image_only_turns(engine: EchoEngine) -> None:
    assert await engine.generate(_request(""))
    image_only = GenerationRequest(system_prompt="x", user_text="", images=[b"fake"])
    assert "image" in (await engine.generate(image_only)).lower()


async def test_echo_streams_the_same_text_it_generates(engine: EchoEngine) -> None:
    request = _request("I'm overwhelmed by everything right now")
    buffered = await engine.generate(request)
    streamed = "".join([chunk async for chunk in engine.stream(request)])
    assert streamed.strip() == buffered.strip()


def test_registry_honours_an_explicit_echo_choice() -> None:
    assert build_engine(Settings(engine="echo")).name == "echo"


def test_registry_falls_back_when_the_ml_stack_is_missing(monkeypatch) -> None:
    monkeypatch.setattr("aura.engine.registry._gemma_available", lambda: False)
    assert build_engine(Settings(engine="auto")).name == "echo"


def test_registry_fails_loudly_when_gemma_is_required(monkeypatch) -> None:
    import pytest

    monkeypatch.setattr("aura.engine.registry._gemma_available", lambda: False)
    with pytest.raises(RuntimeError, match="not installed"):
        build_engine(Settings(engine="gemma"))
