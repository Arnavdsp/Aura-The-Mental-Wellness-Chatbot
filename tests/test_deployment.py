"""Deployment invariants for the container and the free hosting tiers.

Render, Railway, Fly and Hugging Face Spaces all decide the port themselves and
hand it to the process as ``PORT``. A container that ignores it binds the wrong
port, fails the platform's health check, and is reported as a failed deploy with
no error in the application log — so the binding rules are worth pinning.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aura.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile"


def test_platform_port_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "7860")
    monkeypatch.delenv("AURA_PORT", raising=False)
    assert Settings().port == 7860


def test_aura_port_wins_over_platform_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit AURA_PORT is a deliberate choice; PORT is the platform's default."""
    monkeypatch.setenv("PORT", "7860")
    monkeypatch.setenv("AURA_PORT", "9000")
    assert Settings().port == 9000


def test_port_falls_back_to_the_local_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("AURA_PORT", raising=False)
    assert Settings().port == 8000


@pytest.mark.skipif(not DOCKERFILE.is_file(), reason="source checkout only")
def test_container_binds_the_platform_port() -> None:
    """The CMD must read PORT at runtime, not bake a port into the image."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    cmd = next(line for line in text.splitlines() if line.startswith("CMD "))
    assert "${PORT:-" in cmd, "CMD ignores the platform-supplied PORT"


@pytest.mark.skipif(not DOCKERFILE.is_file(), reason="source checkout only")
def test_container_healthcheck_follows_the_same_port() -> None:
    """A health check pinned to 8000 reports unhealthy whenever PORT differs."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    healthcheck = re.search(r"^\s*CMD curl.*$", text, re.MULTILINE)
    assert healthcheck and "${PORT:-" in healthcheck.group(0)


@pytest.mark.skipif(not DOCKERFILE.is_file(), reason="source checkout only")
def test_container_runs_as_the_uid_spaces_expects() -> None:
    """Hugging Face Spaces runs containers as UID 1000; matching it avoids
    unwritable HOME and cache directories at runtime."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "--uid 1000" in text
    assert "USER aura" in text


@pytest.mark.skipif(
    not (REPO_ROOT / "deploy" / "huggingface" / "README.md").is_file(),
    reason="source checkout only",
)
def test_space_readme_carries_the_required_front_matter() -> None:
    """Spaces reads its build config from this YAML; a missing key silently
    falls back to a Gradio build that cannot run this app."""
    text = (REPO_ROOT / "deploy" / "huggingface" / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front_matter = text.split("---", 2)[1]
    assert "sdk: docker" in front_matter
    assert "app_port: 8000" in front_matter
