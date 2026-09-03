"""Packaging invariants.

The UI used to live at the repository root and be resolved by walking up from
``config.py``. That worked for an editable checkout and broke silently for any
non-editable install — ``pip install .`` and every serverless bundle — leaving
an API with no front end and only a log line to say so. These tests pin the
packaged layout so that regression cannot come back quietly.
"""

from __future__ import annotations

from pathlib import Path

import aura
from aura.config import STATIC_DIR, Settings


def test_ui_ships_inside_the_package() -> None:
    package_root = Path(aura.__file__).resolve().parent
    assert package_root / "web" == STATIC_DIR
    assert STATIC_DIR.is_dir()


def test_ui_assets_are_present() -> None:
    for asset in ("index.html", "styles.css", "app.js"):
        assert (STATIC_DIR / asset).is_file(), f"missing packaged asset: {asset}"


def test_static_dir_default_does_not_depend_on_the_working_directory() -> None:
    assert Settings().static_dir == STATIC_DIR


def test_static_dir_is_overridable(tmp_path: Path) -> None:
    assert Settings(static_dir=tmp_path).static_dir == tmp_path


def test_index_html_references_the_mounted_asset_paths() -> None:
    """The app mounts /assets; the markup must ask for exactly that."""
    markup = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "/assets/styles.css" in markup
    assert "/assets/app.js" in markup


def test_vercel_entrypoint_exposes_the_asgi_app() -> None:
    """`api/index.py` is what Vercel imports; keep it importable and correct."""
    import importlib.util

    entrypoint = Path(aura.__file__).resolve().parents[2] / "api" / "index.py"
    if not entrypoint.is_file():  # not present in an installed wheel
        return
    spec = importlib.util.spec_from_file_location("aura_vercel_entry", entrypoint)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from fastapi import FastAPI

    assert isinstance(module.app, FastAPI)
