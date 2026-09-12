"""Vercel serverless entrypoint.

Vercel autodetects a FastAPI ``app`` only in a handful of default locations, and
this project keeps its code under ``src/``. ``pyproject.toml`` declares
``[tool.vercel] entrypoint`` for that, and this module is the belt-and-braces
version: it works whether or not the build installed the package, so a deploy
does not depend on which dependency file Vercel decided to honour.

What runs here is the **echo engine** — Vercel has no GPU and a serverless
bundle cannot hold PyTorch, so Gemma 3n itself must be deployed somewhere with
an accelerator. See "Deploying" in the README.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support running from a source checkout that was never pip-installed.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aura.api.app import app  # noqa: E402  (path setup must precede the import)

__all__ = ["app"]
