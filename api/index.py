"""
Vercel serverless function entrypoint.

Vercel routes every request to this file (vercel.json rewrites → /api/index).
Vercel's Python runtime natively supports ASGI apps — it auto-detects the
``app`` instance and handles the ASGI protocol internally, so no Mangum
adapter is needed.

Static files (index.html, styles.css, app.js) live in ``public/`` and are
served directly by Vercel's CDN (outputDirectory config), so FastAPI only
needs to handle API routes.
"""

from __future__ import annotations

import os
import sys

# Ensure the ``src/`` package tree is importable.  Vercel's Python runtime
# installs third-party deps from requirements.txt but the project's own
# ``aura`` package lives under ``src/`` and isn't pip-installed at runtime.
_src = os.path.join(os.path.dirname(__file__), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Point the app at ``public/`` for static files on Vercel.
# When AURA_STATIC_DIR is not set, config.py falls back to ``web/`` (local dev).
os.environ.setdefault("AURA_STATIC_DIR", os.path.join(os.path.dirname(__file__), "..", "public"))

# Disable uvicorn's HTTP/WS proxies — Vercel handles those.
os.environ.setdefault("AURA_DISABLE_PROXY", "1")

# Vercel serverless functions are always production.
os.environ.setdefault("AURA_ENVIRONMENT", "production")

from aura.api.app import create_app

# Build the app once per cold-start.  Vercel keeps the function warm briefly;
# subsequent warm invocations reuse this same app instance.
app = create_app()

