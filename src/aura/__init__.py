"""Aura — a multimodal mental wellness coach built on Gemma 3n.

Aura listens (text, voice, images), reasons about how someone is doing, and
responds with grounded, non-directive coaching in both text and speech.

Public surface:
    from aura import Settings, get_settings
    from aura.engine import build_engine
    from aura.api.app import create_app
"""

from aura.config import Settings, get_settings

__all__ = ["Settings", "__version__", "get_settings"]
__version__ = "1.0.0"
