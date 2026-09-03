"""Engine selection.

``AURA_ENGINE=auto`` (the default) tries Gemma 3n and quietly falls back to the
echo engine when the ML stack or a GPU is absent — so ``docker run`` and
``pytest`` work on any machine, while a GPU host gets the real model.
"""

from __future__ import annotations

from aura.config import Settings
from aura.engine.base import CoachEngine
from aura.logging import get_logger

log = get_logger(__name__)


def _gemma_available() -> bool:
    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # noqa: F401
    except ImportError:
        return False
    return bool(getattr(torch, "cuda", None) and torch.cuda.is_available()) or True


def build_engine(settings: Settings) -> CoachEngine:
    """Instantiate the configured engine (weights load later, in ``warmup``)."""
    choice = settings.engine

    if choice == "echo":
        from aura.engine.echo import EchoEngine

        return EchoEngine(settings)

    if choice in {"gemma", "auto"}:
        if _gemma_available():
            from aura.engine.gemma import GemmaEngine

            return GemmaEngine(settings)
        if choice == "gemma":
            raise RuntimeError(
                "engine='gemma' requested but torch/transformers are not installed. "
                "Install the ml extra: pip install -e '.[ml]'"
            )
        log.warning("torch/transformers unavailable — falling back to the echo engine")

    from aura.engine.echo import EchoEngine

    return EchoEngine(settings)
