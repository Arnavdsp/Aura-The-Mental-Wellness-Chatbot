"""Image intake.

Gemma 3n consumes images directly, so this module only has to make uploads safe
and cheap: verify the bytes really are an image, strip metadata by re-encoding,
and bound the resolution so one large photo cannot dominate the context window.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from aura.logging import get_logger

log = get_logger(__name__)

MAX_EDGE = 896  # Gemma 3n's vision tower works at 768-896px; more is wasted tokens.
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp"}


class ImageError(ValueError):
    pass


@dataclass
class PreparedImage:
    data: bytes
    media_type: str
    width: int
    height: int


def prepare(data: bytes, media_type: str) -> PreparedImage:
    """Validate, normalise orientation, downscale and re-encode as PNG.

    Re-encoding is deliberate: it drops EXIF (which can carry GPS coordinates)
    and guarantees the bytes we hand the model are a real image, not a payload
    with an image extension.
    """
    base_type = (media_type or "").split(";")[0].strip().lower()
    if base_type and base_type not in ALLOWED_TYPES:
        raise ImageError(f"unsupported image type: {base_type}")

    try:
        from PIL import Image, ImageOps  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImageError("Pillow is required to accept images") from exc

    try:
        image = Image.open(io.BytesIO(data))
        image.verify()  # verify() consumes the file, so reopen afterwards
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ImageError("that file could not be read as an image") from exc

    if max(image.size) > MAX_EDGE:
        image.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return PreparedImage(buffer.getvalue(), "image/png", image.width, image.height)
