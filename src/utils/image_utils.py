"""Image validation helpers for VoiceCreate."""

from __future__ import annotations

import logging

from typing import Tuple

from PIL import Image


logger = logging.getLogger(__name__)


def validate_image_integrity(image: Image.Image) -> Tuple[bool, str]:
    """Validate that a PIL image is present, readable, and has positive dimensions."""
    try:
        if image is None:
            return False, "image is None"

        if image.mode not in ("RGB", "RGBA", "L"):
            logger.warning("Uncommon image mode: %s", image.mode)

        width, height = image.size
        if width <= 0 or height <= 0:
            return False, f"invalid image size: {width}x{height}"

        try:
            image.getdata()
        except Exception as exc:
            return False, f"image data is not readable: {exc}"

        return True, f"image valid: {width}x{height}, mode={image.mode}"
    except Exception as exc:
        return False, f"image validation failed: {exc}"
