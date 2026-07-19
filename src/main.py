#!/usr/bin/env python3
"""VoiceCreate application entry point."""

from __future__ import annotations

import logging

import sys
from pathlib import Path

logger = logging.getLogger("VoiceCreate")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> int:
    """Start the VoiceCreate desktop application."""
    try:
        from gui.main_application import VoiceCreateApp

        VoiceCreateApp().run()
        return 0
    except Exception as exc:
        logger.exception("VoiceCreate startup failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
