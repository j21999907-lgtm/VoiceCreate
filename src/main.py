#!/usr/bin/env python3
"""VoiceCreate application entry point."""

from __future__ import annotations

import logging

import os
import sys
from pathlib import Path

logger = logging.getLogger("VoiceCreate")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_runtime_dirs(config: dict) -> None:
    storage_config = config.get("storage", {})
    for path in [
        storage_config.get("image_dir", "./generated_images"),
        storage_config.get("temp_dir", "./temp"),
        storage_config.get("backup_dir", "./backups"),
        config.get("logging", {}).get("path", "./logs"),
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"[OK] Directory ready: {path}")


def _load_speech_model(config: dict, global_state: dict, logger) -> None:
    from speech.model_loader import load_vosk_model

    speech_config = config.get("speech", {})
    vosk_model = load_vosk_model(speech_config)
    if vosk_model is None:
        logger.warning("VOSK speech model failed to load")
        logger.info("[WARN] VOSK speech model failed to load")
        return

    lock = global_state.get("lock")
    if lock:
        with lock:
            global_state["modules"]["speech_recognizer"] = vosk_model
    else:
        global_state["modules"]["speech_recognizer"] = vosk_model

    logger.info("VOSK speech model loaded")
    logger.info("[OK] VOSK speech model loaded")


def _load_image_model(config: dict, global_state: dict, logger) -> None:
    from image.dreamlite_fixed import (
        initialize_dreamlite_model,
        register_model_to_global_state,
    )

    # The current config file uses "iimage"; support the corrected "image" key too.
    image_config = config.get("image") or config.get("iimage") or {}
    image_model = initialize_dreamlite_model(image_config)
    if image_model is None:
        logger.warning("Image generator failed to load")
        logger.info("[WARN] Image generator failed to load")
        return

    register_model_to_global_state(global_state, image_model)
    model_name = image_config.get("model_type", "image")
    logger.info("%s image generator loaded", model_name)
    logger.info("[OK] %s image generator loaded", model_name)
    if getattr(image_model, "using_mock", False):
        logger.warning("%s is running in mock fallback mode", model_name)
        logger.info("[WARN] %s is running in mock fallback mode", model_name)


def _register_display_module(config: dict, global_state: dict, logger) -> None:
    from display.image_display import display_image_at

    display_config = config.get("display", {})
    display_time = int(display_config.get("display_time", 5000))

    def display_image(image, x, y, display_time_ms=display_time):
        return display_image_at(image, x, y, display_time=display_time_ms)

    lock = global_state.get("lock")
    if lock:
        with lock:
            global_state["modules"]["image_display"] = display_image
    else:
        global_state["modules"]["image_display"] = display_image

    logger.info("Image display module registered")
    logger.info("[OK] Image display module registered")


def main() -> int:
    """Initialize VoiceCreate and register core runtime modules."""
    logger.info("=" * 50)
    logger.info("VoiceCreate starting...")
    logger.info("=" * 50)

    try:
        from utils.config_loader import load_environment_variables, load_system_config
        from utils.global_state import SystemStatus, initialize_global_state, update_system_status
        from utils.logging_setup import setup_logger

        load_environment_variables()
        config = load_system_config()
        logger = setup_logger("VoiceCreate")

        system_config = config.get("system", {})
        logger.info(f"[OK] Config loaded: {system_config.get('name', 'VoiceCreate')}")
        logger.info(f"     Version: {system_config.get('version', '0.1.0')}")

        _ensure_runtime_dirs(config)

        global_state = initialize_global_state(config)
        update_system_status(global_state, SystemStatus.IDLE, "ready")
        logger.info("[OK] Global state initialized")

        _load_speech_model(config, global_state, logger)
        _load_image_model(config, global_state, logger)
        _register_display_module(config, global_state, logger)

        logger.info("=" * 50)
        logger.info("VoiceCreate initialization complete.")
        logger.info("The main workflow loop is not started yet.")
        logger.info("=" * 50)
        logger.info("VoiceCreate initialization complete")
        return 0

    except FileNotFoundError as exc:
        logger.info(f"[ERROR] Required file not found: {exc}")
        return 1
    except Exception as exc:
        logger.info(f"[ERROR] VoiceCreate startup failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
