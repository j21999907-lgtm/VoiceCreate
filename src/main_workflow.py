"""Shared VoiceCreate workflow for voice and keyboard input."""

from __future__ import annotations

import logging

logger = logging.getLogger("VoiceCreate")

import functools
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class WorkflowError(Exception):
    """Recoverable workflow failure with a user-facing message."""

    def __init__(self, message: str, *, fallback: Optional[str] = None) -> None:
        super().__init__(message)
        self.fallback = fallback


def handle_errors(default: Any = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that logs failures and returns a default value."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                logger.info(f"[ERROR] {func.__name__} failed: {exc}")
                traceback.print_exc()
                return default() if callable(default) else default

        return wrapper

    return decorator


@dataclass
class WorkflowContext:
    global_state: Dict[str, Any]
    image_generator: Any = None
    display: bool = True
    display_time: int = 5000
    generation_options: Dict[str, Any] = None
    progress_callback: Any = None
    status_callback: Any = None


def validate_audio(audio: Optional[bytes], min_bytes: int = 1600) -> bool:
    return bool(audio and isinstance(audio, (bytes, bytearray)) and len(audio) >= min_bytes)


def validate_text_input(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"valid": False, "message": "请输入语音指令或文字指令", "score": 0}
    if len(text) < 2:
        return {"valid": False, "message": "指令太短", "score": 0}

    hints = ["生成", "显示", "创建", "画", "绘制", "左", "右", "上", "下", "中"]
    score = sum(1 for hint in hints if hint in text)
    message = "语法看起来可用" if score else "可提交，但建议包含动作和位置"
    return {"valid": True, "message": message, "score": score}


def validate_command(command: Dict[str, Any]) -> bool:
    try:
        from speech.command_extractor import validate_speech_command

        return bool(validate_speech_command(command))
    except Exception:
        return bool(command and command.get("keywords"))


def fallback_audio_capture_error(error: Any = None) -> Dict[str, Any]:
    message = "音频采集失败，请切换到键盘输入"
    if error:
        message = f"{message}: {error}"
    return {"error": message, "fallback": "keyboard", "recoverable": True}


def prompt_keyboard_on_recognition_failure(error: Any = None) -> Dict[str, Any]:
    message = "语音识别失败，请使用键盘输入"
    if error:
        message = f"{message}: {error}"
    return {"error": message, "fallback": "keyboard", "recoverable": True}


def model_fallback_message(error: Any = None) -> Dict[str, Any]:
    message = "图片生成模型不可用，无法继续生成"
    if error:
        message = f"{message}: {error}"
    return {"error": message, "fallback": "mock", "recoverable": False}


def image_generation_error(error: Any = None) -> Dict[str, Any]:
    message = "图片生成失败"
    if error:
        message = f"{message}: {error}"
    return {"error": message, "recoverable": True}


def image_display_error(error: Any = None) -> Dict[str, Any]:
    message = "图片显示失败"
    if error:
        message = f"{message}: {error}"
    return {"error": message, "recoverable": True}


def suggest_retry(message: str) -> Dict[str, Any]:
    return {"error": message, "suggestion": "请重试，或切换到键盘输入。", "recoverable": True}


def process_user_input(
    input_method: str,
    input_data: Any = None,
    *,
    global_state: Optional[Dict[str, Any]] = None,
    image_generator: Any = None,
    display: bool = True,
    display_time: int = 5000,
    generation_options: Optional[Dict[str, Any]] = None,
    progress_callback: Any = None,
    status_callback: Any = None,
) -> Dict[str, Any]:
    """Process voice or keyboard input and run the full generation workflow."""
    context = WorkflowContext(
        global_state=global_state or {"modules": {}},
        image_generator=image_generator,
        display=display,
        display_time=display_time,
        generation_options=generation_options or {},
        progress_callback=progress_callback,
        status_callback=status_callback,
    )
    processor = VoiceCreateWorkflow(context)
    return processor.process(input_method, input_data)


class VoiceCreateWorkflow:
    """Integrated workflow used by the GUI and tests."""

    def __init__(self, context: WorkflowContext) -> None:
        self.context = context

    def _notify(self, stage: str, **details: Any) -> None:
        callback = self.context.status_callback
        if callback is None:
            return
        try:
            callback(stage, details)
        except Exception:
            logger.debug("Workflow status callback failed", exc_info=True)

    def process(self, input_method: str, input_data: Any = None) -> Dict[str, Any]:
        try:
            method = (input_method or "").lower().strip()
            if method == "voice":
                return self.process_voice_command(input_data)
            if method == "keyboard":
                text = str(input_data or "").strip()
                validation = validate_text_input(text)
                if not validation.get("valid"):
                    raise WorkflowError(validation.get("message", "键盘输入无效"), fallback="keyboard")
                return self._process_text(text, input_method="keyboard")
            raise WorkflowError("不支持的输入方式", fallback="keyboard")
        except WorkflowError as exc:
            return self.show_error(str(exc), fallback=exc.fallback)
        except Exception as exc:
            logger.info(f"[ERROR] process failed: {exc}")
            traceback.print_exc()
            return self.show_error(f"处理失败: {exc}", fallback="keyboard")

    def process_voice_command(self, audio: Optional[bytes] = None) -> Dict[str, Any]:
        try:
            audio = audio if isinstance(audio, (bytes, bytearray)) else self.record_audio()
            if not validate_audio(audio):
                raise WorkflowError("音频录制失败", fallback="keyboard")

            if not self.detect_speech(audio):
                return self.switch_to_keyboard_input("未检测到有效语音")

            text = self.recognize_speech(audio)
            if not text:
                raise WorkflowError("语音识别失败", fallback="keyboard")

            return self._process_text(text, input_method="voice")
        except WorkflowError as exc:
            return self.show_error(f"处理失败: {exc}", fallback=exc.fallback)
        except Exception as exc:
            logger.info(f"[ERROR] process_voice_command failed: {exc}")
            traceback.print_exc()
            return self.show_error(f"处理失败: {exc}", fallback="keyboard")

    @handle_errors(default=None)
    def record_audio(self) -> Optional[bytes]:
        from speech.audio_capture import capture_audio_chunk

        return capture_audio_chunk(duration_seconds=2.0)

    @handle_errors(default=False)
    def detect_speech(self, audio: bytes) -> bool:
        from speech.vad_processor import detect_speech_activity

        return bool(detect_speech_activity(audio))

    @handle_errors(default=None)
    def recognize_speech(self, audio: bytes) -> Optional[str]:
        from speech.speech_recognizer import recognize_speech

        return recognize_speech(audio, global_state=self.context.global_state)

    def parse_position(self, text: str, command: Optional[Dict[str, Any]] = None) -> str:
        if command and command.get("position"):
            return str(command["position"])
        from parser.position_resolver import identify_position_description

        return identify_position_description(text) or "中心"

    def generate_image(self, prompt: str, parsed: Dict[str, Any]) -> Any:
        generator = self.context.image_generator or self.context.global_state.get("modules", {}).get("image_generator")
        if generator is None:
            raise WorkflowError("图片生成器不可用", fallback="mock")

        options = self.context.generation_options or {}
        generator_default_size = int(getattr(generator, "default_size", 512) or 512)
        generator_steps = int(getattr(generator, "steps", 25) or 25)
        generator_guidance = float(getattr(generator, "guidance_scale", 7.5))
        generator_negative_prompt = getattr(generator, "negative_prompt", None)

        # Parser parameters describe a generic high-quality preset; they are not
        # user-selected generation settings.  Using them as runtime defaults turns
        # SD-Turbo's 1-4 step inference into a 35-step job and makes generation
        # appear to hang.  Explicit UI/API options still take precedence.
        width = int(options.get("width", generator_default_size))
        height = int(options.get("height", generator_default_size))
        steps = int(options.get("steps", generator_steps))
        guidance_scale = float(options.get("guidance_scale", generator_guidance))
        negative_prompt = options.get("negative_prompt", generator_negative_prompt)
        self._notify("generating", width=width, height=height, steps=steps)
        logger.info("[Workflow] Sending prompt to image generator:")
        logger.debug(f"  prompt: {prompt}")
        logger.debug(f"  negative_prompt: {negative_prompt}")
        logger.debug(f"  width={width} height={height} steps={steps} guidance_scale={guidance_scale}")
        logging.getLogger("VoiceCreateGUI").info(
            "Sending prompt to image generator: prompt=%s negative_prompt=%s width=%s height=%s steps=%s guidance_scale=%s",
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            guidance_scale,
        )

        result = generator.generate(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            progress_callback=self.context.progress_callback,
        )
        success = bool(getattr(result, "success", False))
        image = getattr(result, "image", None)
        if isinstance(result, dict):
            success = bool(result.get("success", image is not None))
            image = result.get("image")
        elif not hasattr(result, "success") and hasattr(result, "save"):
            success = True
            image = result

        metadata = self._result_metadata(result)
        if metadata.get("fallback") == "mock":
            logging.getLogger("VoiceCreateGUI").warning("图像生成回退至Mock模式")
            logging.getLogger("VoiceCreate").warning("图像生成回退至Mock模式")
            raise WorkflowError("图像生成回退至Mock模式", fallback="mock")

        if not success or image is None:
            error = getattr(result, "error_message", None)
            if isinstance(result, dict):
                error = result.get("error") or result.get("error_message")
            raise WorkflowError(error or "图片生成失败")
        return result

    def display_image(self, image: Any, position: str) -> Tuple[bool, Tuple[int, int]]:
        from display.image_display import display_image_at
        from parser.position_resolver import calculate_display_coordinates

        pil_image = self._result_image(image) or image
        width, height = getattr(pil_image, "size", (512, 512))
        coords = calculate_display_coordinates(position, int(width), int(height))
        if not self.context.display:
            return True, (int(coords[0]), int(coords[1]))

        success = display_image_at(pil_image, int(coords[0]), int(coords[1]), display_time=self.context.display_time)
        return bool(success), (int(coords[0]), int(coords[1]))

    def switch_to_keyboard_input(self, reason: str) -> Dict[str, Any]:
        return {"error": f"{reason}，已切换到键盘输入", "fallback": "keyboard", "recoverable": True}

    def show_error(self, message: str, fallback: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"[ERROR] {message}")
        result = suggest_retry(message)
        if fallback:
            result["fallback"] = fallback
        return result

    def _process_text(self, text: str, *, input_method: str) -> Dict[str, Any]:
        from parser.command_parser import parse_command_text
        from speech.command_extractor import extract_command_from_text

        self._notify("analyzing", input_method=input_method)
        command = extract_command_from_text(text)
        if not validate_command(command):
            raise WorkflowError("指令无效", fallback="keyboard")

        parsed = parse_command_text(text, command)
        position = self.parse_position(text, command)
        if not position:
            position = "中心"

        self._notify(
            "parsed",
            subject=command.get("subject") or command.get("object") or "",
            style=command.get("style") or "",
            position=position,
        )

        prompt = self._build_prompt(text, command, parsed)
        enhancer_enabled = bool(self._ai_prompt_enhancer_config().get("enabled", False))
        if enhancer_enabled:
            self._notify("enhancing")
        prompt = self._maybe_enhance_prompt(prompt, command.get("keywords", []))
        self._notify("prompt_ready", prompt=prompt, enhanced=enhancer_enabled)
        result = self.generate_image(prompt, parsed)
        image = self._result_image(result)
        if image is None:
            raise WorkflowError("图片生成失败")

        display_success, coords = self.display_image(image, position)
        if not display_success:
            raise WorkflowError("图片显示失败")

        if isinstance(parsed, dict):
            generation_metadata = self._result_metadata(result)
            actual_prompt = generation_metadata.get("actual_prompt") or generation_metadata.get("prompt") or prompt
            actual_negative_prompt = generation_metadata.get("actual_negative_prompt", generation_metadata.get("negative_prompt"))
            save_path = generation_metadata.get("save_path")
            if save_path:
                logger.info(f"[Workflow] Image saved to: {save_path}")
                logging.getLogger("VoiceCreateGUI").info("Image saved to: %s", save_path)
                self._notify("saved", path=save_path)
            parsed_position = parsed.get("position")
            if isinstance(parsed_position, dict):
                parsed_position["coordinates"] = coords
            actual_width, actual_height = getattr(image, "size", (None, None))
            parsed["model_prompt"] = actual_prompt
            parsed["actual_prompt"] = actual_prompt
            parsed["actual_negative_prompt"] = actual_negative_prompt
            parsed["save_path"] = save_path
            parsed["generation_metadata"] = generation_metadata
            parsed.setdefault("debug_info", {})
            parsed["debug_info"].update(
                {
                    "actual_model_prompt": actual_prompt,
                    "actual_negative_prompt": actual_negative_prompt,
                    "pipeline_call": generation_metadata.get("pipeline_call", {}),
                    "save_path": save_path,
                    "saved": bool(save_path),
                    "generation_options": self.context.generation_options or {},
                    "prompt_build_result": prompt,
                }
            )
            parsed.setdefault("display", {})
            parsed["display"].update(
                {
                    "coordinates": coords,
                    "actual_width": actual_width,
                    "actual_height": actual_height,
                }
            )

        self._notify("complete", position=position, coordinates=coords)
        return {
            "success": True,
            "input_method": input_method,
            "text": text,
            "command": command,
            "parsed": parsed,
            "position": position,
            "prompt": prompt,
            "actual_prompt": self._result_metadata(result).get("actual_prompt", prompt),
            "save_path": self._result_metadata(result).get("save_path"),
            "generation_metadata": self._result_metadata(result),
            "coords": coords,
            "result": result,
            "image": image,
        }

    def _build_prompt(self, text: str, command: Dict[str, Any], parsed: Dict[str, Any]) -> str:
        enhanced_prompt = command.get("enhanced_prompt") if isinstance(command, dict) else None
        if enhanced_prompt:
            return str(enhanced_prompt)
        parsed_prompt = parsed.get("enhanced_prompt") if isinstance(parsed, dict) else None
        if parsed_prompt:
            return str(parsed_prompt)
        scene = parsed.get("scene_description") if isinstance(parsed, dict) else None
        if scene:
            return str(scene)
        keywords = command.get("keywords", []) if isinstance(command, dict) else []
        if keywords:
            return "，".join(str(item) for item in keywords)
        return text

    def _maybe_enhance_prompt(self, prompt: str, required_keywords: Optional[list[str]] = None) -> str:
        enhancer_config = self._ai_prompt_enhancer_config()
        if not enhancer_config.get("enabled", False):
            return prompt

        try:
            from ai.prompt_enhancer import AIPromptEnhancer

            enhancer = AIPromptEnhancer(
                model=str(enhancer_config.get("model", "qwen2.5:7b")),
                system_prompt=enhancer_config.get("system_prompt"),
                options=enhancer_config.get("options"),
            )
            enhanced = enhancer.enhance(prompt, required_keywords=required_keywords)
            if enhanced != prompt:
                logger.info("[Workflow] AI prompt enhancer applied.")
            return enhanced
        except Exception as exc:
            logger.warning("[Workflow] AI prompt enhancer unavailable; using original prompt: %s", exc)
            return prompt

    def _ai_prompt_enhancer_config(self) -> Dict[str, Any]:
        options = self.context.generation_options or {}
        config = options.get("ai_prompt_enhancer") or options.get("prompt_enhancer") or {}
        if not isinstance(config, dict):
            config = {"enabled": bool(config)}

        enabled = bool(
            config.get(
                "enabled",
                options.get("enable_ai_prompt_enhancer", options.get("use_ai_prompt_enhancer", False)),
            )
        )
        merged = dict(config)
        merged["enabled"] = enabled
        return merged

    def _result_image(self, result: Any) -> Any:
        if isinstance(result, dict):
            return result.get("image")
        if hasattr(result, "image"):
            return result.image
        if hasattr(result, "save"):
            return result
        return None

    def _result_metadata(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            metadata = result.get("metadata") or {}
            return metadata if isinstance(metadata, dict) else {}
        metadata = getattr(result, "metadata", {})
        return metadata if isinstance(metadata, dict) else {}


if __name__ == "__main__":
    logger.info(process_user_input("keyboard", "在左上角生成一只猫", display=False))
