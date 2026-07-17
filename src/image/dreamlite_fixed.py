"""
Fixed DreamLite loader for VoiceCreate.

This module keeps the existing image/model_loader.py intact and provides a
clean loader for the custom DreamLitePipeline declared by model_index.json.
It always loads from local files and enables trust_remote_code for the local
DreamLite pipeline implementation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("VoiceCreate")


import os
import sys
import time
import traceback
import inspect
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image



os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelStatus(Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


@dataclass
class GenerationResult:
    success: bool
    image: Optional[Any] = None
    generation_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DreamLiteModel:
    """DreamLite model wrapper that supports the custom DreamLitePipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.model_path = self._resolve_model_path(
            self.config.get("model_path", "./models/sd-turbo")
        )
        self.model_type = str(self.config.get("model_type", "sd-turbo"))
        turbo_defaults = "turbo" in f"{self.model_type} {self.model_path}".lower()
        self.device = self.config.get("device", "cpu")
        self.default_size = int(self.config.get("default_size", 512 if turbo_defaults else 768))
        self.steps = int(self.config.get("steps", 4 if turbo_defaults else 30))
        self.guidance_scale = float(self.config.get("guidance_scale", 0.0 if turbo_defaults else 7.5))
        self.negative_prompt = self.config.get(
            "negative_prompt",
            "模糊, 像素化, 低分辨率, 水印, 文字, 丑陋, 畸形, 失真, 多手指, 残缺, 不对称, 涂抹, 混乱背景, 裁剪不当",
        )
        self.dtype_name = self.config.get("dtype", "float32")
        self.save_path = self._resolve_save_path(self.config.get("save_path") or self.config.get("image_dir") or "./generated_images")
        self.enable_save = bool(self.config.get("save_generated", True))
        self.save_metadata = bool(self.config.get("save_metadata", True))
        self.max_saved_images = int(self.config.get("max_saved_images", self.config.get("max_images", 100)))
        self.filename_template = self.config.get("filename_template", "{timestamp}_{id}_{prompt}.png")
        self.max_filename_length = int(self.config.get("max_filename_length", 100))
        self.pipe = None
        self.status = ModelStatus.NOT_LOADED
        self.last_error: Optional[str] = None
        self.using_mock = False
        self.missing_model_files = self._validate_model_files()
        if self.missing_model_files:
            self.last_error = f"Missing DreamLite model files: {', '.join(self.missing_model_files)}"
            logger.error("[DreamLite] Missing required model files: %s", self.missing_model_files)
            self.status = ModelStatus.ERROR
        self.is_mobile_pipeline = self._detect_mobile_model()
        self.is_turbo_pipeline = self._detect_turbo_model()
        self._ensure_save_directory()

    def _resolve_model_path(self, raw_path: str) -> str:
        path = Path(raw_path)
        candidates = [path] if path.is_absolute() else [
            PROJECT_ROOT / path,
            Path.cwd() / path,
            path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return str((PROJECT_ROOT / path).resolve())

    def _resolve_save_path(self, raw_path: str) -> str:
        path = Path(raw_path)
        if path.is_absolute():
            return str(path.resolve())
        return str((PROJECT_ROOT / path).resolve())

    def _ensure_save_directory(self) -> None:
        if not self.enable_save or not self.save_path:
            return
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        message = f"[DreamLite] Image save directory: {Path(self.save_path).resolve()}"
        logger.info(message)
        logging.getLogger("VoiceCreate").info(message)
        logging.getLogger("VoiceCreateGUI").info(message)

    def _validate_model_files(self) -> list[str]:
        model_dir = Path(self.model_path)
        missing: list[str] = []
        if not model_dir.exists():
            return [str(model_dir)]
        if not model_dir.is_dir():
            return [f"{model_dir}/"]

        model_index = model_dir / "model_index.json"
        if not model_index.is_file():
            missing.append("model_index.json")

        weight_extensions = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
        for subdir_name in ("unet", "vae"):
            subdir = model_dir / subdir_name
            if not subdir.is_dir():
                missing.append(f"{subdir_name}/")
                continue
            has_weights = any(path.is_file() and path.suffix.lower() in weight_extensions for path in subdir.rglob("*"))
            if not has_weights:
                missing.append(f"{subdir_name}/*{{.bin,.ckpt,.pt,.pth,.safetensors}}")

        return missing

    def _detect_mobile_model(self) -> bool:
        try:
            model_index_path = Path(self.model_path) / "model_index.json"
            data = json.loads(model_index_path.read_text(encoding="utf-8"))
            unet_info = data.get("unet", [])
            class_name = str(data.get("_class_name", ""))
            joined = " ".join(str(item) for item in unet_info)
            return "mobile" in joined.lower() or "Mobile" in class_name
        except Exception:
            return False

    def _detect_turbo_model(self) -> bool:
        try:
            model_index_path = Path(self.model_path) / "model_index.json"
            data = json.loads(model_index_path.read_text(encoding="utf-8"))
            identity = " ".join(
                (str(self.model_type), Path(self.model_path).name, str(data.get("_name_or_path", "")))
            ).lower()
            return "turbo" in identity
        except Exception:
            return "turbo" in f"{self.model_type} {self.model_path}".lower()

    def _torch_dtype(self, torch_module: Any) -> Any:
        if self.device == "cuda" and self.dtype_name == "float16":
            return torch_module.float16
        if self.dtype_name == "bfloat16":
            return torch_module.bfloat16
        return torch_module.float32

    def _ensure_device(self, torch_module: Any) -> str:
        requested = str(self.device or "cpu").lower()
        if requested == "cuda" and not torch_module.cuda.is_available():
            logger.info("[DreamLite] CUDA requested but unavailable; using CPU.")
            self.device = "cpu"
            return "cpu"
        self.device = requested
        return requested

    def load(self) -> bool:
        """Load DreamLite with trust_remote_code=True and local files only."""
        logger.info("=" * 60)
        logger.info("[DreamLite] Loading fixed DreamLite model")
        logger.info(f"[DreamLite] Model path: {self.model_path}")
        self.status = ModelStatus.LOADING

        self.missing_model_files = self._validate_model_files()
        if self.missing_model_files:
            self.last_error = f"Missing DreamLite model files: {', '.join(self.missing_model_files)}"
            logger.error("[DreamLite] Missing required model files: %s", self.missing_model_files)
            # Mock fallback disabled: fail fast when the real model cannot load.
            # return self._warn_and_load_mock()
            raise FileNotFoundError(self.last_error)

        try:
            from diffusers import DiffusionPipeline
            import torch
        except Exception as exc:
            self.last_error = f"Missing dependency: {exc}"
            logger.info(f"[DreamLite] {self.last_error}")
            # Mock fallback disabled: fail fast when dependencies are missing.
            # return self._warn_and_load_mock()
            raise RuntimeError(self.last_error) from exc

        try:
            import torchvision  # noqa: F401
        except Exception:
            logger.info("[DreamLite] torchvision is not installed; Qwen3VL processors may fail.")

        try:
            device = self._ensure_device(torch)
            dtype = self._torch_dtype(torch)

            logger.info(f"[DreamLite] PyTorch: {torch.__version__}")
            logger.info(f"[DreamLite] Device: {device}")
            logger.info(f"[DreamLite] Dtype: {dtype}")
            logger.info("[DreamLite] Loading local custom pipeline with trust_remote_code=True")

            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                trust_remote_code=True,
                local_files_only=True,
            )

            return self._finish_real_pipeline_load(device)

        except Exception as exc:
            if "DreamLitePipeline" in str(exc):
                logger.info("[DreamLite] Diffusers did not find DreamLitePipeline; trying local project pipeline.")
                try:
                    return self._load_local_dreamlite_pipeline(torch)
                except Exception as local_exc:
                    self.last_error = f"Local DreamLite pipeline load failed: {local_exc}"
                    if "Torchvision" in str(local_exc) or "torchvision" in str(local_exc):
                        self.last_error += " Install torchvision matching the local PyTorch build."
                    logger.info(f"[DreamLite] {self.last_error}")
                    traceback.print_exc()
                    # Mock fallback disabled: fail fast when the local pipeline cannot load.
                    # return self._warn_and_load_mock()
                    raise RuntimeError(self.last_error) from local_exc

            self.last_error = f"Real model load failed: {exc}"
            logger.info(f"[DreamLite] {self.last_error}")
            traceback.print_exc()
            # Mock fallback disabled: fail fast when the real model cannot load.
            # return self._warn_and_load_mock()
            raise RuntimeError(self.last_error) from exc

    def _load_local_dreamlite_pipeline(self, torch_module: Any) -> bool:
        """Load the bundled models/DreamLite package when diffusers cannot resolve it."""
        import importlib
        import importlib.util

        models_root = PROJECT_ROOT / "models"
        if str(models_root) not in sys.path:
            sys.path.insert(0, str(models_root))

        dreamlite_init = models_root / "DreamLite" / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "dreamlite",
            dreamlite_init,
            submodule_search_locations=[str(models_root / "DreamLite")],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create import spec for {dreamlite_init}")

        dreamlite_pkg = importlib.util.module_from_spec(spec)
        sys.modules["dreamlite"] = dreamlite_pkg
        spec.loader.exec_module(dreamlite_pkg)
        sys.modules.setdefault("DreamLite", dreamlite_pkg)

        for submodule in ("models", "pipelines"):
            module = importlib.import_module(f"dreamlite.{submodule}")
            sys.modules.setdefault(f"DreamLite.{submodule}", module)

        model_index_path = Path(self.model_path) / "model_index.json"
        model_index_data = json.loads(model_index_path.read_text(encoding="utf-8"))
        pipeline_name = str(model_index_data.get("_class_name", "")).strip()
        logger.info(f"[DreamLite] model_index.json _class_name: {pipeline_name or '<missing>'}")
        if not pipeline_name:
            raise ValueError(f"Missing _class_name in {model_index_path}")
        if not hasattr(dreamlite_pkg, pipeline_name):
            raise AttributeError(f"DreamLite package does not expose pipeline class: {pipeline_name}")

        DreamLitePipeline = getattr(dreamlite_pkg, pipeline_name)
        logger.info(f"[DreamLite] Using local pipeline class: {DreamLitePipeline.__name__}")
        device = self._ensure_device(torch_module)
        dtype = self._torch_dtype(torch_module)

        self.pipe = DreamLitePipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        return self._finish_real_pipeline_load(device)

    def _finish_real_pipeline_load(self, device: str) -> bool:
        self.pipe = self.pipe.to(device)

        if self.is_mobile_pipeline and hasattr(self.pipe, "unet") and hasattr(self.pipe.unet, "set_default_attn_processor"):
            try:
                self.pipe.unet.set_default_attn_processor()
                logger.info("[DreamLite] Mobile pipeline: default attention processor restored.")
            except Exception as exc:
                logger.info(f"[DreamLite] Mobile pipeline: could not restore default attention processor: {exc}")

        if not self.is_mobile_pipeline and device == "cuda" and hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
            logger.info("[DreamLite] Attention slicing enabled.")

        if not self.is_mobile_pipeline and self.config.get("enable_attention_slicing") and hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
            logger.info("[DreamLite] Attention slicing enabled by config.")
        elif self.is_mobile_pipeline and self.config.get("enable_attention_slicing"):
            logger.info("[DreamLite] Mobile pipeline: attention slicing disabled to avoid attention shape mismatch.")

        if self.config.get("enable_xformers_memory_efficient_attention") and hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("[DreamLite] xFormers memory efficient attention enabled.")
            except Exception as exc:
                logger.info(f"[DreamLite] xFormers attention unavailable: {exc}")

        self.status = ModelStatus.LOADED
        self.using_mock = False
        logger.info(f"Real DreamLite model loaded successfully. Device: {device}, Type: {self.model_type}")
        return True

    def _warn_and_load_mock(self) -> bool:
        logger.warning("[DreamLite] 真实模型加载失败，检查模型文件完整性")
        return self._load_mock()

    def _load_mock(self) -> bool:
        """Keep the app usable when local model loading fails."""
        self.pipe = None
        self.status = ModelStatus.LOADED
        self.using_mock = True
        logger.info("[DreamLite] Falling back to mock image generation.")
        return True

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        """Generate an image and return a GenerationResult wrapper."""
        if self.status != ModelStatus.LOADED:
            return GenerationResult(False, error_message="Model is not loaded.")

        start_time = time.time()
        width = self._bounded_int(kwargs.get("width", self.default_size), 64, 1024)
        height = self._bounded_int(kwargs.get("height", self.default_size), 64, 1024)
        steps = self._bounded_int(kwargs.get("num_inference_steps", kwargs.get("steps", self.steps)), 1, 50)
        if steps < 20 and not self.is_turbo_pipeline:
            steps = 20
            logger.info("Steps auto-adjusted to 20 for quality")
        guidance_scale = float(kwargs.get("guidance_scale", self.guidance_scale))
        negative_prompt = kwargs.get("negative_prompt")
        seed = kwargs.get("seed")
        progress_callback = kwargs.get("progress_callback")

        requested_call = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "model": self.model_type,
            "device": self.device,
            "mock": bool(self.pipe is None or self.using_mock),
        }
        self._log_prompt_payload("requested", requested_call)

        if self.pipe is None or self.using_mock:
            return self._generate_mock(
                prompt,
                width,
                height,
                steps,
                seed,
                start_time,
                progress_callback=progress_callback,
                prompt_payload=requested_call,
            )

        try:
            import torch

            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))

            call_kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
            }
            final_negative_prompt = self._merge_negative_prompts(negative_prompt)
            if final_negative_prompt and not self.is_mobile_pipeline and self._pipeline_supports_kwarg("negative_prompt"):
                call_kwargs["negative_prompt"] = final_negative_prompt
                logger.info(f"[DreamLite] Final negative prompt: {final_negative_prompt}")
            elif final_negative_prompt and self.is_mobile_pipeline:
                logger.info(f"[DreamLite] Final negative prompt ignored by mobile pipeline: {final_negative_prompt}")
                logger.info("[DreamLite] Mobile pipeline detected; negative_prompt is ignored to avoid CFG shape mismatch.")
            elif final_negative_prompt:
                logger.info(f"[DreamLite] Pipeline does not support negative_prompt; final negative prompt not sent: {final_negative_prompt}")
            if generator is not None:
                call_kwargs["generator"] = generator
            call_kwargs = self._filter_pipeline_kwargs(call_kwargs)
            actual_call = dict(requested_call)
            actual_call.update({key: value for key, value in call_kwargs.items() if key != "generator"})
            actual_call["negative_prompt_sent"] = call_kwargs.get("negative_prompt")
            actual_call["sent_keys"] = sorted(call_kwargs.keys())
            actual_call["mock"] = False
            self._log_prompt_payload("actual", actual_call)

            with torch.inference_mode():
                if progress_callback is not None:
                    try:
                        progress_callback(1, steps)
                    except Exception:
                        pass
                output = self.pipe(**call_kwargs)
                if progress_callback is not None:
                    try:
                        progress_callback(steps, steps)
                    except Exception:
                        pass

            image = output.images[0] if hasattr(output, "images") else output[0]
            image = self._ensure_exact_size(image, width, height, bool(kwargs.get("resize_to_requested", True)))
            image = self._upscale_if_enabled(image)
            generation_time = time.time() - start_time
            save_path = self.save_image(image, prompt, **self._metadata_kwargs(actual_call))
            return GenerationResult(
                success=True,
                image=image,
                generation_time=generation_time,
                metadata={
                    "prompt": prompt,
                    "actual_prompt": call_kwargs.get("prompt", prompt),
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "negative_prompt": negative_prompt,
                    "actual_negative_prompt": call_kwargs.get("negative_prompt"),
                    "pipeline_call": actual_call,
                    "save_path": save_path,
                    "saved": bool(save_path),
                    "seed": seed,
                    "model": self.model_type,
                    "device": self.device,
                    "mock": False,
                },
            )
        except Exception as exc:
            generation_time = time.time() - start_time
            self.last_error = f"Generation failed: {exc}"
            logger.info(f"[DreamLite] {self.last_error}")
            traceback.print_exc()
            if self.is_mobile_pipeline and self._is_attention_shape_error(exc):
                logger.info("[DreamLite] Mobile attention shape mismatch detected; using mock fallback for this generation.")
                return self._generate_mock(
                    prompt,
                    width,
                    height,
                    steps,
                    seed,
                    start_time,
                    progress_callback=progress_callback,
                    prompt_payload=requested_call,
                )
            return GenerationResult(
                success=False,
                generation_time=generation_time,
                error_message=self.last_error,
            )

    def generate_image_with_exact_size(self, prompt: str, width: int = 512, height: int = 512, **kwargs: Any) -> Optional[Any]:
        """Generate a PIL image and ensure the returned image matches the requested size."""
        logger.info(f"[DreamLite] Generating exact-size image: requested={width}x{height}")
        result = self.generate(prompt, width=width, height=height, **kwargs)
        if not result.success or result.image is None:
            logger.info(f"[DreamLite] Exact-size generation failed: {result.error_message}")
            return None
        return self._ensure_exact_size(result.image, width, height, bool(kwargs.get("resize_to_requested", True)))

    def _ensure_exact_size(self, image: Any, width: int, height: int, resize_to_requested: bool = True) -> Any:
        actual_width, actual_height = getattr(image, "size", (None, None))
        logger.info(f"[DreamLite] Image generated: actual={actual_width}x{actual_height}, requested={width}x{height}")
        if (actual_width, actual_height) == (width, height):
            return image

        logger.info(f"[DreamLite] Image size mismatch: requested={width}x{height}, actual={actual_width}x{actual_height}")
        if resize_to_requested and hasattr(image, "resize"):
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image = image.resize((width, height), resampling)
            logger.info(f"[DreamLite] Resized image to requested size: {width}x{height}")
        return image

    def _upscale_if_enabled(self, image: Any) -> Any:
        if self.config.get("enable_upscale") is False:
            return image
        if image is None or not hasattr(image, "resize"):
            return image

        actual_width, actual_height = getattr(image, "size", (None, None))
        if actual_width is None or actual_height is None:
            return image
        if actual_width >= 1024 and actual_height >= 1024:
            return image

        scale = 1.5
        target_width = max(1, int(round(actual_width * scale)))
        target_height = max(1, int(round(actual_height * scale)))
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        logger.info(
            "[DreamLite] Upscaling image with LANCZOS: %sx%s -> %sx%s",
            actual_width,
            actual_height,
            target_width,
            target_height,
        )
        return image.resize((target_width, target_height), resampling)

    def _generate_mock(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: Optional[int],
        start_time: float,
        progress_callback: Any = None,
        prompt_payload: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:
            return GenerationResult(
                success=False,
                generation_time=time.time() - start_time,
                error_message=f"Pillow is unavailable for mock generation: {exc}",
            )

        image = Image.new("RGB", (width, height), color=(38, 42, 54))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text = (prompt or "DreamLite mock")[:40]
        if progress_callback is not None:
            for step in range(1, steps + 1):
                try:
                    progress_callback(step, steps)
                except Exception:
                    pass
        draw.rectangle((8, 8, width - 8, height - 8), outline=(120, 170, 220), width=2)
        draw.text((16, max(16, height // 2 - 8)), text, fill=(235, 240, 255), font=font)
        draw.text((16, height - 24), "DreamLite mock fallback", fill=(160, 170, 190), font=font)
        actual_call = dict(prompt_payload or {})
        actual_call.update(
            {
                "prompt": prompt,
                "actual_prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
                "mock": True,
                "sent_keys": ["prompt", "width", "height", "steps"],
            }
        )
        self._log_prompt_payload("mock_actual", actual_call)
        save_path = self.save_image(image, prompt, **self._metadata_kwargs(actual_call))

        return GenerationResult(
            success=False,
            image=image,
            generation_time=time.time() - start_time,
            metadata={
                "prompt": prompt,
                "actual_prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": self.guidance_scale,
                "negative_prompt": self.negative_prompt,
                "actual_negative_prompt": actual_call.get("negative_prompt"),
                "pipeline_call": actual_call,
                "save_path": save_path,
                "saved": bool(save_path),
                "seed": seed,
                "model": self.model_type,
                "device": self.device,
                "mock": True,
                "fallback": "mock",
                "error": "Real model not available, using placeholder",
                "load_error": self.last_error,
            },
        )

    @staticmethod
    def _bounded_int(value: Any, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = minimum
        return max(minimum, min(maximum, parsed))

    def _merge_negative_prompts(self, custom_negative_prompt: Any = None) -> str:
        prompts = [str(item).strip() for item in (self.negative_prompt, custom_negative_prompt) if str(item or "").strip()]
        return ", ".join(prompts)

    def _pipeline_supports_kwarg(self, kwarg_name: str) -> bool:
        try:
            signature = inspect.signature(self.pipe.__call__)
        except Exception:
            return True
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return True
        return kwarg_name in signature.parameters

    def _filter_pipeline_kwargs(self, call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Drop kwargs unsupported by the active custom DreamLite pipeline."""
        try:
            signature = inspect.signature(self.pipe.__call__)
        except Exception:
            return call_kwargs

        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return call_kwargs

        supported = set(signature.parameters)
        filtered = {key: value for key, value in call_kwargs.items() if key in supported}
        dropped = sorted(set(call_kwargs) - set(filtered))
        if dropped:
            logger.info(f"[DreamLite] Dropped unsupported pipeline kwargs: {dropped}")
        return filtered

    def generate_and_save(self, prompt: str, **kwargs: Any) -> tuple[Optional[Any], Optional[str]]:
        result = self.generate(prompt, **kwargs)
        if not result.success or result.image is None:
            logger.info(f"[DreamLite] Image generation failed; nothing saved: {result.error_message}")
            return None, None
        return result.image, result.metadata.get("save_path")

    def save_image(self, image: Any, prompt: str, **kwargs: Any) -> Optional[str]:
        if not self.enable_save or not self.save_path:
            logger.info("[DreamLite] Image saving is disabled.")
            return None
        if image is None or not hasattr(image, "save"):
            logger.info("[DreamLite] Image saving skipped: invalid image object.")
            return None

        try:
            self._ensure_save_directory()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            safe_prompt = self._create_safe_filename(prompt)
            filename = self.filename_template.format(timestamp=timestamp, id=unique_id, prompt=safe_prompt)
            filename = self._create_safe_filename(filename, max_length=self.max_filename_length)
            if not filename.lower().endswith(".png"):
                filename += ".png"

            save_path = Path(self.save_path) / filename
            image.save(save_path, format="PNG")
            if self.save_metadata:
                self._save_metadata(str(save_path), prompt, **kwargs)
            self._enforce_save_limit()

            size = save_path.stat().st_size
            message = (
                f"[DreamLite] Image saved: {save_path.resolve()}\n"
                f"  file_size={size} bytes image_size={getattr(image, 'size', None)}"
            )
            logger.info(message)
            logging.getLogger("VoiceCreate").info(message)
            logging.getLogger("VoiceCreateGUI").info(message)
            return str(save_path.resolve())
        except Exception as exc:
            message = f"[DreamLite] Failed to save image: {exc}"
            logger.info(message)
            logging.getLogger("VoiceCreate").error(message, exc_info=True)
            logging.getLogger("VoiceCreateGUI").error(message, exc_info=True)
            return None

    def _save_metadata(self, image_path: str, prompt: str, **kwargs: Any) -> None:
        metadata_path = Path(image_path).with_suffix(".json")
        metadata = {
            "prompt": prompt,
            "parameters": self._json_safe(kwargs),
            "generated_at": datetime.now().isoformat(),
            "image_path": str(Path(image_path).resolve()),
        }
        try:
            with metadata_path.open("w", encoding="utf-8") as file:
                json.dump(metadata, file, ensure_ascii=False, indent=2)
            logger.info(f"[DreamLite] Metadata saved: {metadata_path.resolve()}")
        except Exception as exc:
            logger.info(f"[DreamLite] Metadata save failed: {exc}")

    def _enforce_save_limit(self) -> None:
        if self.max_saved_images <= 0:
            return
        image_paths = sorted(Path(self.save_path).glob("*.png"), key=lambda path: path.stat().st_mtime)
        extra_count = max(0, len(image_paths) - self.max_saved_images)
        for path in image_paths[:extra_count]:
            try:
                path.unlink(missing_ok=True)
                path.with_suffix(".json").unlink(missing_ok=True)
                logger.info(f"[DreamLite] Removed old saved image: {path}")
            except Exception as exc:
                logger.info(f"[DreamLite] Failed to remove old image {path}: {exc}")

    def get_save_directory(self) -> Optional[str]:
        return str(Path(self.save_path).resolve()) if self.save_path else None

    def list_saved_images(self) -> list[str]:
        if not self.save_path or not Path(self.save_path).exists():
            return []
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        files = []
        for pattern in patterns:
            files.extend(Path(self.save_path).glob(pattern))
        return [str(path.resolve()) for path in sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)]

    def _create_safe_filename(self, text: str, max_length: int = 50) -> str:
        safe_text = re.sub(r'[<>:"/\\|?*\r\n\t]', "", str(text or "untitled"))
        safe_text = re.sub(r"\s+", "_", safe_text.strip())
        safe_text = safe_text.strip("._ ")
        return (safe_text or "untitled")[:max_length]

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items() if key != "generator"}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _metadata_kwargs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in payload.items() if key != "prompt"}

    def _log_prompt_payload(self, stage: str, payload: Dict[str, Any]) -> None:
        prompt = payload.get("prompt", "")
        negative_prompt = payload.get("negative_prompt")
        message = (
            f"[DreamLite] Prompt payload ({stage})\n"
            f"  prompt: {prompt}\n"
            f"  negative_prompt: {negative_prompt}\n"
            f"  width={payload.get('width')} height={payload.get('height')} "
            f"steps={payload.get('num_inference_steps', payload.get('steps'))} "
            f"guidance_scale={payload.get('guidance_scale')} seed={payload.get('seed')}\n"
            f"  sent_keys={payload.get('sent_keys')}"
        )
        logger.info(message)
        logging.getLogger("VoiceCreateGUI").info(message)
        logging.getLogger("VoiceCreate").info(message)

    @staticmethod
    def _is_attention_shape_error(exc: Exception) -> bool:
        message = str(exc)
        return "Expected size for first two dimensions of batch2 tensor" in message

    def generate_high_quality_image(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a high-quality PIL image with stronger defaults."""
        result = self.generate(
            prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt or self.negative_prompt,
            **kwargs,
        )
        if not result.success or result.image is None:
            raise RuntimeError(result.error_message or "High quality image generation failed.")
        return result.image


def initialize_dreamlite_model(model_config: Dict[str, Any]) -> Optional[DreamLiteModel]:
    model = DreamLiteModel(model_config)
    if model.load():
        return model
    return None


def register_model_to_global_state(global_state: Dict[str, Any], model_obj: DreamLiteModel) -> None:
    if not global_state or "modules" not in global_state:
        return

    lock = global_state.get("lock")
    if lock:
        with lock:
            global_state["modules"]["image_generator"] = model_obj
    else:
        global_state["modules"]["image_generator"] = model_obj
    logger.info("[DreamLite] Image generator registered in global state.")


def test_fixed_dreamlite() -> bool:
    config = {
        "model_path": "./models/DreamLite-base",
        "device": "cpu",
        "default_size": 256,
        "steps": 25,
        "guidance_scale": 7.5,
        "negative_prompt": "模糊, 像素化, 低质量, 水印, 文字, 丑陋, 畸形, 失真",
        "dtype": "float32",
    }

    model = DreamLiteModel(config)
    assert model.load() is True, "Model should load or fall back to mock mode."

    result = model.generate("a cat", width=128, height=128, steps=2)
    assert result.success, result.error_message or "Image generation failed."
    assert result.image is not None, "Generation should return an image."
    assert hasattr(result.image, "save"), "Result image should behave like a PIL Image."
    high_quality_image = model.generate_high_quality_image("a cat", width=128, height=128, num_inference_steps=25)
    assert high_quality_image.size == (128, 128)

    output_path = PROJECT_ROOT / "temp" / "dreamlite_fixed_test.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(output_path)
    logger.info(f"[DreamLite] Test image saved to: {output_path}")
    logger.info("[DreamLite] Fixed DreamLite test passed.")
    return True


if __name__ == "__main__":
    test_fixed_dreamlite()
