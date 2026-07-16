"""Unified VoiceCreate workflow using the universal parser and optimizer."""

from __future__ import annotations

import logging

import time
from typing import Any, Dict, List, Optional

from PIL import Image

from parser.position_resolver import calculate_display_coordinates
from parser.universal_instruction_parser import UniversalInstructionParser
from prompt.intelligent_prompt_optimizer import IntelligentPromptOptimizer


logger = logging.getLogger(__name__)


class UnifiedVoiceCreateWorkflow:
    """End-to-end workflow for text or recognized voice instructions."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        image_generator: Any = None,
        display_manager: Any = None,
        dry_run: bool = False,
    ):
        self.config = config or {}
        root_config = self.config.get("voicecreate", self.config)
        self.parser_config = root_config.get("parser", {})
        self.optimizer_config = root_config.get("optimizer", {})
        self.workflow_config = root_config.get("workflow", {})
        self.min_confidence = float(self.parser_config.get("min_confidence", 0.3))
        self.dry_run = dry_run

        self.parser = UniversalInstructionParser(self.parser_config)
        self.optimizer = IntelligentPromptOptimizer(self.optimizer_config)
        self.image_generator = image_generator
        self.display_manager = display_manager

    def process_instruction(self, user_input: str, enhancement_level: Optional[str] = None) -> Dict[str, Any]:
        logger.info("开始处理指令: %s", user_input)
        started_at = time.time()
        result: Dict[str, Any] = {
            "success": False,
            "error": None,
            "original_input": user_input,
            "parsed_result": None,
            "optimized_prompt": None,
            "actual_prompt": None,
            "image": None,
            "save_path": None,
            "display_position": None,
            "generation_time": 0.0,
            "dry_run": self.dry_run,
        }

        try:
            parsed_result = self.parser.parse(user_input)
            result["parsed_result"] = parsed_result
            if parsed_result.get("confidence", 0.0) < self.min_confidence and not parsed_result.get("subject"):
                result["error"] = "无法理解指令内容"
                return result

            optimized_prompt = self.optimizer.optimize(parsed_result, enhancement_level)
            result["optimized_prompt"] = optimized_prompt
            result["actual_prompt"] = optimized_prompt
            logger.info("优化提示词: %s", optimized_prompt)

            if self.dry_run:
                result["success"] = True
                result["generation_time"] = time.time() - started_at
                return result

            generator = self.image_generator or self._init_image_generator()
            generation_result = generator.generate(optimized_prompt)
            image = self._extract_image(generation_result)
            result["image"] = image
            metadata = self._extract_metadata(generation_result)
            result["actual_prompt"] = metadata.get("actual_prompt") or metadata.get("prompt") or optimized_prompt
            result["save_path"] = metadata.get("save_path")
            if result["save_path"]:
                logger.info("图片已保存: %s", result["save_path"])

            position = parsed_result.get("position") or "中心"
            result["display_position"] = self._calculate_display_position(position, image.size)

            if self.workflow_config.get("enable_image_display", True):
                display_manager = self.display_manager or self._init_display_manager()
                display_success = self._display_image(display_manager, image, result["display_position"], parsed_result)
                result["display_success"] = display_success
                if not display_success:
                    logger.warning("图片显示失败，但生成成功")

            result["success"] = True
            result["generation_time"] = time.time() - started_at
            logger.info("指令处理完成")
        except Exception as exc:
            logger.error("处理指令时出错: %s", exc, exc_info=True)
            result["error"] = str(exc)
            result["generation_time"] = time.time() - started_at

        return result

    def batch_test(self, test_cases: List[str], enhancement_level: str = "balanced") -> Dict[str, Any]:
        results = []
        for index, test_case in enumerate(test_cases, 1):
            logger.info("测试用例 %s/%s: %s", index, len(test_cases), test_case)
            result = self.process_instruction(test_case, enhancement_level)
            results.append(
                {
                    "input": test_case,
                    "success": result["success"],
                    "parsed": result["parsed_result"],
                    "prompt": result["optimized_prompt"],
                    "actual_prompt": result["actual_prompt"],
                    "error": result["error"],
                }
            )

        success_count = sum(1 for item in results if item["success"])
        total_count = len(results)
        return {
            "total": total_count,
            "success": success_count,
            "success_rate": success_count / total_count if total_count else 0,
            "results": results,
        }

    def _init_image_generator(self) -> Any:
        from image.dreamlite_fixed import DreamLiteModel

        image_config = self.config.get("image") or self.config.get("iimage") or {}
        model = DreamLiteModel(image_config)
        model.load()
        self.image_generator = model
        logger.info("图片生成器初始化完成")
        return model

    def _init_display_manager(self) -> Any:
        from display.image_display import ImageDisplayManager

        self.display_manager = ImageDisplayManager()
        logger.info("显示管理器初始化完成")
        return self.display_manager

    def _calculate_display_position(self, position: str, image_size: tuple[int, int]) -> tuple[int, int]:
        width, height = image_size
        return calculate_display_coordinates(position, width, height)

    def _display_image(self, display_manager: Any, image: Image.Image, coords: tuple[int, int], parsed: Dict[str, Any]) -> bool:
        title = f"VoiceCreate: {parsed.get('subject') or '图片'}"
        if hasattr(display_manager, "display_image_at"):
            return bool(display_manager.display_image_at(image, coords[0], coords[1], title=title))
        if hasattr(display_manager, "display"):
            return bool(display_manager.display(image, coords[0], coords[1], title))
        raise AttributeError("display manager does not expose display_image_at or display")

    def _extract_image(self, generation_result: Any) -> Image.Image:
        image = None
        if isinstance(generation_result, dict):
            image = generation_result.get("image")
        elif hasattr(generation_result, "image"):
            image = generation_result.image
        elif hasattr(generation_result, "save"):
            image = generation_result
        if image is None:
            raise RuntimeError("图片生成失败：未返回图像")
        return image

    def _extract_metadata(self, generation_result: Any) -> Dict[str, Any]:
        if isinstance(generation_result, dict):
            metadata = generation_result.get("metadata") or {}
            return metadata if isinstance(metadata, dict) else {}
        metadata = getattr(generation_result, "metadata", {})
        return metadata if isinstance(metadata, dict) else {}
