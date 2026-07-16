"""Local Ollama-backed prompt enhancement."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

import requests


logger = logging.getLogger("VoiceCreate")


class AIPromptEnhancer:
    """Enhance image prompts with a local Ollama model."""

    def __init__(
        self,
        model: str = "qwen2:7b",
        *,
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt or (
            "你是专业的文生图提示词导演。任务是把用户的简短中文描述改写成一条英文图像生成提示词。"
            "必须保留用户明确说出的主体、动作、风格、位置、数量和场景；不得询问更多信息，不得拒绝，不得解释。"
            "当用户描述很短时，合理补全环境、构图、镜头、光线、材质、细节和画质词。"
            "输出必须是一行英文 prompt，40 到 90 个英文词，逗号分隔短语。"
            "不要输出标题、编号、JSON、Markdown、中文、引号或负面提示词。"
        )
        self.options = options or {"temperature": 0.4}
        self.api_url = "http://127.0.0.1:11434/api/chat"

    def enhance(self, prompt: str) -> str:
        source_prompt = str(prompt or "").strip()
        if not source_prompt:
            return source_prompt

        max_attempts = 3
        system_content = (
            "你是一个AI绘画提示词生成器。请将用户的中文描述转化为一段自然流畅的英文图像生成提示词。要求：\n"
            "- 输出一个完整的句子，而不是列表或标签。\n"
            "- 自然地包含主体、环境、光线、风格、画质等信息。\n"
            "- 使用英文，长度在40-80词之间。\n"
            "- 只输出提示词本身，不要任何解释、格式标记或标点之外的符号。\n"
            "- 如果用户描述很简单，请合理增加细节（如颜色、材质、氛围）。"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "将以下描述扩展成一个丰富的图像生成提示词：" + source_prompt},
            ],
            "stream": False,
            "options": self.options,
        }
        for attempt in range(1, max_attempts + 1):
            try:
                os.environ["NO_PROXY"] = "127.0.0.1,localhost"
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()

                response_data = json.loads(response.text)
                enhanced = self._clean_prompt(self._extract_content(response_data))
                if not enhanced:
                    logger.warning("[AIPromptEnhancer] Ollama API returned empty prompt; using original.")
                    return self._ensure_quality_suffix(source_prompt)
                if len(enhanced) < 20:
                    logger.warning(
                        "[AIPromptEnhancer] Ollama API returned too-short prompt; length=%s, content=%s; using original.",
                        len(enhanced),
                        enhanced,
                    )
                    return self._ensure_quality_suffix(source_prompt)
                logger.info("[AIPromptEnhancer] Prompt enhanced with Ollama model: %s", self.model)
                return self._ensure_quality_suffix(enhanced)
            except Exception as exc:
                status_code = self._extract_status_code(exc)
                logger.warning(
                    "[AIPromptEnhancer] Ollama API enhancement failed: attempt=%s/%s, type=%s, status_code=%s, message=%s",
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    status_code,
                    exc,
                )
                if attempt < max_attempts:
                    time.sleep(1)

        logger.warning("[AIPromptEnhancer] All Ollama API enhancement attempts failed; using original prompt.")
        return self._ensure_quality_suffix(source_prompt)

    @staticmethod
    def _extract_status_code(exc: Exception) -> Optional[int]:
        status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None) or getattr(response, "status", None)
        try:
            return int(status_code) if status_code is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_content(response: Any) -> str:
        if isinstance(response, dict):
            message = response.get("message") or {}
            if isinstance(message, dict):
                return str(message.get("content", ""))
            return str(getattr(message, "content", ""))

        message = getattr(response, "message", None)
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(getattr(message, "content", ""))

    @staticmethod
    def _clean_prompt(content: str) -> str:
        prompt = str(content or "").strip()
        if not prompt:
            return ""

        prompt = prompt.replace("\r", "\n")
        lines = [line.strip() for line in prompt.split("\n") if line.strip()]
        prompt = next(
            (
                line
                for line in lines
                if not re.match(r"^(input|输入|output|输出|prompt|提示词|final)\s*[:：]?\s*$", line, re.I)
            ),
            "",
        )
        prompt = re.sub(r"^(output|输出|prompt|提示词|final prompt|final)\s*[:：]\s*", "", prompt, flags=re.I)
        prompt = prompt.strip().strip("\"'“”‘’`")
        prompt = re.sub(r"\s+", " ", prompt)
        prompt = re.sub(r"\s*,\s*", ", ", prompt)
        prompt = prompt.strip("，,;；.。 ")
        return prompt

    @staticmethod
    def _ensure_quality_suffix(prompt: str) -> str:
        quality_suffix = ", high resolution, sharp focus, masterpiece"
        cleaned = str(prompt or "").strip()
        if not cleaned:
            return cleaned
        if quality_suffix.strip(", ").lower() in cleaned.lower():
            return cleaned
        return f"{cleaned.rstrip(' ,')}{quality_suffix}"
