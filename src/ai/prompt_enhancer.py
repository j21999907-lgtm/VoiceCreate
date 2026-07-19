"""Local Ollama-backed prompt enhancement."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Iterable, Optional

import requests


logger = logging.getLogger("VoiceCreate")


class AIPromptEnhancer:
    """Enhance image prompts with a local Ollama model."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        *,
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt or (
            "你是专业的文生图提示词编辑器。将中文描述改写成一行英文图像生成提示词。"
            "必须逐项保留主体、属性、穿戴、动作、关系、数量、场景和风格，不得省略、合并或改变含义。"
            "必须保留关键词列表中的每一项，并把它们自然地翻译到英文提示词中。"
            "可以补充构图、镜头、光线和画质，但补充内容不能与关键词冲突。"
            "输出 40 到 90 个英文单词，使用逗号分隔短语。"
            "不要输出标题、编号、JSON、Markdown、中文、引号或负面提示词。"
        )
        self.options = options or {"temperature": 0.4}
        self.api_url = "http://127.0.0.1:11434/api/chat"

    def enhance(self, prompt: str, required_keywords: Optional[Iterable[str]] = None) -> str:
        source_prompt = str(prompt or "").strip()
        if not source_prompt:
            return source_prompt

        max_attempts = 3
        keywords = list(dict.fromkeys(str(item).strip() for item in (required_keywords or []) if str(item).strip()))
        keyword_text = " | ".join(keywords) if keywords else source_prompt
        user_content = (
            f"原始提示词：{source_prompt}\n"
            f"必须逐项保留的关键词：{keyword_text}\n"
            "只输出一行英文提示词，并确保上述关键词对应的含义全部出现。"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
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
                # A refused local connection means Ollama is not running. Retrying
                # the same endpoint only delays the model's built-in prompt path.
                if isinstance(exc, requests.ConnectionError):
                    break
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
