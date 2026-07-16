"""Intelligent prompt optimizer for VoiceCreate image generation."""

from __future__ import annotations

import logging

import random
import re
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class IntelligentPromptOptimizer:
    """Generate high-quality image prompts from parsed instruction data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.random = random.Random(self.config.get("seed", 42))
        self.default_enhancement_level = self.config.get("default_enhancement_level", "balanced")

        self.default_enhancers = {
            "quality": ["高清", "4K", "细节丰富", "专业摄影", "大师作品"],
            "lighting": ["自然光", "戏剧性灯光", "柔光", "黄金时刻光线"],
            "composition": ["中心构图", "三分法构图", "对称构图", "引导线构图"],
            "camera": ["广角镜头", "长焦镜头", "微距", "全景"],
        }
        self.style_enhancements = {
            "水墨画": ["笔触流畅", "墨色层次丰富", "留白艺术", "意境深远"],
            "油画": ["厚重笔触", "丰富色彩", "光影对比", "古典质感"],
            "水彩": ["透明色彩", "柔和边缘", "轻盈笔触", "纸张纹理"],
            "卡通": ["色彩鲜艳", "线条简洁", "可爱风格", "动画效果"],
            "写实": ["真实感强", "细节精确", "自然光影", "照片级质感"],
            "抽象": ["创意构图", "色彩碰撞", "几何图形", "现代艺术"],
            "赛博朋克": ["霓虹灯光", "未来城市", "高科技质感", "夜景氛围"],
            "中国风": ["传统美学", "东方意境", "雅致构图", "古典色彩"],
            "极简": ["留白充足", "简洁线条", "克制配色", "现代感"],
        }
        self.conflicting_style_terms = {
            "写实": ["卡通", "像素风", "像素", "动漫", "动画", "二次元"],
            "油画": ["卡通", "像素风", "像素", "动漫", "动画", "二次元"],
        }
        self.theme_enhancements = {
            "人物": ["表情生动", "姿态自然", "服装精致", "肖像特写"],
            "动物": ["毛发细腻", "眼神灵动", "姿态优美", "自然栖息"],
            "风景": ["壮丽景色", "自然和谐", "季节特征", "广阔视野"],
            "建筑": ["结构精确", "透视准确", "材质真实", "环境融合"],
        }

    def optimize(self, parsed_result: Dict[str, Any], enhancement_level: Optional[str] = None) -> str:
        level = enhancement_level or self.default_enhancement_level
        logger.info("优化提示词，增强级别: %s", level)

        base_prompt = self._build_base_prompt(parsed_result)
        prompt = self._add_theme_enhancements(base_prompt, parsed_result)
        prompt = self._add_style_enhancements(prompt, parsed_result)
        prompt = self._add_quality_enhancements(prompt, parsed_result, level)
        prompt = self._add_mood_enhancements(prompt, parsed_result)
        final_prompt = self._finalize_prompt(prompt)

        logger.info("优化完成: %s -> %s", base_prompt, final_prompt)
        return final_prompt

    def _build_base_prompt(self, parsed: Dict[str, Any]) -> str:
        parts: List[str] = []
        if parsed.get("quantifier"):
            parts.append(str(parsed["quantifier"]))
        parts.extend(str(item) for item in parsed.get("adjectives", []) if item)
        if parsed.get("subject"):
            parts.append(str(parsed["subject"]))
        if parsed.get("actions"):
            parts.append("正在" + str(parsed["actions"][0]))
        if parsed.get("scenes"):
            parts.append("在" + str(parsed["scenes"][0]) + "中")

        base = "".join(parts) or parsed.get("full_description") or parsed.get("raw_instruction") or "一张图片"
        return str(base)

    def _add_theme_enhancements(self, prompt: str, parsed: Dict[str, Any]) -> str:
        theme = self._detect_theme(parsed)
        if theme not in self.theme_enhancements:
            return prompt
        selected = self._sample(self.theme_enhancements[theme], 2)
        logger.info("添加主题增强: %s -> %s", theme, selected)
        return self._join(prompt, selected)

    def _add_style_enhancements(self, prompt: str, parsed: Dict[str, Any]) -> str:
        styles = parsed.get("styles", [])
        if not styles:
            return prompt

        style = styles[0]
        additions = [style if str(style).endswith("风格") else f"{style}风格"]
        candidates = self._filter_conflicting_style_terms(style, self.style_enhancements.get(style, []))
        additions.extend(self._sample(candidates, 1))
        logger.info("添加风格增强: %s -> %s", style, additions)
        return self._join(prompt, additions)

    def _add_quality_enhancements(self, prompt: str, parsed: Dict[str, Any], level: str) -> str:
        existing = list(parsed.get("qualities", []) or [])
        additions: List[str] = []

        if level == "minimal":
            if not existing:
                additions.append(self.default_enhancers["quality"][0])
        elif level == "maximal":
            if not existing:
                additions.append(self._choice(self.default_enhancers["quality"]))
            additions.append(self._choice(self.default_enhancers["lighting"]))
        else:
            if not existing:
                additions.append(self._choice(self.default_enhancers["quality"][:3]))
            additions.append(self._choice(self.default_enhancers[self._choice(["lighting", "composition", "camera"])]))

        if existing:
            additions = existing + additions
        logger.info("添加质量增强: %s", additions)
        return self._join(prompt, additions)

    def _add_mood_enhancements(self, prompt: str, parsed: Dict[str, Any]) -> str:
        mood = parsed.get("mood")
        mood_enhancements = {
            "温馨": ["温暖色调", "柔和氛围", "舒适感"],
            "恐怖": ["阴森氛围", "紧张感", "暗黑风格"],
            "欢乐": ["明亮色彩", "活泼氛围", "欢乐感"],
            "悲伤": ["冷色调", "忧郁氛围", "沉思感"],
            "神秘": ["神秘氛围", "奇幻感", "未知感"],
            "浪漫": ["浪漫氛围", "温馨感", "爱情主题"],
        }
        if mood not in mood_enhancements:
            return prompt
        enhancement = self._choice(mood_enhancements[mood])
        logger.info("添加情感增强: %s -> %s", mood, enhancement)
        return self._join(prompt, [enhancement])

    def _detect_theme(self, parsed: Dict[str, Any]) -> str:
        text = " ".join(
            str(item)
            for item in [parsed.get("subject", ""), *(parsed.get("objects", []) or []), *(parsed.get("scenes", []) or [])]
        )
        if any(word in text for word in ["人", "人物", "肖像", "脸", "面部", "家庭", "李白"]):
            return "人物"
        if any(word in text for word in ["狗", "猫", "鸟", "鱼", "动物", "宠物", "熊猫"]):
            return "动物"
        if any(word in text for word in ["风景", "山水", "海", "山", "森林", "自然", "日落", "草地", "雪地"]):
            return "风景"
        if any(word in text for word in ["建筑", "房子", "楼", "城市", "街道", "鬼屋"]):
            return "建筑"
        return "其他"

    def _finalize_prompt(self, prompt: str) -> str:
        prompt = re.sub(r"\s+", " ", prompt).strip()
        prompt = re.sub(r"，+", "，", prompt).strip("， ")
        if not prompt.endswith(("。", ".", "!", "！")):
            prompt += "。"
        return prompt

    def _join(self, prompt: str, additions: List[str]) -> str:
        parts = [prompt]
        for addition in additions:
            if addition and addition not in parts and addition not in prompt:
                parts.append(str(addition))
        return "，".join(parts)

    def _sample(self, items: List[str], count: int) -> List[str]:
        if not items:
            return []
        return self.random.sample(items, k=min(count, len(items)))

    def _filter_conflicting_style_terms(self, style: str, items: List[str]) -> List[str]:
        conflicts = self.conflicting_style_terms.get(str(style), [])
        if not conflicts:
            return items
        return [item for item in items if not any(conflict in item for conflict in conflicts)]

    def _choice(self, items: List[str]) -> str:
        return self.random.choice(items)
