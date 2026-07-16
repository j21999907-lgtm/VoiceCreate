"""Universal natural-language instruction parser for VoiceCreate."""

from __future__ import annotations

import logging

import re
from typing import Any, Dict, List, Optional

try:
    import jieba
    import jieba.posseg as pseg
except ImportError:  # pragma: no cover - exercised when jieba is absent.
    jieba = None
    pseg = None


logger = logging.getLogger(__name__)


class UniversalInstructionParser:
    """Parse free-form Chinese instructions into image-generation intent."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enable_pos_tagging = bool(self.config.get("enable_pos_tagging", True))
        self.max_instruction_length = int(self.config.get("max_instruction_length", 200))

        self.position_patterns = {
            "左上角": ["左上角", "左上", "左上方", "左上边", "屏幕左上角", "画面左上角"],
            "右上角": ["右上角", "右上", "右上方", "右上边", "屏幕右上角", "画面右上角"],
            "左下角": ["左下角", "左下", "左下方", "左下边", "屏幕左下角", "画面左下角"],
            "右下角": ["右下角", "右下", "右下方", "右下边", "屏幕右下角", "画面右下角"],
            "中心": ["中心", "中间", "中央", "正中", "正中央", "屏幕中间", "画面中心", "画面中央"],
            "顶部": ["顶部", "上方", "上面", "屏幕顶部"],
            "底部": ["底部", "下方", "下面", "屏幕底部"],
            "左边": ["左边", "左侧", "左方", "屏幕左边"],
            "右边": ["右边", "右侧", "右方", "屏幕右边"],
        }
        self.action_verbs = {
            "生成", "创建", "制作", "绘制", "显示", "展示", "呈现", "产生",
            "做一个", "来一个", "来一张", "搞一个", "搞一张", "弄一个", "弄一张", "画",
        }
        self.quantifiers = {"一个", "一只", "一张", "一幅", "一件", "一批", "一些", "很多"}
        self.stop_words = {"的", "了", "在", "和", "与", "及", "以及", "而且", "然后", "请", "帮我", "给我", "用"}
        self.scene_markers = {"草地", "雪地", "森林", "公园", "沙发", "城市", "夜景", "山脉", "海边", "天空", "街道", "房间", "家庭", "鬼屋", "日落"}

        self._init_jieba_dict()

    def _init_jieba_dict(self) -> None:
        if jieba is None:
            return

        custom_words = [
            "水墨画", "油画", "水彩画", "素描", "卡通", "动漫", "像素风", "赛博朋克",
            "中国风", "写实", "抽象", "极简", "二次元", "高清", "超清", "细节丰富",
            "黄金时刻", "清晨光线", "科幻城市", "城市夜景",
            *self.scene_markers,
        ]
        for word in custom_words:
            jieba.add_word(word, freq=1000, tag="n")

    def parse(self, instruction: str) -> Dict[str, Any]:
        if not instruction or not instruction.strip():
            return self._create_empty_result()

        original = instruction.strip()[: self.max_instruction_length]
        working = original
        logger.info("解析指令: %s", original)

        result = self._create_empty_result()
        result["raw_instruction"] = original
        result["debug_info"] = {"original_instruction": original}

        position_info = self._extract_position(working)
        if position_info:
            result["position"] = position_info["position"]
            result["position_keywords"] = position_info["keywords"]
            working = position_info["cleaned_text"]

        action_info = self._extract_action(working)
        if action_info:
            result["action_verb"] = action_info["action"]
            working = action_info["cleaned_text"]

        quantifier_info = self._extract_quantifier(working)
        if quantifier_info:
            result["quantifier"] = quantifier_info["quantifier"]
            working = quantifier_info["cleaned_text"]

        working = self._normalize_text(working)
        semantic_info = self._analyze_semantics(working)
        result.update(semantic_info)
        result["full_description"] = self._build_full_description(result)
        result["confidence"] = self._calculate_confidence(result)
        result["debug_info"].update(
            {
                "cleaned_for_semantics": working,
                "removed_position": result["position_keywords"],
                "removed_action": result["action_verb"],
                "removed_quantifier": result["quantifier"],
                "summary": self._summarize_result(result),
            }
        )

        logger.info("解析结果: %s", result["debug_info"]["summary"])
        return result

    def _extract_position(self, text: str) -> Optional[Dict[str, Any]]:
        for position, variants in self.position_patterns.items():
            for variant in sorted(variants, key=len, reverse=True):
                if variant in text:
                    return {
                        "position": position,
                        "keywords": [variant],
                        "cleaned_text": text.replace(variant, " "),
                    }
        return None

    def _extract_action(self, text: str) -> Optional[Dict[str, str]]:
        for action in sorted(self.action_verbs, key=len, reverse=True):
            if action in text:
                return {"action": action, "cleaned_text": text.replace(action, " ")}
        return None

    def _extract_quantifier(self, text: str) -> Optional[Dict[str, str]]:
        for quantifier in sorted(self.quantifiers, key=len, reverse=True):
            if quantifier in text:
                return {"quantifier": quantifier, "cleaned_text": text.replace(quantifier, " ")}
        return None

    def _analyze_semantics(self, text: str) -> Dict[str, Any]:
        result = self._create_empty_semantic_result()
        if not text:
            return result

        tokens = self._segment(text)
        result["parsed_components"]["tokens"] = tokens

        for word, flag in tokens:
            if not word or word in self.stop_words:
                continue
            if word in result["styles"] or word in result["qualities"]:
                continue
            if word in self.scene_markers or flag.startswith("s"):
                self._append_unique(result["scenes"], word)
                continue
            if flag.startswith("a"):
                self._append_unique(result["adjectives"], word)
                continue
            if flag.startswith("v"):
                self._append_unique(result["actions"], word)
                continue
            if flag.startswith("n") or flag in {"x", "eng"}:
                if not result["subject"]:
                    result["subject"] = word
                self._append_unique(result["objects"], word)

        for style in self._detect_styles(text):
            self._append_unique(result["styles"], style)
        for quality in self._detect_qualities(text):
            self._append_unique(result["qualities"], quality)
        for scene in self._detect_scene_markers(text):
            self._append_unique(result["scenes"], scene)
        result["mood"] = self._detect_mood(text)

        if not result["subject"] and result["objects"]:
            result["subject"] = result["objects"][0]
        if not result["subject"] and text:
            result["subject"] = self._fallback_subject(text)

        return result

    def _segment(self, text: str) -> List[tuple[str, str]]:
        if pseg is not None and self.enable_pos_tagging:
            return [(word.strip(), flag) for word, flag in pseg.lcut(text) if word.strip()]

        parts = [part for part in re.split(r"[\s,，。.!！?？;；:：、]+", text) if part]
        return [(part, "n") for part in parts]

    def _detect_styles(self, text: str) -> List[str]:
        style_keywords = {
            "水墨画": ["水墨", "水墨画", "国画", "山水画"],
            "油画": ["油画", "油彩"],
            "水彩": ["水彩", "水彩画"],
            "素描": ["素描", "速写"],
            "卡通": ["卡通", "动画", "动漫", "二次元"],
            "像素风": ["像素", "像素风"],
            "赛博朋克": ["赛博", "赛博朋克", "cyberpunk"],
            "中国风": ["中国风", "古风", "传统", "古典"],
            "写实": ["写实", "真实", "逼真", "照片"],
            "抽象": ["抽象", "抽象艺术"],
            "极简": ["极简", "简约", "简洁"],
        }
        return self._detect_by_keywords(text, style_keywords)

    def _detect_qualities(self, text: str) -> List[str]:
        quality_keywords = {
            "高清": ["高清", "高清晰", "清晰"],
            "4K": ["4K", "4k", "超清"],
            "细节丰富": ["细节", "细腻", "精细"],
            "专业": ["专业", "大师", "大师级"],
            "精美": ["精美", "漂亮", "美丽", "好看"],
        }
        return self._detect_by_keywords(text, quality_keywords)

    def _detect_scene_markers(self, text: str) -> List[str]:
        return [scene for scene in sorted(self.scene_markers, key=len, reverse=True) if scene in text]

    def _detect_mood(self, text: str) -> Optional[str]:
        mood_keywords = {
            "温馨": ["温馨", "温暖", "柔和", "舒适"],
            "恐怖": ["恐怖", "可怕", "惊悚", "诡异"],
            "欢乐": ["欢乐", "快乐", "开心", "高兴"],
            "悲伤": ["悲伤", "忧郁", "哀伤", "凄凉"],
            "神秘": ["神秘", "神秘感", "奇幻", "魔幻"],
            "浪漫": ["浪漫", "浪漫的", "爱情", "爱"],
        }
        detected = self._detect_by_keywords(text, mood_keywords)
        return detected[0] if detected else None

    def _build_full_description(self, result: Dict[str, Any]) -> str:
        parts: List[str] = []
        if result.get("quantifier"):
            parts.append(result["quantifier"])
        parts.extend(result.get("adjectives") or [])
        if result.get("subject"):
            parts.append(result["subject"])
        if result.get("actions"):
            parts.append("正在" + result["actions"][0])
        if result.get("scenes"):
            parts.append("在" + result["scenes"][0] + "中")

        description = "".join(parts) or result.get("subject") or ""
        return re.sub(r"\s+", "", description)

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        confidence = 1.0
        if not result.get("subject"):
            confidence *= 0.6
        if len(result.get("full_description", "")) < 2:
            confidence *= 0.7
        if result.get("position"):
            confidence += 0.05
        return round(min(confidence, 1.0), 2)

    def _summarize_result(self, result: Dict[str, Any]) -> str:
        summary = []
        if result.get("position"):
            summary.append(f"位置: {result['position']}")
        if result.get("subject"):
            summary.append(f"主体: {result['subject']}")
        if result.get("adjectives"):
            summary.append(f"形容词: {result['adjectives']}")
        if result.get("styles"):
            summary.append(f"风格: {result['styles']}")
        return "; ".join(summary)

    def _create_empty_result(self) -> Dict[str, Any]:
        return {
            "raw_instruction": "",
            "position": None,
            "position_keywords": [],
            "action_verb": None,
            "quantifier": None,
            "subject": None,
            "adjectives": [],
            "objects": [],
            "scenes": [],
            "styles": [],
            "qualities": [],
            "actions": [],
            "mood": None,
            "full_description": "",
            "confidence": 0.0,
            "parsed_components": {},
            "debug_info": {},
        }

    def _create_empty_semantic_result(self) -> Dict[str, Any]:
        return {
            "subject": None,
            "adjectives": [],
            "objects": [],
            "scenes": [],
            "styles": [],
            "qualities": [],
            "actions": [],
            "mood": None,
            "parsed_components": {},
        }

    def _fallback_subject(self, text: str) -> str:
        cleaned = self._normalize_text(text)
        for style in self._detect_styles(cleaned):
            cleaned = cleaned.replace(style, "")
        for quality in self._detect_qualities(cleaned):
            cleaned = cleaned.replace(quality, "")
        return cleaned or text

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"[\s,，。.!！?？;；:：、]+", " ", text).strip()
        for word in sorted(self.stop_words, key=len, reverse=True):
            if len(word) > 1:
                text = text.replace(word, " ")
        return re.sub(r"\s+", " ", text).strip()

    def _detect_by_keywords(self, text: str, keyword_map: Dict[str, List[str]]) -> List[str]:
        detected: List[str] = []
        lower_text = text.lower()
        for label, keywords in keyword_map.items():
            if any(keyword.lower() in lower_text for keyword in keywords):
                detected.append(label)
        return detected

    @staticmethod
    def _append_unique(items: List[str], value: str) -> None:
        if value and value not in items:
            items.append(value)
