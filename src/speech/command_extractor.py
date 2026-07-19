"""Chinese command extraction for VoiceCreate."""

from __future__ import annotations

import logging

logger = logging.getLogger("VoiceCreate")


import re
import traceback
from typing import Any, Dict, List, Optional

from src.shared.position_aliases import POSITION_ALIASES


try:
    from keywords.keyword_classifier import KeywordClassifier
    from keywords.keyword_dictionary import ChineseKeywordDictionary
    from keywords.phrase_expander import PhraseExpander
    from keywords.stopwords_cn import STOPWORDS_CN
except ModuleNotFoundError:
    from src.keywords.keyword_classifier import KeywordClassifier
    from src.keywords.keyword_dictionary import ChineseKeywordDictionary
    from src.keywords.phrase_expander import PhraseExpander
    from src.keywords.stopwords_cn import STOPWORDS_CN

try:
    import jieba
except ImportError:
    jieba = None


ACTION_WORDS = ["生成", "显示", "创建", "画", "绘制", "制作", "做", "搞", "放", "呈现", "展示"]
PUNCTUATION_RE = re.compile(r"[\s,，。.!！?？;；:：、\"'“”‘’（）()【】\[\]{}]+")
SINGLE_CHAR_KEEP = set("红黄蓝绿黑白金银猫狗鸟鱼花树山水海车人吃跑飞跳")
QUALITY_TERMS = ["高清", "4K", "细节丰富", "专业摄影", "精美", "细腻"]
DESCRIPTIVE_PREFIXES = ("戴", "穿", "拿", "抱", "背", "举", "撑", "骑", "趴", "站", "坐", "躺", "看", "望", "吃", "喝", "追", "拉", "推")


class CommandExtractor:
    """Extract position, keywords, categories and enhanced prompt from Chinese text."""

    def __init__(self) -> None:
        self.dictionary = ChineseKeywordDictionary()
        self.classifier = KeywordClassifier()
        self.expander = PhraseExpander()
        self.position_keywords = POSITION_ALIASES
        self.action_keywords = ACTION_WORDS
        self.use_jieba = jieba is not None
        self.dictionary_terms = sorted(self.dictionary.all_terms(), key=len, reverse=True)
        if self.use_jieba:
            self._register_jieba_words()
        self._build_regex_patterns()
        logger.info("[CommandExtractor] initialized")

    def _register_jieba_words(self) -> None:
        for word in self._all_position_forms() + ACTION_WORDS + self.dictionary_terms + sorted(STOPWORDS_CN, key=len, reverse=True):
            jieba.add_word(word, freq=200000)

    def _all_position_forms(self) -> List[str]:
        forms: List[str] = []
        for standard, aliases in POSITION_ALIASES.items():
            forms.append(standard)
            forms.extend(aliases)
        return sorted(set(forms), key=len, reverse=True)

    def _build_regex_patterns(self) -> None:
        self.position_regex = re.compile("|".join(re.escape(form) for form in self._all_position_forms()))
        self.action_regex = re.compile("|".join(re.escape(word) for word in ACTION_WORDS))

    def extract_command_from_text(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return self._empty_result(text or "", "输入文本为空或无效")

        try:
            cleaned_text = self._preprocess_text(text)
            position, position_raw, without_position = self._extract_position(cleaned_text)
            action = self._extract_action(cleaned_text)
            keyword_source = self._strip_command_noise(without_position)
            raw_keywords = self._extract_keywords(without_position)
            categorized = self._categorize_keywords(raw_keywords)
            theme = self._determine_theme(categorized)
            enhanced_prompt = self._generate_enhanced_prompt(categorized, theme, raw_keywords)
            confidence = self._calculate_confidence(raw_keywords, bool(position_raw), categorized)
            debug_info = {
                "original_text": text,
                "cleaned_text": cleaned_text,
                "position_removed_text": without_position.strip(),
                "command_noise_removed_text": keyword_source,
                "removed_position": position_raw,
                "removed_action": action if action in cleaned_text else "",
                "position_forms_checked": self._all_position_forms(),
                "action_words_checked": ACTION_WORDS,
                "segmentation_engine": "jieba" if self.use_jieba else "dictionary_longest_match",
            }
            logger.info(f"[CommandExtractor] cleaned_text: {cleaned_text}")
            logger.info(f"[CommandExtractor] position_removed_text: {without_position.strip()}")
            logger.info(f"[CommandExtractor] command_noise_removed_text: {keyword_source}")
            logger.debug(f"[CommandExtractor] enhanced_prompt: {enhanced_prompt}")

            return {
                "raw_text": text,
                "success": bool(raw_keywords),
                "keywords": raw_keywords,
                "raw_keywords": raw_keywords,
                "categorized": categorized,
                "theme": theme,
                "enhanced_prompt": enhanced_prompt,
                "position": position,
                "position_raw": position_raw,
                "action": action,
                "confidence": confidence,
                "message": self._generate_message(raw_keywords, position, bool(position_raw), confidence),
                "debug_info": debug_info,
            }
        except Exception as exc:
            logger.info(f"[CommandExtractor] failed to parse text: {exc}")
            traceback.print_exc()
            return self._empty_result(text, f"解析过程发生错误: {exc}")

    def _extract_position(self, text: str) -> tuple[str, str, str]:
        working = text
        for form in self._all_position_forms():
            if form in working:
                standard = self._get_standard_position(form)
                return standard, form, working.replace(form, " ")
        return "中心", "", working

    def _extract_keywords(self, text: str) -> List[str]:
        working = self._strip_command_noise(text)
        if not working:
            return []

        words = [*self._extract_descriptive_phrases(working), *self._segment_words(working)]
        keywords: List[str] = []
        for word in words:
            cleaned = self._clean_keyword(word)
            if not cleaned:
                continue
            if cleaned in STOPWORDS_CN:
                continue
            if len(cleaned) == 1 and cleaned not in SINGLE_CHAR_KEEP:
                continue
            if cleaned not in keywords:
                keywords.append(cleaned)
        return self._remove_embedded_keywords(keywords)

    @staticmethod
    def _extract_descriptive_phrases(text: str) -> List[str]:
        """Preserve free-form verb phrases such as '戴眼镜' and '拿着雨伞'."""
        pattern = re.compile(
            r"(?:戴|穿|拿|抱|背|举|撑|骑|趴|站|坐|躺|看|望|吃|喝|追|拉|推)(?:着|在)?"
            r"[^\s,，。；;]+?(?=的|[\s,，。；;]|$)"
        )
        return list(dict.fromkeys(match.group(0).strip() for match in pattern.finditer(text)))

    def _strip_command_noise(self, text: str) -> str:
        working = text
        for phrase in sorted(STOPWORDS_CN | set(ACTION_WORDS), key=len, reverse=True):
            if len(phrase) == 1:
                continue
            working = working.replace(phrase, " ")
        return PUNCTUATION_RE.sub(" ", working).strip()

    def _segment_words(self, text: str) -> List[str]:
        if self.use_jieba:
            words = [word.strip() for word in jieba.lcut(text) if word.strip()]
            expanded = self._add_dictionary_matches(text, words)
            return expanded
        return self._dictionary_longest_match(text)

    def _add_dictionary_matches(self, text: str, words: List[str]) -> List[str]:
        result = list(words)
        for term in self.dictionary_terms:
            if term in text and term not in result:
                result.append(term)
        return sorted(result, key=lambda item: text.find(item) if item in text else len(text))

    def _remove_embedded_keywords(self, keywords: List[str]) -> List[str]:
        filtered: List[str] = []
        for index, keyword in enumerate(keywords):
            should_drop = False
            for other_index, other in enumerate(keywords):
                if index == other_index or len(other) <= len(keyword) or keyword not in other:
                    continue
                descriptive_fragment = (
                    other.startswith(DESCRIPTIVE_PREFIXES)
                    and not self.classifier.classify_word(keyword, self.dictionary)
                )
                if len(keyword) == 1 and not self.classifier.classify_word(keyword, self.dictionary):
                    should_drop = True
                    break
                if other.endswith("风格") or descriptive_fragment:
                    should_drop = True
                    break
            if should_drop:
                continue
            filtered.append(keyword)
        return filtered

    def _dictionary_longest_match(self, text: str) -> List[str]:
        words: List[tuple[int, str]] = []
        consumed = [False] * len(text)
        for term in self.dictionary_terms:
            start = text.find(term)
            while start >= 0:
                end = start + len(term)
                if not any(consumed[start:end]):
                    words.append((start, term))
                    for index in range(start, end):
                        consumed[index] = True
                start = text.find(term, start + 1)

        residuals: List[tuple[int, str]] = []
        start = None
        for index, is_consumed in enumerate([*consumed, True]):
            if not is_consumed and start is None:
                start = index
            elif is_consumed and start is not None:
                raw_fragment = text[start:index]
                for part in PUNCTUATION_RE.split(raw_fragment):
                    cleaned = self._clean_residual_phrase(part)
                    if cleaned:
                        offset = raw_fragment.find(part)
                        residuals.append((start + max(0, offset), cleaned))
                start = None

        combined = [*words, *residuals]
        if not combined:
            return [part for part in PUNCTUATION_RE.split(text) if part]
        return [term for _, term in sorted(combined, key=lambda item: item[0])]

    @staticmethod
    def _clean_residual_phrase(fragment: str) -> str:
        """Keep meaningful free-form attributes that are absent from the dictionary."""
        cleaned = str(fragment or "").strip()
        cleaned = re.sub(r"^(?:在|有|的|地|得|一个|一只|一幅|一张|一位|一些|背景是|背景为)+", "", cleaned)
        cleaned = re.sub(r"(?:的|地|得|风格|场景|画面)+$", "", cleaned)
        cleaned = cleaned.strip()
        if (len(cleaned) < 2 and cleaned not in SINGLE_CHAR_KEEP) or cleaned in STOPWORDS_CN:
            return ""
        return cleaned

    def _categorize_keywords(self, keywords: List[str]) -> Dict[str, List[str]]:
        categorized: Dict[str, List[str]] = {
            "animals": [],
            "objects": [],
            "scenes": [],
            "styles": [],
            "modifiers": [],
            "actions": [],
            "colors": [],
        }
        for word in keywords:
            for category, normalized in self.classifier.classify_and_normalize(word, self.dictionary):
                if normalized not in categorized[category]:
                    categorized[category].append(normalized)
        return categorized

    def _determine_theme(self, categorized: Dict[str, List[str]]) -> str:
        if categorized["animals"]:
            if categorized["scenes"]:
                return "animal_scene"
            if categorized["actions"]:
                return "animal_action"
            return "animal_portrait"

        if categorized["scenes"]:
            scenes = set(categorized["scenes"])
            if scenes & {"自然风景", "山水", "山水画", "森林", "草原", "海洋", "星空"}:
                return "landscape"
            if scenes & {"城市", "街景", "夜景", "城市风光", "高速公路"}:
                return "cityscape"
            return "scene"

        if categorized["objects"]:
            return "object"
        if categorized["styles"]:
            return "art"
        return "general"

    def _generate_enhanced_prompt(self, categorized: Dict[str, List[str]], theme: str, raw_keywords: List[str]) -> str:
        prompt_parts: List[str] = []

        subjects = self.expander.expand_subjects(categorized)
        if subjects:
            prompt_parts.extend(subjects)
        else:
            prompt_parts.extend(raw_keywords[:4])

        uncategorized = [
            keyword
            for keyword in raw_keywords
            if not self.classifier.classify_word(keyword, self.dictionary)
        ]
        prompt_parts.extend(uncategorized)

        for action in categorized["actions"][:1]:
            prompt_parts.append(f"正在{action}")

        for scene in categorized["scenes"][:2]:
            prompt_parts.append(f"在{scene}中")

        if categorized["styles"]:
            style = categorized["styles"][0]
            prompt_parts.append(style if style.endswith("风格") else f"{style}风格")
        else:
            prompt_parts.append(self._default_style(theme))

        prompt_parts.append(self._quality_term(categorized, theme))
        return "，".join(self._dedupe(prompt_parts))

    def _default_style(self, theme: str) -> str:
        if theme in {"animal_scene", "landscape", "cityscape", "scene"}:
            return "写实风格"
        if theme in {"animal_portrait", "animal_action", "object"}:
            return "高清照片风格"
        if theme == "art":
            return "艺术风格"
        return "高质量"

    def _quality_term(self, categorized: Dict[str, List[str]], theme: str) -> str:
        key = "|".join(categorized.get(name, [""])[0] if categorized.get(name) else "" for name in ["animals", "objects", "scenes", "styles", "actions"]) + theme
        return QUALITY_TERMS[sum(ord(char) for char in key) % len(QUALITY_TERMS)]

    def _dedupe(self, items: List[str]) -> List[str]:
        result: List[str] = []
        for item in items:
            if item and item not in result:
                result.append(item)
        return result

    def _empty_result(self, text: str, message: str) -> Dict[str, Any]:
        categorized = {"animals": [], "objects": [], "scenes": [], "styles": [], "modifiers": [], "actions": [], "colors": []}
        return {
            "raw_text": text,
            "success": False,
            "keywords": [],
            "raw_keywords": [],
            "categorized": categorized,
            "theme": "general",
            "enhanced_prompt": "",
            "position": "",
            "position_raw": "",
            "action": "",
            "confidence": 0.0,
            "message": message,
            "debug_info": {"original_text": text, "error": message},
        }

    def _preprocess_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _get_standard_position(self, matched_position: str) -> str:
        for standard, aliases in POSITION_ALIASES.items():
            if matched_position == standard or matched_position in aliases:
                return standard
        return matched_position

    def _extract_action(self, text: str) -> str:
        match = self.action_regex.search(text)
        return match.group(0) if match else "生成"

    def _clean_keyword(self, word: str) -> str:
        cleaned = PUNCTUATION_RE.sub("", word.strip())
        for form in self._all_position_forms():
            cleaned = cleaned.replace(form, "")
        return cleaned.strip()

    def _calculate_confidence(self, keywords: List[str], position_found: bool, categorized: Dict[str, List[str]]) -> float:
        confidence = 0.35 if keywords else 0.0
        if position_found:
            confidence += 0.25
        if len(keywords) >= 2:
            confidence += 0.15
        elif len(keywords) == 1:
            confidence += 0.1
        if any(categorized.values()):
            confidence += 0.2
        return min(confidence, 1.0)

    def _generate_message(self, keywords: List[str], position: str, position_found: bool, confidence: float) -> str:
        if not keywords:
            return "未提取到有效内容关键词"
        position_label = position if position_found else "默认位置(中心)"
        return f"将在{position_label}生成关于[{', '.join(keywords[:4])}]的图片(置信度 {confidence:.0%})"


_global_command_extractor: Optional[CommandExtractor] = None


def get_command_extractor() -> CommandExtractor:
    global _global_command_extractor
    if _global_command_extractor is None:
        _global_command_extractor = CommandExtractor()
    return _global_command_extractor


def extract_command_from_text(text: str) -> Dict[str, Any]:
    return get_command_extractor().extract_command_from_text(text)


def validate_speech_command(command_result: Dict[str, Any]) -> bool:
    if not command_result or not isinstance(command_result, dict):
        return False
    if not command_result.get("success", False):
        return False
    if not command_result.get("keywords"):
        return False
    return float(command_result.get("confidence", 0.0)) >= 0.3


def test_left_top_cat_extraction() -> bool:
    result = extract_command_from_text("在左上角生成一只猫")
    assert result["position"] == "左上角", result
    assert result["keywords"] == ["猫"], result
    assert result["categorized"]["animals"] == ["猫"], result
    assert validate_speech_command(result), result
    return True


def test_command_extraction() -> bool:
    test_left_top_cat_extraction()
    samples = [
        "在屏幕中间显示一张风景图",
        "创建抽象艺术图案在右下角",
        "在右上角画一个太阳",
        "在左上角生成一只可爱的小狗在草地上玩耍",
    ]
    for sample in samples:
        result = extract_command_from_text(sample)
        logger.info(result)
        assert result["success"], result
    return True


def check_dependencies() -> bool:
    if jieba is None:
        logger.info("[Dependency] jieba not installed; using dictionary longest-match fallback")
        return False
    logger.info("[Dependency] jieba available")
    return True


if __name__ == "__main__":
    check_dependencies()
    ok = test_command_extraction()
    logger.info("command extractor tests passed" if ok else "command extractor tests failed")
