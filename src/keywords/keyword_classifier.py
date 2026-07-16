"""Keyword classifier for the Chinese keyword dictionary."""

from __future__ import annotations

from typing import List, Tuple

from .keyword_dictionary import ChineseKeywordDictionary


class KeywordClassifier:
    """Classify words into image prompt categories."""

    def classify_word(self, word: str, dictionary: ChineseKeywordDictionary) -> List[str]:
        return dictionary.word_to_categories.get(word, [])

    def normalize_word(self, word: str, dictionary: ChineseKeywordDictionary, category_name: str) -> str:
        category_dict = dictionary.category_maps().get(category_name, {})
        for main_word, variants in category_dict.items():
            if word == main_word or word in variants:
                if category_name == "objects":
                    return word
                if category_name == "scenes":
                    return word
                if category_name in {"styles", "modifiers", "actions", "colors"}:
                    return word
                return main_word
        return word

    def classify_and_normalize(self, word: str, dictionary: ChineseKeywordDictionary) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        for category_name in self.classify_word(word, dictionary):
            results.append((category_name, self.normalize_word(word, dictionary, category_name)))
        return results
