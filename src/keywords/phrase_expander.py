"""Phrase expansion helpers for image prompts."""

from __future__ import annotations

from typing import Dict, List


class PhraseExpander:
    """Build fluent Chinese prompt fragments from categorized keywords."""

    def expand_subjects(self, categorized: Dict[str, List[str]]) -> List[str]:
        modifiers = categorized.get("modifiers", [])
        colors = categorized.get("colors", [])
        descriptor = ""
        if modifiers:
            descriptor = modifiers[0]
        if colors:
            descriptor = f"{colors[0]}的{descriptor}" if descriptor else colors[0]

        subjects = categorized.get("animals", []) or categorized.get("objects", [])
        if not subjects:
            return []
        if descriptor:
            return [f"{descriptor}的{subject}" for subject in subjects]
        return list(subjects)
