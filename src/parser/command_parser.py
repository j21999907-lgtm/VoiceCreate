"""Lightweight command parser used by the integrated VoiceCreate workflow."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, Optional


DEFAULT_NEGATIVE_PROMPT = "模糊, 像素化, 低质量, 水印, 文字, 丑陋, 畸形, 失真"


def parse_command_text(text: str, extracted_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    extracted_info = extracted_info or {}
    keywords = list(extracted_info.get("keywords") or [])
    categorized = extracted_info.get("categorized") or {}
    enhanced_prompt = extracted_info.get("enhanced_prompt") or ""
    theme = extracted_info.get("theme") or "general"
    position = extracted_info.get("position") or "中心"
    position_raw = extracted_info.get("position_raw") or (position if position != "中心" else "")
    action = extracted_info.get("action") or "生成"

    scene_description = enhanced_prompt or "，".join(str(item) for item in keywords) or str(text or "").strip()
    if scene_description and not enhanced_prompt and not scene_description.endswith("的场景"):
        scene_description = f"{scene_description}的场景"

    confidence = 0.4
    if keywords:
        confidence += 0.25
    if position_raw:
        confidence += 0.2
    if action:
        confidence += 0.1
    confidence = min(confidence, 1.0)
    prompt_source = "extractor.enhanced_prompt" if enhanced_prompt else "keywords_or_raw_text"
    debug_info = {
        "extractor_debug": extracted_info.get("debug_info") or {},
        "prompt_source": prompt_source,
        "candidate_prompt": scene_description,
        "removed_position": position_raw,
        "removed_action": action,
        "raw_keywords": list(extracted_info.get("raw_keywords") or keywords),
        "categorized": categorized,
    }

    return {
        "raw_text": text,
        "parsed_success": bool(keywords),
        "command_id": _command_id(text),
        "timestamp": datetime.now().isoformat(),
        "task_type": "generate_image",
        "action": action,
        "action_en": "generate",
        "keywords": keywords,
        "raw_keywords": list(extracted_info.get("raw_keywords") or keywords),
        "categorized": categorized,
        "theme": theme,
        "enhanced_prompt": enhanced_prompt,
        "categories": [name for name, values in categorized.items() if values],
        "objects": _build_objects(keywords, categorized),
        "scene_description": scene_description,
        "style_tags": list(categorized.get("styles") or []),
        "position": {
            "raw": position_raw,
            "normalized": position,
            "coordinates": None,
            "explicit": bool(position_raw),
        },
        "parameters": {
            "quality": "high",
            "size": "medium",
            "width": 512,
            "height": 512,
            "steps": 35,
            "guidance_scale": 7.5,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "aspect_ratio": "1:1",
            "detail_level": "high",
        },
        "confidence": confidence,
        "validation": {
            "has_keywords": bool(keywords),
            "position_explicit": bool(position_raw),
        },
        "suggestions": [],
        "debug_info": debug_info,
    }


def _command_id(text: str) -> str:
    digest = hashlib.md5((text or "").encode("utf-8")).hexdigest()[:8]
    return f"cmd_{datetime.now().strftime('%Y%m%d%H%M%S')}_{digest}"


def _build_objects(keywords: list[str], categorized: Dict[str, Any]) -> list[Dict[str, Any]]:
    category_by_word = {}
    for category, values in categorized.items():
        for value in values or []:
            category_by_word.setdefault(value, category)

    objects = []
    for item in keywords:
        objects.append({"name": item, "category": category_by_word.get(item, "unknown"), "attributes": []})
    return objects
