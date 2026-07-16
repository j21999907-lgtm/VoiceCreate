"""Resolve VoiceCreate position descriptions into screen coordinates."""

from __future__ import annotations

import logging

logger = logging.getLogger("VoiceCreate")


import difflib
import random
from dataclasses import make_dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.shared.position_aliases import POSITION_ALIASES


try:
    from screeninfo import get_monitors
except ImportError:
    get_monitors = None


logger = logging.getLogger(__name__)


def get_screen_info(monitor_index: int = 0) -> Tuple[int, int, int, int]:
    """Return monitor width, height, x offset, and y offset."""
    try:
        monitors = get_monitors() if get_monitors else []
        if not monitors:
            logger.warning("Unable to read monitor info; using fallback 1920x1080.")
            return 1920, 1080, 0, 0

        primary_index = next((index for index, monitor in enumerate(monitors) if getattr(monitor, "is_primary", False)), 0)
        index = monitor_index if 0 <= monitor_index < len(monitors) else primary_index
        monitor = monitors[index]
        width = int(getattr(monitor, "width", 1920))
        height = int(getattr(monitor, "height", 1080))
        x = int(getattr(monitor, "x", 0))
        y = int(getattr(monitor, "y", 0))
        logger.info("Monitor %s: %sx%s @ (%s, %s)", index, width, height, x, y)
        return width, height, x, y
    except Exception as exc:
        logger.error("Failed to read monitor info: %s", exc)
        return 1920, 1080, 0, 0


def _position_kind(position_description: str) -> str:
    text = "".join(str(position_description or "").strip().split())
    if any(token in text for token in ("右上角", "右上", "右上方", "鍙充笂瑙?", "鍙充笂", "鍙充笂鏂?")):
        return "top_right"
    if any(token in text for token in ("左上角", "左上", "左上方", "宸︿笂瑙?", "宸︿笂", "宸︿笂鏂?")):
        return "top_left"
    if any(token in text for token in ("右下角", "右下", "右下方", "鍙充笅瑙?", "鍙充笅", "鍙充笅鏂?")):
        return "bottom_right"
    if any(token in text for token in ("左下角", "左下", "左下方", "宸︿笅瑙?", "宸︿笅", "宸︿笅鏂?")):
        return "bottom_left"
    if any(token in text for token in ("顶部", "上方", "上面", "顶端", "椤堕儴", "涓婃柟", "涓婇潰")):
        return "top"
    if any(token in text for token in ("底部", "下方", "下面", "底端", "搴曢儴", "涓嬫柟", "涓嬮潰")):
        return "bottom"
    if any(token in text for token in ("左边", "左侧", "左方", "宸﹁竟", "宸︿晶", "宸︽柟")):
        return "left"
    if any(token in text for token in ("右边", "右侧", "右方", "鍙宠竟", "鍙充晶", "鍙虫柟")):
        return "right"
    return "center"


def calculate_display_coordinates(
    position_description: str,
    image_width: int,
    image_height: int,
    margin: int = 20,
    monitor_index: int = 0,
) -> Tuple[int, int]:
    """Calculate a top-left display coordinate that keeps the image on screen."""
    screen_width, screen_height, screen_x, screen_y = get_screen_info(monitor_index)
    width = max(1, int(image_width))
    height = max(1, int(image_height))
    margin = max(0, int(margin))
    kind = _position_kind(position_description)

    logger.info(
        "Calculating display coordinates: position=%s kind=%s image=%sx%s screen=%sx%s@(%s,%s)",
        position_description,
        kind,
        width,
        height,
        screen_width,
        screen_height,
        screen_x,
        screen_y,
    )

    if kind == "top_right":
        x = screen_x + screen_width - width - margin
        y = screen_y + margin
    elif kind == "top_left":
        x = screen_x + margin
        y = screen_y + margin
    elif kind == "bottom_right":
        x = screen_x + screen_width - width - margin
        y = screen_y + screen_height - height - margin
    elif kind == "bottom_left":
        x = screen_x + margin
        y = screen_y + screen_height - height - margin
    elif kind == "top":
        x = screen_x + (screen_width - width) // 2
        y = screen_y + margin
    elif kind == "bottom":
        x = screen_x + (screen_width - width) // 2
        y = screen_y + screen_height - height - margin
    elif kind == "left":
        x = screen_x + margin
        y = screen_y + (screen_height - height) // 2
    elif kind == "right":
        x = screen_x + screen_width - width - margin
        y = screen_y + (screen_height - height) // 2
    else:
        x = screen_x + (screen_width - width) // 2
        y = screen_y + (screen_height - height) // 2

    max_x = screen_x + max(0, screen_width - width - margin)
    max_y = screen_y + max(0, screen_height - height - margin)
    x = max(screen_x + margin, min(x, max_x))
    y = max(screen_y + margin, min(y, max_y))
    logger.info("Display coordinates calculated: (%s, %s)", x, y)
    return int(x), int(y)


class PositionResolver:
    """Convert Chinese position words into virtual-screen coordinates."""

    def __init__(self, padding_ratio: float = 0.05) -> None:
        self.padding_ratio = min(max(float(padding_ratio), 0.0), 0.5)
        self.monitors: List[Any] = []
        self.primary_monitor: Any = None
        self.position_mapping_table: Dict[str, Tuple[int, int]] = {}
        self.valid_positions: set[str] = set()
        self._alias_to_standard = self._build_alias_map()
        self._initialize_monitors()
        self.create_position_mapping_table(0)

    def _build_alias_map(self) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        for standard, aliases in POSITION_ALIASES.items():
            alias_map[standard] = standard
            for alias in aliases:
                alias_map[alias] = standard
        return alias_map

    def _initialize_monitors(self) -> None:
        try:
            monitors = get_monitors() if get_monitors else []
            if not monitors:
                raise RuntimeError("No monitor information available")
            self.monitors = list(monitors)
            self.primary_monitor = next((m for m in self.monitors if getattr(m, "is_primary", False)), self.monitors[0])
        except Exception as exc:
            logger.info(f"[PositionResolver] monitor detection failed; using fallback display: {exc}")
            Monitor = make_dataclass("Monitor", ["x", "y", "width", "height", "is_primary", "name"])
            self.primary_monitor = Monitor(0, 0, 1920, 1080, True, "fallback")
            self.monitors = [self.primary_monitor]

    def create_position_mapping_table(self, monitor_index: int = 0) -> Dict[str, Tuple[int, int]]:
        if monitor_index < 0 or monitor_index >= len(self.monitors):
            monitor_index = 0

        monitor = self.monitors[monitor_index]
        screen_x = int(getattr(monitor, "x", 0))
        screen_y = int(getattr(monitor, "y", 0))
        screen_width = int(getattr(monitor, "width", 1920))
        screen_height = int(getattr(monitor, "height", 1080))

        padding_x = int(screen_width * self.padding_ratio)
        padding_y = int(screen_height * self.padding_ratio)
        center_x = screen_width // 2
        center_y = screen_height // 2

        left = screen_x + padding_x
        right = screen_x + screen_width - padding_x
        top = screen_y + padding_y
        bottom = screen_y + screen_height - padding_y
        center = (screen_x + center_x, screen_y + center_y)

        mapping = {
            "左上角": (left, top),
            "左上": (left, top),
            "左上方": (left, top),
            "右上角": (right, top),
            "右上": (right, top),
            "右上方": (right, top),
            "左下角": (left, bottom),
            "左下": (left, bottom),
            "左下方": (left, bottom),
            "右下角": (right, bottom),
            "右下": (right, bottom),
            "右下方": (right, bottom),
            "中心": center,
            "中间": center,
            "中央": center,
            "正中": center,
            "正中央": center,
            "居中": center,
            "顶部": (screen_x + center_x, top),
            "上方": (screen_x + center_x, top),
            "上面": (screen_x + center_x, top),
            "底部": (screen_x + center_x, bottom),
            "下方": (screen_x + center_x, bottom),
            "下面": (screen_x + center_x, bottom),
            "左边": (left, screen_y + center_y),
            "左侧": (left, screen_y + center_y),
            "左方": (left, screen_y + center_y),
            "右边": (right, screen_y + center_y),
            "右侧": (right, screen_y + center_y),
            "右方": (right, screen_y + center_y),
            "随机位置": self._get_random_position(screen_x, screen_y, screen_width, screen_height),
        }

        self.position_mapping_table = mapping
        self.valid_positions = set(mapping)
        return mapping

    def _get_random_position(self, screen_x: int, screen_y: int, screen_width: int, screen_height: int) -> Tuple[int, int]:
        padding_x = max(40, int(screen_width * self.padding_ratio))
        padding_y = max(40, int(screen_height * self.padding_ratio))
        return (
            random.randint(screen_x + padding_x, screen_x + screen_width - padding_x),
            random.randint(screen_y + padding_y, screen_y + screen_height - padding_y),
        )

    def identify_position_description(self, position_text: str) -> Optional[str]:
        if not position_text:
            return None

        normalized = "".join(str(position_text).strip().split())
        if not normalized:
            return None

        if normalized in self.position_mapping_table:
            return normalized
        if normalized in self._alias_to_standard:
            return self._alias_to_standard[normalized]

        for alias in sorted(self._alias_to_standard, key=len, reverse=True):
            if alias in normalized or normalized in alias:
                return self._alias_to_standard[alias]

        candidates = list(self._alias_to_standard) + list(self.valid_positions)
        matches = difflib.get_close_matches(normalized, candidates, n=1, cutoff=0.55)
        if matches:
            return self._alias_to_standard.get(matches[0], matches[0])

        return None

    def convert_position_to_coordinates(
        self,
        position_description: str,
        monitor_index: int = 0,
    ) -> Optional[Tuple[int, int]]:
        if not self.position_mapping_table:
            self.create_position_mapping_table(monitor_index)

        standard = self.identify_position_description(position_description) or "中心"
        coords = self.position_mapping_table.get(standard)
        if coords is not None:
            return coords

        self.create_position_mapping_table(monitor_index)
        return self.position_mapping_table.get(standard, self.position_mapping_table.get("中心"))

    def validate_position(self, position_description: str) -> bool:
        return self.identify_position_description(position_description) is not None

    def get_all_positions(self) -> List[str]:
        return sorted(self.valid_positions)

    def get_screen_info(self, monitor_index: int = 0) -> Dict[str, Any]:
        if monitor_index < 0 or monitor_index >= len(self.monitors):
            return {}
        monitor = self.monitors[monitor_index]
        return {
            "index": monitor_index,
            "x": int(getattr(monitor, "x", 0)),
            "y": int(getattr(monitor, "y", 0)),
            "width": int(getattr(monitor, "width", 1920)),
            "height": int(getattr(monitor, "height", 1080)),
            "is_primary": bool(getattr(monitor, "is_primary", False)),
            "name": getattr(monitor, "name", f"monitor_{monitor_index}"),
        }


_global_position_resolver: Optional[PositionResolver] = None


def get_position_resolver(padding_ratio: float = 0.05) -> PositionResolver:
    global _global_position_resolver
    if _global_position_resolver is None:
        _global_position_resolver = PositionResolver(padding_ratio)
    return _global_position_resolver


def create_position_mapping_table(monitor_index: int = 0) -> Dict[str, Tuple[int, int]]:
    return get_position_resolver().create_position_mapping_table(monitor_index)


def identify_position_description(position_text: str) -> Optional[str]:
    if not position_text:
        return get_position_resolver().identify_position_description(position_text)

    text = "".join(str(position_text).strip().split())
    direct_aliases = {standard: tuple(aliases) for standard, aliases in POSITION_ALIASES.items()}
    for standard, aliases in direct_aliases.items():
        if any(alias in text for alias in aliases):
            return standard
    return get_position_resolver().identify_position_description(position_text)


def convert_position_to_coordinates(position_description: str, monitor_index: int = 0) -> Optional[Tuple[int, int]]:
    return get_position_resolver().convert_position_to_coordinates(position_description, monitor_index)


def test_position_resolution() -> bool:
    resolver = PositionResolver()
    assert resolver.identify_position_description("左上角") == "左上角"
    assert resolver.identify_position_description("屏幕左上角附近") == "左上角"
    assert resolver.identify_position_description("左上脚") == "左上角"
    assert resolver.convert_position_to_coordinates("左上角") is not None
    return True


if __name__ == "__main__":
    logger.info("position resolver tests passed" if test_position_resolution() else "position resolver tests failed")
