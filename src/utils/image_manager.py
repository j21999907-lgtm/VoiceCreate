"""Generated image save and history management."""

from __future__ import annotations

import glob
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ImageManager:
    """Manage generated image files under the configured save directory."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.save_path = self._resolve_path(self.config.get("save_path") or self.config.get("image_dir") or "./generated_images")
        self.max_saved = int(self.config.get("max_saved_images", self.config.get("max_images", 100)))
        self.keep_days = int(self.config.get("keep_days", self.config.get("auto_cleanup_days", 30)))
        self.auto_clean_old = bool(self.config.get("auto_clean_old", True))
        self.save_path.mkdir(parents=True, exist_ok=True)

    def get_save_path(self) -> str:
        return str(self.save_path.resolve())

    def get_image_count(self) -> int:
        return len(self._image_paths())

    def list_images(self, limit: int = 20) -> List[Dict[str, Any]]:
        images = []
        for path in sorted(self._image_paths(), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
            stat = path.stat()
            images.append(
                {
                    "path": str(path.resolve()),
                    "filename": path.name,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime),
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                }
            )
        return images

    def cleanup_old_images(self) -> int:
        if not self.auto_clean_old or self.keep_days <= 0:
            return 0

        cutoff = datetime.now() - timedelta(days=self.keep_days)
        deleted = 0
        for path in self._image_paths():
            try:
                if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                    self._delete_with_metadata(path)
                    deleted += 1
            except OSError:
                continue
        return deleted

    def enforce_max_limit(self) -> int:
        if self.max_saved <= 0:
            return 0

        images = sorted(self._image_paths(), key=lambda item: item.stat().st_mtime)
        delete_count = max(0, len(images) - self.max_saved)
        for path in images[:delete_count]:
            self._delete_with_metadata(path)
        return delete_count

    def get_latest_image(self) -> Optional[str]:
        images = self._image_paths()
        if not images:
            return None
        return str(max(images, key=lambda item: item.stat().st_mtime).resolve())

    def open_latest_image(self) -> Optional[Image.Image]:
        latest = self.get_latest_image()
        if not latest:
            return None
        return Image.open(latest)

    def _image_paths(self) -> List[Path]:
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
        paths: List[Path] = []
        for pattern in patterns:
            paths.extend(Path(item) for item in glob.glob(str(self.save_path / pattern)))
        return paths

    def _delete_with_metadata(self, path: Path) -> None:
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path
