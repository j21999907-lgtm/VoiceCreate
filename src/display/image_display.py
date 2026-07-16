"""Tkinter image display helpers for VoiceCreate."""

from __future__ import annotations

import logging

logger = logging.getLogger("VoiceCreate")


import sys
import traceback
import weakref
from pathlib import Path
from typing import Any, Tuple


def _extract_image(image_or_result: Any) -> Any:
    if hasattr(image_or_result, "image"):
        return image_or_result.image
    return image_or_result


def _load_pil_image(image: Any) -> Any:
    from PIL import Image

    image = _extract_image(image)
    if image is None:
        raise ValueError("image is None")

    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"image path does not exist: {path}")
        pil_image = Image.open(path)
    elif hasattr(image, "save") and hasattr(image, "size"):
        pil_image = image
    else:
        raise TypeError(f"unsupported image type: {type(image).__name__}")

    if not hasattr(pil_image, "size") or len(pil_image.size) != 2:
        raise ValueError("image does not expose a valid size")
    width, height = pil_image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {pil_image.size}")

    if getattr(pil_image, "mode", None) not in ("RGB", "RGBA"):
        pil_image = pil_image.convert("RGBA")
    return pil_image.copy()


class _ImageRef:
    """Hashable weakref target that owns a display image while its window is open."""

    __slots__ = ("image", "__weakref__")

    def __init__(self, image: Any) -> None:
        self.image = image


class ImageDisplayManager:
    """Create user-controlled image windows while retaining Tk references."""

    def __init__(self) -> None:
        self.image_refs: weakref.WeakSet[Any] = weakref.WeakSet()
        self.window_refs: weakref.WeakSet[Any] = weakref.WeakSet()
        self.last_display_coordinates: Tuple[int, int] | None = None
        self.last_display_size: Tuple[int, int] | None = None

    def display_image_complete(self, image: Any, x: int, y: int, title: str = "VoiceCreate") -> bool:
        return self.display_image_at(image, x, y, title=title)

    def display_image_at(self, image: Any, x: int, y: int, duration_ms: int = 5000, title: str = "VoiceCreate") -> bool:
        try:
            import tkinter as tk
            from PIL import ImageTk

            pil_image = _load_pil_image(image)
            original_width, original_height = pil_image.size
            logger.info("[DISPLAY] 开始显示图片")
            logger.info(f"        图片尺寸: {original_width}x{original_height}")
            logger.info(f"        图片模式: {getattr(pil_image, 'mode', 'Unknown')}")
            logger.info(f"        图片格式: {getattr(pil_image, 'format', 'Unknown')}")
            logger.info(f"        请求位置: ({x}, {y})")
            parent_root = tk._default_root
            owns_mainloop = parent_root is None
            if parent_root is None:
                root = tk.Tk()
            else:
                root = tk.Toplevel(parent_root)

            root.withdraw()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.title(title)

            left, top, right, bottom = self._virtual_screen_bounds(root)
            screen_width = max(1, right - left)
            screen_height = max(1, bottom - top)
            display_image = self._fit_image_to_screen(pil_image, screen_width, screen_height)
            width, height = display_image.size
            fitted_x, fitted_y = self._fit_position(root, int(x), int(y), width, height)
            self.last_display_coordinates = (int(fitted_x), int(fitted_y))
            self.last_display_size = (int(width), int(height))
            geometry = f"{width}x{height}+{fitted_x}+{fitted_y}"
            logger.debug(f"[DEBUG] 屏幕范围: ({left}, {top})-({right}, {bottom})")
            logger.debug(f"[DEBUG] 实际显示尺寸: {width}x{height}")
            logger.debug(f"[DEBUG] 窗口几何设置: {geometry}")
            root.geometry(geometry)
            root.minsize(width, height)
            root.maxsize(width, height)

            photo = ImageTk.PhotoImage(display_image, master=root)
            pil_ref = _ImageRef(display_image)
            label = tk.Label(root, image=photo, borderwidth=0, highlightthickness=0, bg="#000000")
            label.place(x=0, y=0, width=width, height=height)

            close_button = tk.Button(
                root,
                text="X",
                command=lambda: cleanup(),
                bg="#7F1D1D",
                fg="#FFFFFF",
                activebackground="#991B1B",
                activeforeground="#FFFFFF",
                relief="flat",
                borderwidth=0,
                font=("Arial", 10, "bold"),
            )
            close_button.place(x=10, y=10, width=60, height=30)

            # Tkinter drops images if Python releases the PhotoImage object.
            self.image_refs.add(photo)
            self.image_refs.add(pil_ref)
            self.window_refs.add(root)
            root._photo_ref = photo
            root._pil_ref = pil_ref
            root._label_ref = label
            root._normal_geometry = root.geometry()
            root._is_maximized = False
            label.image = photo

            def cleanup() -> None:
                nonlocal display_image, label, photo, pil_ref, root

                current_label = label
                current_root = root
                current_photo = photo
                current_pil_ref = pil_ref

                if current_root is None:
                    return

                try:
                    self.image_refs.discard(current_photo)
                    self.image_refs.discard(current_pil_ref)
                    self.window_refs.discard(current_root)

                    if current_label is not None:
                        try:
                            current_label.config(image="")
                            current_label.image = None
                        except Exception:
                            pass

                    for attr in ("_photo_ref", "_pil_ref", "_label_ref"):
                        if hasattr(current_root, attr):
                            try:
                                setattr(current_root, attr, None)
                            except Exception:
                                pass

                    if current_pil_ref is not None:
                        current_pil_ref.image = None

                    try:
                        current_root.quit()
                    except Exception:
                        pass
                    current_root.destroy()
                except Exception as exc:
                    logger.warning("Failed to close display window: %s", exc)
                finally:
                    photo = None
                    pil_ref = None
                    display_image = None
                    label = None
                    root = None

            def minimize_window() -> None:
                try:
                    root.overrideredirect(False)
                    root.iconify()
                except Exception as exc:
                    logger.warning("Failed to minimize display window: %s", exc)

            def restore_overrideredirect(_event: Any = None) -> None:
                try:
                    if root.state() != "iconic":
                        root.overrideredirect(True)
                except Exception:
                    pass

            def toggle_maximize() -> None:
                try:
                    if getattr(root, "_is_maximized", False):
                        root.geometry(root._normal_geometry)
                        root._is_maximized = False
                    else:
                        root._normal_geometry = root.geometry()
                        left, top, right, bottom = self._virtual_screen_bounds(root)
                        root.geometry(f"{right - left}x{bottom - top}+{left}+{top}")
                        root._is_maximized = True
                except Exception as exc:
                    logger.warning("Failed to maximize display window: %s", exc)

            def close_on_focus_out() -> None:
                try:
                    if root.focus_get() is None:
                        cleanup()
                except Exception:
                    cleanup()

            drag_state = {"x": 0, "y": 0}

            def start_drag(event: Any) -> None:
                drag_state["x"] = event.x_root - root.winfo_x()
                drag_state["y"] = event.y_root - root.winfo_y()

            def drag_window(event: Any) -> None:
                if getattr(root, "_is_maximized", False):
                    return
                root.geometry(f"+{event.x_root - drag_state['x']}+{event.y_root - drag_state['y']}")

            label.bind("<ButtonPress-1>", start_drag)
            label.bind("<B1-Motion>", drag_window)
            root.bind("<Escape>", lambda _event: cleanup())
            root.bind("<FocusOut>", lambda _event: root.after(120, close_on_focus_out))
            root.bind("<Map>", restore_overrideredirect)
            root.protocol("WM_DELETE_WINDOW", cleanup)

            root.deiconify()
            root.lift()
            try:
                root.focus_force()
            except Exception:
                pass
            if owns_mainloop:
                root.mainloop()
            logger.info("[DISPLAY] 图片显示完成")
            return True
        except Exception as exc:
            logger.error("Failed to display image at (%s, %s): %s", x, y, exc)
            logger.info(f"[DISPLAY ERROR] Failed to display image at ({x}, {y}): {exc}")
            traceback.print_exc(file=sys.stderr)
            return False

    def _fit_image_to_screen(self, image: Any, screen_width: int, screen_height: int) -> Any:
        from PIL import Image

        width, height = image.size
        min_side = 100
        max_width = max(min_side, screen_width)
        max_height = max(min_side, screen_height)
        scale = min(max_width / width, max_height / height, 1.0)

        if width < min_side and height < min_side:
            scale = min(max(min_side / width, min_side / height), max_width / width, max_height / height)

        if abs(scale - 1.0) < 0.001:
            return image

        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        logger.debug(f"[DEBUG] 图片按比例缩放: {width}x{height} -> {new_size[0]}x{new_size[1]}")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        return image.resize(new_size, resampling)

    def _fit_position(self, root: Any, x: int, y: int, width: int, height: int) -> Tuple[int, int]:
        left, top, right, bottom = self._virtual_screen_bounds(root)
        max_x = max(left, right - width)
        max_y = max(top, bottom - height)
        return min(max(x, left), max_x), min(max(y, top), max_y)

    def _virtual_screen_bounds(self, root: Any) -> Tuple[int, int, int, int]:
        try:
            from screeninfo import get_monitors

            monitors = get_monitors()
            if monitors:
                left = min(m.x for m in monitors)
                top = min(m.y for m in monitors)
                right = max(m.x + m.width for m in monitors)
                bottom = max(m.y + m.height for m in monitors)
                return left, top, right, bottom
        except Exception as exc:
            logger.debug("screeninfo unavailable; using Tk primary bounds: %s", exc)

        return 0, 0, int(root.winfo_screenwidth()), int(root.winfo_screenheight())


_display_manager = ImageDisplayManager()


def get_display_manager() -> ImageDisplayManager:
    return _display_manager


def display_image_at(
    image: Any,
    x: int,
    y: int,
    display_time: int = 5000,
    duration_ms: int | None = None,
    title: str = "VoiceCreate",
) -> bool:
    duration = display_time if duration_ms is None else duration_ms
    return _display_manager.display_image_at(image, x, y, duration_ms=duration, title=title)


def display_generated_image(image_or_result: Any, coords: Tuple[int, int], display_time: int = 5000) -> bool:
    x, y = coords
    return display_image_at(image_or_result, x, y, display_time=display_time)


if __name__ == "__main__":
    from PIL import Image, ImageDraw

    sample = Image.new("RGB", (320, 240), (36, 40, 52))
    draw = ImageDraw.Draw(sample)
    draw.rectangle((12, 12, 308, 228), outline=(120, 180, 240), width=3)
    draw.text((24, 106), "VoiceCreate display test", fill=(240, 244, 255))
    display_image_at(sample, 50, 50, display_time=2000)
