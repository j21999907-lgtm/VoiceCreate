#!/usr/bin/env python3
"""VoiceCreate graphical main application with voice and keyboard input."""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, Frame, IntVar, Label, StringVar, Text, Tk
from tkinter import ttk
from typing import Any, Dict, Tuple


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class VoiceCreateApp:
    WIDTH = 1040
    HEIGHT = 720

    COLORS = {
        "bg": "#FFFFFF",
        "panel": "#FFFFFF",
        "panel2": "#FFFFFF",
        "text": "#000000",
        "muted": "#525252",
        "accent": "#000000",
        "accent_hover": "#000000",
        "accent2": "#000000",
        "danger": "#000000",
        "warning": "#525252",
        "border": "#000000",
    }

    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("VoiceCreate")
        self.root.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.root.minsize(900, 640)
        self._center_window()

        self.status_var = StringVar(value="Initializing...")
        self.record_button_var = StringVar(value="Start Recording")
        self.progress_var = StringVar(value="Ready")
        self.input_mode_var = StringVar(value="voice")
        self.syntax_var = StringVar(value="Keyboard input ready")
        self.history_var = StringVar(value="")
        self.steps_var = IntVar(value=4)
        self.quality_mode = StringVar(value="high")
        self.steps_label_var = StringVar(value="4 steps")
        self.ai_enhancer_enabled = True
        self.ai_enhancer_label = StringVar(value="AI Enhance: On")

        self.is_recording = False
        self.worker_running = False
        self.modules_ready = False
        self.ui_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()

        self.config: Dict[str, Any] = {}
        self.global_state: Dict[str, Any] = {"modules": {}}
        self.logger = None
        self.image_generator = None
        self.preview_photo = None
        self.input_history = []
        self.generation_details: Dict[str, str] = {}

        self._configure_theme()
        self._build_layout()
        self._start_model_initialization()
        self._poll_queue()

    def _center_window(self) -> None:
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = max(0, (screen_w - self.WIDTH) // 2)
        y = max(0, (screen_h - self.HEIGHT) // 2)
        self.root.geometry(f"{self.WIDTH}x{self.HEIGHT}+{x}+{y}")

    def _configure_theme(self) -> None:
        self.root.configure(bg=self.COLORS["bg"])
        self.root.option_add("*Font", ("Segoe UI", 10))
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Primary.TButton",
            background=self.COLORS["accent"],
            foreground="#FFFFFF",
            borderwidth=0,
            focusthickness=0,
            padding=(16, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.COLORS["accent_hover"]), ("disabled", "#D9D9D9")],
            foreground=[("disabled", "#737373")],
        )
        style.configure(
            "Secondary.TButton",
            background=self.COLORS["panel2"],
            foreground=self.COLORS["text"],
            bordercolor=self.COLORS["border"],
            borderwidth=1,
            padding=(13, 9),
            font=("Segoe UI", 9),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#E5E5E5"), ("disabled", self.COLORS["panel"])],
            foreground=[("disabled", "#8C8C8C")],
        )
        style.configure(
            "Voice.Horizontal.TProgressbar",
            background=self.COLORS["accent"],
            troughcolor=self.COLORS["panel2"],
            bordercolor=self.COLORS["panel2"],
        )
        style.configure("Voice.Horizontal.TScale", background=self.COLORS["panel"], troughcolor=self.COLORS["panel2"])
        style.configure("Voice.TRadiobutton", background=self.COLORS["panel"], foreground=self.COLORS["muted"], indicatorcolor=self.COLORS["panel2"], padding=(0, 4))
        style.map("Voice.TRadiobutton", foreground=[("selected", self.COLORS["text"])], indicatorcolor=[("selected", self.COLORS["accent"])])
        style.configure("Voice.TCombobox", fieldbackground=self.COLORS["panel2"], background=self.COLORS["panel2"], foreground=self.COLORS["text"], arrowcolor=self.COLORS["muted"], bordercolor=self.COLORS["border"], padding=5)

    def _build_layout(self) -> None:
        header = Frame(self.root, bg=self.COLORS["bg"])
        header.pack(fill=X, padx=28, pady=(22, 14))
        Label(header, text="VoiceCreate", bg=self.COLORS["bg"], fg=self.COLORS["text"], font=("Segoe UI Semibold", 19)).pack(side=LEFT)
        Label(
            header,
            text="LOCAL IMAGE STUDIO",
            bg=self.COLORS["bg"],
            fg=self.COLORS["muted"],
            font=("Segoe UI", 8, "bold"),
        ).pack(side=LEFT, padx=(12, 0), pady=(5, 0))
        Label(header, textvariable=self.status_var, bg=self.COLORS["bg"], fg=self.COLORS["accent2"], font=("Segoe UI", 9)).pack(side=RIGHT, pady=(4, 0))

        controls = Frame(self.root, bg=self.COLORS["panel"])
        controls.pack(fill=X, padx=28, pady=(0, 14))
        controls.grid_columnconfigure(1, weight=1)

        self.record_button = ttk.Button(
            controls,
            textvariable=self.record_button_var,
            command=self._on_record_clicked,
            style="Primary.TButton",
        )
        self.record_button.grid(row=0, column=0, padx=(14, 8), pady=14, sticky="nw")

        self.mode_button = ttk.Button(
            controls,
            text="Mode: Voice",
            command=self._toggle_input_mode,
            style="Secondary.TButton",
        )
        self.mode_button.grid(row=0, column=2, padx=8, pady=14, sticky="nw")

        keyboard_panel = Frame(controls, bg=self.COLORS["panel"])
        keyboard_panel.grid(row=0, column=1, sticky="ew", pady=14)
        keyboard_panel.grid_columnconfigure(0, weight=1)

        self.keyboard_input = Text(
            keyboard_panel,
            height=2,
            bg=self.COLORS["panel2"],
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            relief="flat",
            wrap="word",
            font=("Segoe UI", 11),
            padx=10,
            pady=8,
            selectbackground="#D9D9D9",
            highlightbackground=self.COLORS["border"],
            highlightcolor=self.COLORS["border"],
            highlightthickness=1,
        )
        self.keyboard_input.grid(row=0, column=0, sticky="ew")
        self.keyboard_input.insert("1.0", "在左上角生成一只戴眼镜的猫，水彩风格")
        self.keyboard_input.bind("<KeyRelease>", self._on_keyboard_changed)
        self.keyboard_input.bind("<Return>", self._on_keyboard_return)
        self.keyboard_input.bind("<Control-Return>", self._insert_keyboard_newline)

        helper = Frame(keyboard_panel, bg=self.COLORS["panel"])
        helper.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        helper.grid_columnconfigure(0, weight=1)
        Label(helper, textvariable=self.syntax_var, bg=self.COLORS["panel"], fg=self.COLORS["muted"], font=("Segoe UI", 8)).grid(row=0, column=0, sticky="w")
        self.ai_enhancer_button = ttk.Button(
            helper,
            textvariable=self.ai_enhancer_label,
            command=self._toggle_ai_enhancer,
            style="Secondary.TButton",
        )
        self.ai_enhancer_button.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.history_combo = ttk.Combobox(helper, textvariable=self.history_var, state="readonly", width=24, style="Voice.TCombobox")
        self.history_combo.grid(row=0, column=2, sticky="e", padx=(8, 0))
        self.history_combo.bind("<<ComboboxSelected>>", self._on_history_selected)

        self.submit_button = ttk.Button(controls, text="Generate", command=self._on_keyboard_submit, style="Primary.TButton")
        self.submit_button.grid(row=0, column=3, padx=(0, 14), pady=14, sticky="nw")

        content = Frame(self.root, bg=self.COLORS["bg"])
        content.pack(fill=BOTH, expand=True, padx=28, pady=(0, 16))
        content.grid_columnconfigure(0, weight=4)
        content.grid_columnconfigure(1, weight=5)
        content.grid_rowconfigure(0, weight=1)

        left_panel = self._panel(content)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        right_panel = self._panel(content)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        Label(left_panel, text="INPUT", **self._label_style(9, True, "muted")).pack(anchor="w", padx=18, pady=(16, 7))
        self.text_output = Text(left_panel, height=5, bg=self.COLORS["panel2"], fg=self.COLORS["text"], insertbackground=self.COLORS["text"], relief="flat", wrap="word", font=("Segoe UI", 10), highlightbackground=self.COLORS["border"], highlightthickness=1)
        self.text_output.pack(fill=X, padx=18)

        Label(left_panel, text="GENERATION DETAILS", **self._label_style(9, True, "muted")).pack(anchor="w", padx=18, pady=(18, 7))
        self.parse_output = Text(left_panel, bg=self.COLORS["panel2"], fg=self.COLORS["text"], insertbackground=self.COLORS["text"], relief="flat", wrap="word", font=("Consolas", 9), highlightbackground=self.COLORS["border"], highlightthickness=1)
        self.parse_output.pack(fill=BOTH, expand=True, padx=18, pady=(0, 18))

        Label(right_panel, text="OUTPUT", **self._label_style(9, True, "muted")).pack(anchor="w", padx=18, pady=(16, 7))
        self.preview_frame = Frame(right_panel, bg=self.COLORS["panel2"], highlightbackground=self.COLORS["border"], highlightthickness=1)
        self.preview_frame.pack(fill=BOTH, expand=True, padx=18, pady=(0, 12))
        self.preview_label = Label(self.preview_frame, text="Your image will appear here", bg=self.COLORS["panel2"], fg=self.COLORS["muted"], font=("Segoe UI", 10))
        self.preview_label.pack(fill=BOTH, expand=True)

        self.add_quality_controls(right_panel)

        self.progress = ttk.Progressbar(right_panel, mode="determinate", maximum=100, style="Voice.Horizontal.TProgressbar")
        self.progress.pack(fill=X, padx=18)
        Label(right_panel, textvariable=self.progress_var, **self._label_style(9, False, "muted")).pack(anchor="w", padx=18, pady=(6, 16))

        self._apply_input_mode()
        self._on_keyboard_changed()

    def _panel(self, parent: Frame) -> Frame:
        return Frame(parent, bg=self.COLORS["panel"])

    def _label_style(self, size: int, bold: bool, color_key: str = "text") -> Dict[str, Any]:
        return {"bg": self.COLORS["panel"], "fg": self.COLORS[color_key], "font": ("Segoe UI", size, "bold" if bold else "normal")}

    def add_quality_controls(self, parent: Frame) -> None:
        quality_frame = Frame(parent, bg=self.COLORS["panel"])
        quality_frame.pack(fill=X, padx=18, pady=(0, 12))
        quality_frame.grid_columnconfigure(1, weight=1)

        Label(quality_frame, text="QUALITY", bg=self.COLORS["panel"], fg=self.COLORS["muted"], font=("Segoe UI", 8, "bold")).grid(row=0, column=0, pady=(2, 5), sticky="w")
        Label(quality_frame, textvariable=self.steps_label_var, bg=self.COLORS["panel"], fg=self.COLORS["text"], font=("Segoe UI", 9)).grid(row=0, column=2, pady=(2, 5), sticky="e")

        steps_slider = ttk.Scale(
            quality_frame,
            from_=1,
            to=50,
            variable=self.steps_var,
            orient="horizontal",
            command=lambda _value: self.update_steps_label(),
            style="Voice.Horizontal.TScale",
        )
        steps_slider.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        fast_radio = ttk.Radiobutton(
            quality_frame,
            text="Fast",
            variable=self.quality_mode,
            value="fast",
            command=self.update_quality_settings,
            style="Voice.TRadiobutton",
        )
        fast_radio.grid(row=2, column=0, sticky="w")

        high_radio = ttk.Radiobutton(
            quality_frame,
            text="High quality",
            variable=self.quality_mode,
            value="high",
            command=self.update_quality_settings,
            style="Voice.TRadiobutton",
        )
        high_radio.grid(row=2, column=1, sticky="w", padx=(14, 0))

    def update_steps_label(self) -> None:
        self.steps_label_var.set(f"{int(float(self.steps_var.get()))} steps")

    def update_quality_settings(self) -> None:
        if self.quality_mode.get() == "fast":
            self.steps_var.set(1)
        else:
            self.steps_var.set(4)
        self.update_steps_label()

    def _toggle_ai_enhancer(self) -> None:
        self._set_ai_enhancer(not self.ai_enhancer_enabled)

    def _set_ai_enhancer(self, enabled: bool) -> None:
        self.ai_enhancer_enabled = bool(enabled)
        self.ai_enhancer_label.set(f"AI Enhance: {'On' if self.ai_enhancer_enabled else 'Off'}")
        self._set_status(f"AI prompt enhancement {'enabled' if self.ai_enhancer_enabled else 'disabled'}")

    def _generation_options(self) -> Dict[str, Any]:
        steps = int(float(self.steps_var.get()))
        configured_enhancer = self.config.get("ai_prompt_enhancer", {}) if isinstance(self.config, dict) else {}
        ai_prompt_enhancer = dict(configured_enhancer) if isinstance(configured_enhancer, dict) else {}
        ai_prompt_enhancer["enabled"] = self.ai_enhancer_enabled
        image_config = self.config.get("image", {}) if isinstance(self.config, dict) else {}
        default_size = int(image_config.get("default_size", 768)) if isinstance(image_config, dict) else 768
        return {
            "steps": steps,
            "width": int(image_config.get("width", default_size)) if isinstance(image_config, dict) else default_size,
            "height": int(image_config.get("height", default_size)) if isinstance(image_config, dict) else default_size,
            "guidance_scale": float(image_config.get("guidance_scale", 7.5)) if isinstance(image_config, dict) else 7.5,
            "negative_prompt": "模糊, 像素化, 低质量, 水印, 文字, 丑陋, 畸形, 失真",
            "quality_mode": self.quality_mode.get(),
            "ai_prompt_enhancer": ai_prompt_enhancer,
        }

    def _start_model_initialization(self) -> None:
        threading.Thread(target=self._initialize_modules, daemon=True).start()

    def _initialize_modules(self) -> None:
        self._queue_status("Loading modules...", 5)
        try:
            from utils.config_loader import load_environment_variables, load_system_config
            from utils.global_state import SystemStatus, initialize_global_state, update_system_status
            from utils.logging_setup import setup_logger
            from speech.model_loader import load_vosk_model
            from image.generator import initialize_image_model

            load_environment_variables()
            self.config = load_system_config()
            enhancer_config = self.config.get("ai_prompt_enhancer", {})
            enhancer_enabled = enhancer_config.get("enabled", False) if isinstance(enhancer_config, dict) else False
            self.ui_queue.put(("ai_enhancer", bool(enhancer_enabled)))
            self.logger = setup_logger("VoiceCreateGUI")
            self.global_state = initialize_global_state(self.config)
            update_system_status(self.global_state, SystemStatus.IDLE, "gui-ready")

            self._queue_status("Loading VOSK speech model...", 20)
            vosk_model = load_vosk_model(self.config.get("speech", {}))
            if vosk_model is not None:
                self.global_state["modules"]["speech_recognizer"] = vosk_model

            self._queue_status("Loading image model...", 50)
            image_config = self.config.get("image") or self.config.get("iimage") or {}
            self.image_generator = initialize_image_model(image_config)
            if self.image_generator is not None:
                self.global_state["modules"]["image_generator"] = self.image_generator

            self._queue_status("Ready", 100)
            self.ui_queue.put(("ready", None))
        except Exception as exc:
            self.ui_queue.put(("error", f"Initialization failed: {exc}"))
            traceback.print_exc()

    def _on_record_clicked(self) -> None:
        if self.worker_running:
            self._set_status("A command is already running.")
            return
        self.worker_running = True
        self.is_recording = True
        self.record_button.state(["disabled"])
        self.submit_button.state(["disabled"])
        self.record_button_var.set("Recording...")
        self._set_progress(0, "Recording 2 seconds...")
        self._clear_outputs()
        generation_options = self._generation_options()
        threading.Thread(
            target=self._run_workflow,
            args=("voice", None, generation_options),
            daemon=True,
        ).start()

    def _on_keyboard_submit(self) -> None:
        if self.worker_running:
            self._set_status("A command is already running.")
            return
        text = self._get_keyboard_text()
        try:
            from main_workflow import validate_text_input

            validation = validate_text_input(text)
            self.syntax_var.set(validation.get("message", ""))
            if not validation.get("valid"):
                self._set_status(validation.get("message", "Invalid keyboard command"))
                return
        except Exception:
            pass

        self._add_history(text)
        self.worker_running = True
        self.record_button.state(["disabled"])
        self.submit_button.state(["disabled"])
        self._clear_outputs()
        self._set_progress(0, "Submitting keyboard command...")
        generation_options = self._generation_options()
        threading.Thread(
            target=self._run_workflow,
            args=("keyboard", text, generation_options),
            daemon=True,
        ).start()

    def _run_workflow(
        self,
        input_method: str,
        input_data: Any = None,
        generation_options: Dict[str, Any] | None = None,
    ) -> None:
        try:
            from main_workflow import process_user_input

            self._queue_status("Processing voice command..." if input_method == "voice" else "Processing keyboard command...", 10)
            result = process_user_input(
                input_method,
                input_data,
                global_state=self.global_state,
                image_generator=self.image_generator,
                # Tk windows must be created by the main thread. The generated
                # image is sent back through ui_queue below for display.
                display=False,
                display_time=5000,
                generation_options=generation_options or {},
                progress_callback=lambda step, total: self.ui_queue.put(("generation_progress", (step, total))),
                status_callback=lambda stage, details: self.ui_queue.put(("workflow_detail", (stage, details))),
            )
            self.is_recording = False
            if result.get("error"):
                message = result["error"]
                if result.get("suggestion"):
                    message = f"{message}\n{result['suggestion']}"
                self.ui_queue.put(("error", message))
                if result.get("fallback") == "keyboard":
                    self.ui_queue.put(("mode", "keyboard"))
                return
            self.ui_queue.put(("text", result.get("text", "")))
            self.ui_queue.put(
                (
                    "parse",
                    {
                        "extracted": result.get("command", {}),
                        "parsed": result.get("parsed", {}),
                        "valid": True,
                        "prompt": result.get("prompt", ""),
                        "actual_prompt": result.get("actual_prompt", ""),
                        "save_path": result.get("save_path", ""),
                        "generation_metadata": result.get("generation_metadata", {}),
                    },
                )
            )
            self.ui_queue.put(("preview", result.get("image")))
            self.ui_queue.put(("coords", result.get("coords")))
            self.ui_queue.put(("positioned_display", (result.get("image"), result.get("coords"))))
            if result.get("save_path"):
                self.ui_queue.put(("save_path", result.get("save_path")))
            self._queue_status("Complete", 100)
        except Exception as exc:
            self.ui_queue.put(("error", f"Workflow failed: {exc}"))
            traceback.print_exc()
        finally:
            self.ui_queue.put(("done", None))

    def _queue_status(self, message: str, progress: int) -> None:
        self.ui_queue.put(("status", message))
        self.ui_queue.put(("progress", (progress, message)))

    def _poll_queue(self) -> None:
        try:
            while True:
                event, payload = self.ui_queue.get_nowait()
                self._handle_ui_event(event, payload)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _handle_ui_event(self, event: str, payload: Any) -> None:
        if event == "status":
            self._set_status(str(payload))
        elif event == "progress":
            value, text = payload
            self._set_progress(value, text)
        elif event == "generation_progress":
            step, total = payload
            percent = int((step / max(1, total)) * 100)
            self._set_progress(percent, f"Generating image: step {step}/{total}")
            self._update_generation_detail("progress", f"正在生成图片  {step}/{total}")
        elif event == "workflow_detail":
            stage, details = payload
            self._handle_workflow_detail(str(stage), details or {})
        elif event == "ready":
            self.modules_ready = True
            self._set_status("Ready")
            self._apply_input_mode()
        elif event == "ai_enhancer":
            self._set_ai_enhancer(bool(payload))
        elif event == "text":
            self._set_text(self.text_output, payload)
        elif event == "parse":
            summary = self._format_parse(payload)
            self.generation_details = {"summary": summary}
            self._set_text(self.parse_output, summary)
        elif event == "preview":
            self._show_preview(payload)
        elif event == "positioned_display":
            image, coords = payload
            self._show_positioned_image(image, coords)
        elif event == "coords":
            if payload:
                self._update_generation_detail("position", f"显示位置  ({payload[0]}, {payload[1]})")
        elif event == "save_path":
            self._update_generation_detail("saved", f"已保存  {Path(str(payload)).name}")
            self._set_status(f"Saved: {Path(str(payload)).name}")
        elif event == "display_info":
            self._append_parse(f"\nDisplay info: {payload}")
        elif event == "error":
            self._set_status(str(payload))
            self._set_progress(0, "Error")
            self._append_parse(f"\nERROR: {payload}")
        elif event == "mode":
            self.input_mode_var.set(str(payload))
            self._apply_input_mode()
        elif event == "done":
            self.worker_running = False
            self.is_recording = False
            self.record_button_var.set("Start Recording")
            self._apply_input_mode()

    def _format_parse(self, payload: Dict[str, Any]) -> str:
        extracted = payload.get("extracted", {})
        parsed = payload.get("parsed", {})
        metadata = payload.get("generation_metadata", {})
        actual_prompt = payload.get("actual_prompt") or parsed.get("actual_prompt", "") if isinstance(parsed, dict) else payload.get("actual_prompt", "")
        save_path = payload.get("save_path", "")
        if isinstance(metadata, dict):
            save_path = save_path or metadata.get("save_path", "")
        pipeline_call = metadata.get("pipeline_call", {}) if isinstance(metadata, dict) else {}
        width = pipeline_call.get("width") or metadata.get("width") if isinstance(metadata, dict) else None
        height = pipeline_call.get("height") or metadata.get("height") if isinstance(metadata, dict) else None
        position = extracted.get("position", "") if isinstance(extracted, dict) else ""
        style = extracted.get("style", "") if isinstance(extracted, dict) else ""
        lines = ["生成完成", "", "最终提示词", str(actual_prompt or payload.get("prompt", ""))]
        if position:
            lines.extend(["", f"显示位置  {position}"])
        if style:
            lines.append(f"画面风格  {style}")
        if width and height:
            lines.append(f"图片尺寸  {width} x {height}")
        lines.extend(["保存结果  " + (Path(str(save_path)).name if save_path else "未保存")])
        return "\n".join(lines)

    def _handle_workflow_detail(self, stage: str, details: Dict[str, Any]) -> None:
        if stage == "analyzing":
            self._update_generation_detail("stage", "正在理解指令")
        elif stage == "parsed":
            self._update_generation_detail("stage", "指令理解完成")
            if details.get("subject"):
                self._update_generation_detail("subject", f"画面主体  {details['subject']}")
            if details.get("style"):
                self._update_generation_detail("style", f"画面风格  {details['style']}")
            if details.get("position"):
                self._update_generation_detail("position", f"显示位置  {details['position']}")
        elif stage == "enhancing":
            self._update_generation_detail("enhancer", "正在优化提示词")
        elif stage == "prompt_ready":
            label = "AI 优化后的提示词" if details.get("enhanced") else "生成提示词"
            self._update_generation_detail("enhancer", f"{label}\n{details.get('prompt', '')}")
        elif stage == "generating":
            self._update_generation_detail(
                "settings",
                f"生成设置  {details.get('width')} x {details.get('height')} · {details.get('steps')} 步",
            )
            self._update_generation_detail("progress", "正在生成图片  0/{}".format(details.get("steps", 0)))
        elif stage == "saved":
            self._update_generation_detail("saved", f"已保存  {Path(str(details.get('path', ''))).name}")
        elif stage == "complete":
            self._update_generation_detail("stage", "生成完成")

    def _update_generation_detail(self, key: str, text: str) -> None:
        if not hasattr(self, "generation_details"):
            self.generation_details = {}
        self.generation_details[key] = text
        self._set_text(self.parse_output, "\n\n".join(self.generation_details.values()))

    def _toggle_input_mode(self) -> None:
        self.input_mode_var.set("keyboard" if self.input_mode_var.get() == "voice" else "voice")
        self._apply_input_mode()

    def _apply_input_mode(self) -> None:
        mode = self.input_mode_var.get()
        if self.worker_running or not self.modules_ready:
            self.record_button.state(["disabled"])
            self.submit_button.state(["disabled"])
            if not self.modules_ready:
                self._set_status("Loading image model...")
            return
        if mode == "voice":
            self.mode_button.configure(text="Mode: Voice")
            self.record_button.state(["!disabled"])
            self.submit_button.state(["disabled"])
            self.keyboard_input.configure(state="disabled")
            self._set_status("Voice mode ready")
        else:
            self.mode_button.configure(text="Mode: Keyboard")
            self.record_button.state(["disabled"])
            self.submit_button.state(["!disabled"])
            self.keyboard_input.configure(state="normal")
            self._set_status("Keyboard mode ready")
            self.keyboard_input.focus_set()

    def _on_keyboard_changed(self, _event=None) -> None:
        try:
            from main_workflow import validate_text_input

            validation = validate_text_input(self._get_keyboard_text())
            self.syntax_var.set(validation.get("message", ""))
        except Exception:
            self.syntax_var.set("语法检查暂不可用")

    def _on_keyboard_return(self, _event=None):
        self._on_keyboard_submit()
        return "break"

    def _insert_keyboard_newline(self, _event=None):
        self.keyboard_input.insert("insert", "\n")
        return "break"

    def _get_keyboard_text(self) -> str:
        return self.keyboard_input.get("1.0", END).strip()

    def _add_history(self, text: str) -> None:
        if not text:
            return
        if text in self.input_history:
            self.input_history.remove(text)
        self.input_history.insert(0, text)
        self.input_history = self.input_history[:12]
        self.history_combo.configure(values=self.input_history)
        self.history_var.set(self.input_history[0])

    def _on_history_selected(self, _event=None) -> None:
        value = self.history_var.get()
        if value:
            self.keyboard_input.configure(state="normal")
            self.keyboard_input.delete("1.0", END)
            self.keyboard_input.insert("1.0", value)
            self._on_keyboard_changed()

    def _clear_outputs(self) -> None:
        self.generation_details = {}
        self._set_text(self.text_output, "")
        self._set_text(self.parse_output, "")
        self.preview_label.configure(image="", text="Generating preview...")
        self.preview_photo = None

    def _set_text(self, widget: Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", END)
        widget.insert("1.0", value or "")

    def _append_parse(self, value: str) -> None:
        self.parse_output.insert(END, value)
        self.parse_output.see(END)

    def _show_preview(self, image: Any) -> None:
        try:
            from PIL import Image, ImageTk

            if hasattr(image, "image"):
                image = image.image
            if not isinstance(image, Image.Image):
                image = Image.open(image)
            preview = image.copy()
            preview.thumbnail((330, 285))
            self.preview_photo = ImageTk.PhotoImage(preview)
            self.preview_label.configure(image=self.preview_photo, text="")
        except Exception as exc:
            self.preview_label.configure(text=f"Preview failed: {exc}", image="")

    def _show_positioned_image(self, image: Any, coords: Any) -> None:
        if image is None or not coords:
            return
        try:
            from display.image_display import display_image_at

            display_image_at(image, int(coords[0]), int(coords[1]), display_time=5000)
        except Exception as exc:
            self._set_status(f"Image display failed: {exc}")

    def _set_status(self, value: str) -> None:
        self.status_var.set(value)

    def _set_progress(self, value: int, text: str) -> None:
        self.progress["value"] = value
        self.progress_var.set(text)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    VoiceCreateApp().run()


if __name__ == "__main__":
    main()
