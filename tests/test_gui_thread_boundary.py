from pathlib import Path
import queue
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gui.main_application import VoiceCreateApp


def test_worker_defers_tk_display_to_ui_queue(monkeypatch):
    calls = []
    image = object()

    def process_user_input(*args, **kwargs):
        calls.append((args, kwargs))
        return {"image": image, "coords": (10, 20), "command": {}, "parsed": {}}

    monkeypatch.setitem(
        sys.modules,
        "main_workflow",
        types.SimpleNamespace(process_user_input=process_user_input),
    )

    app = VoiceCreateApp.__new__(VoiceCreateApp)
    app.ui_queue = queue.Queue()
    app.global_state = {"modules": {}}
    app.image_generator = object()
    app.is_recording = False

    options = {"steps": 2}
    app._run_workflow("keyboard", "生成一只猫", options)

    assert calls[0][1]["display"] is False
    assert calls[0][1]["generation_options"] is options
    events = []
    while not app.ui_queue.empty():
        events.append(app.ui_queue.get_nowait())
    assert ("positioned_display", (image, (10, 20))) in events
