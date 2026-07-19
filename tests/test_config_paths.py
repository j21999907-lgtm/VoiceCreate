from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.config_loader import _normalize_paths, load_system_config


def test_default_config_uses_one_portable_image_section():
    config = load_system_config()

    assert "iimage" not in config
    assert Path(config["image"]["model_path"]) == (ROOT / "models" / "sd-turbo").resolve()
    assert Path(config["speech"]["model_path"]) == (ROOT / "models" / "vosk").resolve()


def test_legacy_iimage_config_is_migrated():
    config = _normalize_paths({"iimage": {"model_path": "./models/example"}})

    assert "iimage" not in config
    assert Path(config["image"]["model_path"]) == (ROOT / "models" / "example").resolve()
