import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image.dreamlite_fixed import DreamLiteModel
from utils.image_manager import ImageManager


def test_dreamlite_generate_saves_image_and_metadata(tmp_path):
    save_dir = tmp_path / "generated_images"
    model = DreamLiteModel(
        {
            "model_path": str(tmp_path / "missing_model"),
            "device": "cpu",
            "default_size": 128,
            "steps": 2,
            "save_generated": True,
            "save_path": str(save_dir),
            "save_metadata": True,
            "max_saved_images": 10,
        }
    )
    # This test exercises saving, while production loading intentionally fails fast.
    assert model._load_mock()

    result = model.generate("测试图片 保存功能", width=128, height=128, steps=1)

    assert not result.success
    assert result.metadata["fallback"] == "mock"
    assert result.metadata["error"] == "Real model not available, using placeholder"
    assert result.image is not None
    save_path = result.metadata.get("save_path")
    assert save_path
    saved_file = Path(save_path)
    assert saved_file.exists()
    assert saved_file.parent == save_dir
    metadata_path = saved_file.with_suffix(".json")
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["prompt"] == "测试图片 保存功能"


def test_image_manager_lists_saved_images(tmp_path):
    save_dir = tmp_path / "images"
    save_dir.mkdir()
    (save_dir / "a.png").write_bytes(b"png")
    manager = ImageManager({"save_path": str(save_dir)})

    assert manager.get_image_count() == 1
    assert manager.get_latest_image().endswith("a.png")
    assert manager.list_images()[0]["filename"] == "a.png"
