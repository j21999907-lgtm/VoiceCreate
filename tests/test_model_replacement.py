import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image.generator import DiffusersImageModel, ModelStatus


class _PipelineOutput:
    def __init__(self):
        from PIL import Image

        self.images = [Image.new("RGB", (64, 64))]


class _TurboPipeline:
    def __init__(self):
        self.call = None

    def __call__(self, **kwargs):
        self.call = kwargs
        return _PipelineOutput()


def test_sd_turbo_keeps_low_step_count(tmp_path):
    model_dir = tmp_path / "sd-turbo"
    for component in ("unet", "vae"):
        path = model_dir / component
        path.mkdir(parents=True)
        (path / "weights.safetensors").write_bytes(b"weights")
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline", "_name_or_path": "stabilityai/sd-turbo"}),
        encoding="utf-8",
    )

    model = DiffusersImageModel(
        {
            "model_path": str(model_dir),
            "model_type": "sd-turbo",
            "device": "cpu",
            "steps": 4,
            "guidance_scale": 0.0,
            "save_generated": False,
        }
    )
    model.pipe = _TurboPipeline()
    model.status = ModelStatus.LOADED

    result = model.generate("test", width=64, height=64, steps=2)

    assert result.success
    assert model.pipe.call["num_inference_steps"] == 2
    assert model.pipe.call["guidance_scale"] == 0.0


def test_sd_turbo_uses_turbo_defaults_when_generation_values_are_omitted(tmp_path):
    model_dir = tmp_path / "sd-turbo"
    for component in ("unet", "vae"):
        path = model_dir / component
        path.mkdir(parents=True)
        (path / "weights.safetensors").write_bytes(b"weights")
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline", "_name_or_path": "stabilityai/sd-turbo"}),
        encoding="utf-8",
    )

    model = DiffusersImageModel({"model_path": str(model_dir), "model_type": "sd-turbo", "save_generated": False})

    assert model.default_size == 512
    assert model.steps == 4
    assert model.guidance_scale == 0.0
