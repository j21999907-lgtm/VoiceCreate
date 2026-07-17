import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from main_workflow import VoiceCreateWorkflow, WorkflowContext


class _Result:
    success = True
    image = object()
    metadata = {}


class _TurboGenerator:
    default_size = 512
    steps = 4
    guidance_scale = 0.0
    negative_prompt = "model negative prompt"

    def __init__(self):
        self.call = None

    def generate(self, **kwargs):
        self.call = kwargs
        return _Result()


def test_parser_quality_preset_does_not_override_model_defaults():
    generator = _TurboGenerator()
    workflow = VoiceCreateWorkflow(
        WorkflowContext(global_state={"modules": {}}, image_generator=generator, generation_options={})
    )

    workflow.generate_image(
        "a cat",
        {
            "parameters": {
                "width": 512,
                "height": 512,
                "steps": 35,
                "guidance_scale": 7.5,
                "negative_prompt": "generic parser preset",
            }
        },
    )

    assert generator.call["steps"] == 4
    assert generator.call["guidance_scale"] == 0.0
    assert generator.call["negative_prompt"] == "model negative prompt"


def test_explicit_generation_options_override_model_defaults():
    generator = _TurboGenerator()
    workflow = VoiceCreateWorkflow(
        WorkflowContext(
            global_state={"modules": {}},
            image_generator=generator,
            generation_options={"steps": 2, "guidance_scale": 1.0},
        )
    )

    workflow.generate_image("a cat", {"parameters": {"steps": 35}})

    assert generator.call["steps"] == 2
    assert generator.call["guidance_scale"] == 1.0
