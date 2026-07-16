import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parser.universal_instruction_parser import UniversalInstructionParser
from prompt.intelligent_prompt_optimizer import IntelligentPromptOptimizer
from workflow.unified_workflow import UnifiedVoiceCreateWorkflow
from utils.config_loader import load_config


def test_universal_parser_removes_control_words_and_keeps_semantics():
    parser = UniversalInstructionParser()

    result = parser.parse("在左下角生成一只可爱的小狗在雪地里玩耍，卡通风格")

    assert result["position"] == "左下角"
    assert result["action_verb"] == "生成"
    assert result["quantifier"] == "一只"
    assert "卡通" in result["styles"]
    assert "雪地" in result["scenes"]
    assert result["subject"]
    assert "左下角" not in result["full_description"]
    assert "生成" not in result["full_description"]
    assert result["confidence"] >= 0.3


def test_prompt_optimizer_adds_style_quality_and_mood():
    parser = UniversalInstructionParser()
    optimizer = IntelligentPromptOptimizer({"seed": 1})

    parsed = parser.parse("生成一个温馨的家庭场景，油画风格")
    prompt = optimizer.optimize(parsed, "balanced")

    assert "油画风格" in prompt
    assert "温" in prompt or "舒适" in prompt
    assert prompt.endswith("。")


def test_unified_workflow_batch_dry_run():
    config = load_config("configs/universal.yaml")
    workflow = UnifiedVoiceCreateWorkflow(config, dry_run=True)
    cases = [
        "在左上角生成一只狗",
        "用水墨画风格生成李白",
        "在右上角显示一个科幻城市夜景，赛博朋克风格",
        "生成一个浪漫的日落",
    ]

    results = workflow.batch_test(cases)

    assert results["total"] == len(cases)
    assert results["success"] == len(cases)
    assert results["success_rate"] == 1
    assert all(item["prompt"] for item in results["results"])
