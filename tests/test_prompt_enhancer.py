from src.ai.prompt_enhancer import AIPromptEnhancer


def test_prompt_enhancer_defaults_to_qwen2_5_7b():
    enhancer = AIPromptEnhancer()

    assert enhancer.model == "qwen2.5:7b"


def test_enhance_uses_sentence_prompt_system_prompt_and_user_message(monkeypatch):
    captured = {}

    class FakeResponse:
        text = '{"message": {"content": "A cute cat in a sunlit garden, soft lighting, balanced composition, photorealistic style, rich detail, 4K quality"}}'

        def raise_for_status(self):
            return None

    def fake_post(url, json, timeout):
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr("src.ai.prompt_enhancer.requests.post", fake_post)

    result = AIPromptEnhancer().enhance("猫，戴眼镜，水彩风格", required_keywords=["猫", "戴眼镜", "水彩"])

    messages = captured["payload"]["messages"]
    assert captured["payload"]["model"] == "qwen2.5:7b"
    assert "不得省略、合并或改变含义" in messages[0]["content"]
    assert "必须保留关键词列表中的每一项" in messages[0]["content"]
    assert "原始提示词：猫，戴眼镜，水彩风格" in messages[1]["content"]
    assert "必须逐项保留的关键词：猫 | 戴眼镜 | 水彩" in messages[1]["content"]
    assert result.startswith("A cute cat")


def test_clean_prompt_removes_dialog_wrapping():
    raw = """
    输出：
    "A cute cat sitting on grass, soft natural light, detailed fur, high detail, 4K quality."
    解释：这是优化后的提示词。
    """

    cleaned = AIPromptEnhancer._clean_prompt(raw)

    assert cleaned == "A cute cat sitting on grass, soft natural light, detailed fur, high detail, 4K quality"


def test_clean_prompt_handles_prefixed_single_line_output():
    cleaned = AIPromptEnhancer._clean_prompt(
        "Final prompt: A futuristic city at night, neon lights, cinematic composition, high detail"
    )

    assert cleaned == "A futuristic city at night, neon lights, cinematic composition, high detail"


def test_connection_refused_falls_back_without_retrying(monkeypatch):
    calls = 0

    def refused(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AIPromptEnhancerConnectionError("Ollama is not running")

    from requests import ConnectionError as AIPromptEnhancerConnectionError

    monkeypatch.setattr("src.ai.prompt_enhancer.requests.post", refused)

    result = AIPromptEnhancer().enhance("a cat")

    assert calls == 1
    assert result.startswith("a cat")
