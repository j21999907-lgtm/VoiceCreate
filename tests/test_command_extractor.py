from src.speech.command_extractor import extract_command_from_text, validate_speech_command


def test_daily_animal_scene_extraction():
    result = extract_command_from_text("在左上角生成一只可爱的小狗在草地上玩耍")

    assert result["position"] == "左上角"
    assert result["raw_keywords"] == ["可爱", "小狗", "草地", "玩耍"]
    assert result["categorized"]["animals"] == ["狗"]
    assert result["categorized"]["scenes"] == ["草地"]
    assert result["categorized"]["actions"] == ["玩耍"]
    assert result["theme"] == "animal_scene"
    assert "可爱的狗" in result["enhanced_prompt"]
    assert validate_speech_command(result)


def test_style_and_city_scene_extraction():
    result = extract_command_from_text("在中心位置创建一幅科幻风格的城市夜景")

    assert result["position"] == "中心"
    assert result["categorized"]["styles"] == ["科幻风格"]
    assert result["categorized"]["scenes"] == ["城市", "夜景"]
    assert result["theme"] == "cityscape"
    assert "科幻风格" in result["enhanced_prompt"]


def test_single_character_action_is_preserved_when_meaningful():
    result = extract_command_from_text("画一只可爱的大熊猫在吃竹子，背景是竹林")

    assert result["categorized"]["animals"] == ["熊猫"]
    assert "吃" in result["categorized"]["actions"]
    assert "正在吃" in result["enhanced_prompt"]


def test_free_form_wearable_attribute_is_preserved():
    result = extract_command_from_text("在左上角生成一只戴眼镜的猫，水彩风格")

    assert result["keywords"] == ["戴眼镜", "猫", "水彩"]
    assert "戴眼镜" in result["enhanced_prompt"]
    assert "猫" in result["enhanced_prompt"]
    assert "水彩风格" in result["enhanced_prompt"]


def test_unlisted_descriptive_phrases_are_not_discarded():
    result = extract_command_from_text("生成一个穿红色西装、拿着雨伞的人")

    assert "穿红色西装" in result["keywords"]
    assert "拿着雨伞" in result["keywords"]
    assert all(keyword in result["enhanced_prompt"] for keyword in ("穿红色西装", "拿着雨伞"))
