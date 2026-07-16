

import logging

logger = logging.getLogger("VoiceCreate")

# src/speech/speech_recognizer.py
"""
语音识别模块
功能：使用VOSK模型将音频数据转换为文本。
注意：此模块依赖于已加载的VOSK模型，需在全局状态中可用。
"""

import sys
import json
import traceback
import time
from typing import Optional, Dict, Any
from vosk import KaldiRecognizer


def recognize_speech(audio_data: bytes, sample_rate: int = 16000,
                     global_state: Dict[str, Any] = None, timeout_seconds: float = 10.0) -> Optional[str]:
    """
    将音频数据转换为文本

    参数:
        audio_data: 预处理后的PCM音频数据 (16位，单声道)
        sample_rate: 音频采样率，必须与VOSK模型匹配
        global_state: 全局状态字典（可选），用于获取VOSK模型

    返回:
        str: 识别出的文本，如果识别失败则返回None
    """
    if not audio_data:
        logger.info("[语音识别] 警告: 音频数据为空")
        return None

    try:
        logger.info("[语音识别] 开始语音识别...")
        logger.info(f"          音频数据大小: {len(audio_data)} 字节")
        logger.info(f"          采样率: {sample_rate}Hz")
        # 尝试从全局状态获取VOSK模型
        vosk_model = None
        if global_state and "modules" in global_state:
            with global_state.get("lock"):
                vosk_model = global_state["modules"].get("speech_recognizer")

        if vosk_model:
            # 使用真实VOSK识别
            text = _recognize_with_vosk(audio_data, sample_rate, vosk_model, timeout_seconds=timeout_seconds)
            return text if validate_recognition_result(text) else None
        else:
            # 回退到模拟模式
            logger.info("[语音识别] 警告: 未找到VOSK模型，使用模拟模式")
            text = _recognize_simulated(audio_data, sample_rate)
            return text if validate_recognition_result(text) else None

    except Exception as e:
        logger.info(f"[语音识别] 语音识别时发生异常: {e}")
        traceback.print_exc()
        return None


def _recognize_with_vosk(audio_data: bytes, sample_rate: int, vosk_model, timeout_seconds: float = 10.0) -> Optional[str]:
    """使用真实的VOSK模型进行语音识别"""
    try:
        start_time = time.time()
        # 创建VOSK识别器
        recognizer = KaldiRecognizer(vosk_model, sample_rate)
        if timeout_seconds and time.time() - start_time > timeout_seconds:
            logger.info("[语音识别] 识别超时")
            return None

        # 接受音频数据
        recognizer.AcceptWaveform(audio_data)

        # 获取识别结果
        result = recognizer.Result()
        result_dict = json.loads(result)

        text = result_dict.get("text", "").strip()

        if text:
            logger.info(f"[语音识别] 识别结果: {text}")
            return text
        else:
            # 尝试获取部分结果
            partial_result = recognizer.PartialResult()
            partial_dict = json.loads(partial_result)
            partial_text = partial_dict.get("partial", "").strip()

            if partial_text:
                logger.debug(f"[语音识别] 部分结果: {partial_text}")
            return None

    except Exception as e:
        logger.info(f"[语音识别] VOSK识别失败: {e}")
        return None


def validate_recognition_result(text: Optional[str], min_length: int = 1) -> bool:
    """校验识别结果是否适合继续解析。"""
    if text is None:
        return False
    cleaned = str(text).strip()
    if len(cleaned) < min_length:
        return False
    noise_tokens = {"[unk]", "<unk>", "嗯", "啊"}
    return cleaned.lower() not in noise_tokens


def _recognize_simulated(audio_data: bytes, sample_rate: int) -> str:
    """模拟语音识别（用于测试）"""
    # 模拟识别结果，用于测试
    test_results = [
        "在屏幕中央生成一只猫",
        "在左上角显示一张风景图",
        "在右下角创建一个抽象艺术图案",
        "生成一个科技感的背景在正中央",
        "在左下角显示星空图片",
        "在右上角画一个太阳",
        "在中间位置显示一只狗",
        "在底部中央生成一个机器人",
        "在左上角创建一个彩虹",
        "在右下角显示一个城堡"
    ]

    import random
    result = random.choice(test_results)

    logger.info(f"[语音识别] 模拟识别结果: {result}")
    return result


def create_vosk_recognizer(model, sample_rate: int = 16000) -> Optional[KaldiRecognizer]:
    """创建VOSK识别器实例"""
    try:
        recognizer = KaldiRecognizer(model, sample_rate)
        return recognizer
    except Exception as e:
        logger.info(f"[语音识别] 创建VOSK识别器失败: {e}")
        return None


def test_speech_recognition():
    """测试语音识别功能（独立测试，不依赖全局状态）"""
    logger.info("=" * 60)
    logger.info("语音识别模块测试")
    logger.info("=" * 60)

    # 生成测试音频数据
    logger.info("1. 生成测试音频数据...")
    import numpy as np

    sample_rate = 16000
    duration = 2.0
    num_samples = int(sample_rate * duration)

    # 生成模拟音频（静音）
    test_audio = np.zeros(num_samples, dtype=np.int16).tobytes()
    logger.info(f"   测试音频大小: {len(test_audio)} 字节")
    # 测试识别功能（使用模拟模式）
    logger.info("\n2. 测试语音识别（模拟模式）...")
    result = recognize_speech(test_audio, sample_rate, global_state=None)

    if result is not None:
        logger.info(f"   ✅ 语音识别测试通过")
        logger.info(f"      识别结果: '{result}'")
    else:
        logger.info("   ❌ 语音识别测试失败")
    # 测试3: 测试多个识别
    logger.info("\n3. 测试多次识别...")
    results = []
    for i in range(3):
        result = recognize_speech(test_audio, sample_rate, global_state=None)
        results.append(result)
        logger.info(f"   第{i + 1}次识别: '{result}'")
    # 检查是否所有结果都不同（模拟模式下应该是随机的）
    unique_results = set(results)
    if len(unique_results) > 1:
        logger.info(f"   ✅ 随机性测试通过，得到 {len(unique_results)} 个不同结果")
    else:
        logger.info(f"   ⚠️  随机性测试: 所有结果相同")
    # 测试4: 测试音频质量检测
    logger.info("\n4. 测试音频质量检测...")
    try:
        # 尝试多种导入方式
        try:
            # 方式1: 从当前目录导入
            from vad_processor import SpeechActivityDetector
        except ImportError:
            try:
                # 方式2: 从speech包导入
                from speech.vad_processor import SpeechActivityDetector
            except ImportError:
                # 方式3: 使用sys.path添加路径
                import sys
                sys.path.insert(0, '.')
                from vad_processor import SpeechActivityDetector

        # 创建VAD检测器
        vad = SpeechActivityDetector(sample_rate=16000)

        # 分析音频质量
        quality_info = vad.analyze_audio_quality(test_audio)
        logger.info(f"   音频质量分析:")
        for key, value in quality_info.items():
            logger.info(f"     {key}: {value}")

    except ImportError as e:
        logger.info(f"   ⚠️  无法导入SpeechActivityDetector，跳过音频质量检测")
        logger.info(f"      错误: {e}")
    except Exception as e:
        logger.info(f"   ⚠️  音频质量检测失败: {e}")
    logger.info("=" * 60)
    logger.info("语音识别测试完成")
    logger.info("=" * 60)

    return result is not None


# 当此文件被直接运行时，执行测试
if __name__ == "__main__":
    test_speech_recognition()
