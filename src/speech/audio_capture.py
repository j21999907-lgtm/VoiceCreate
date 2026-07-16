

import logging

logger = logging.getLogger("VoiceCreate")

# src/speech/audio_capture.py
"""
音频采集模块
功能：从系统默认麦克风录制指定时长的原始音频数据。
注意：使用前需安装 pyaudio 库: `pip install pyaudio`
"""

import sys
import time
import traceback
from typing import Optional, Union, Tuple
import threading

# 尝试导入 pyaudio
try:
    import pyaudio
    import numpy as np

    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    np = None
    PYAUDIO_AVAILABLE = False
    logger.info("[警告] 未找到 pyaudio 或 numpy 库。音频采集将使用模拟模式。")
    logger.info("       请执行: pip install pyaudio numpy")

class AudioCaptureDevice:
    """音频采集设备管理类"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_size: int = 1024, device_index: Optional[int] = None):
        """
        初始化音频采集参数

        参数:
            sample_rate: 采样率 (Hz)，必须与VOSK模型匹配（通常为16000）
            channels: 声道数，1=单声道
            chunk_size: 每次从音频流读取的帧数
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        self.audio_interface = None
        self.stream = None
        self.is_initialized = False
        self.device_index = device_index
        self.last_volume = 0.0

    def initialize(self) -> bool:
        """初始化音频设备"""
        if not PYAUDIO_AVAILABLE:
            logger.info("[音频] 使用模拟音频设备（pyaudio不可用）")
            self.is_initialized = True
            return True

        try:
            self.audio_interface = pyaudio.PyAudio()

            # 获取默认输入设备的索引
            try:
                default_device_index = self.device_index
                if default_device_index is None:
                    default_device_index = self.audio_interface.get_default_input_device_info()["index"]
                device_info = self.audio_interface.get_device_info_by_index(default_device_index)
                logger.info(f"[音频] 使用音频设备: {device_info.get('name', '未知设备')}")
                logger.info(f"      采样率: {self.sample_rate}Hz, 声道: {self.channels}")
                # 检查设备是否支持指定采样率
                supported_rate = device_info.get('defaultSampleRate', self.sample_rate)
                if float(supported_rate) != self.sample_rate:
                    logger.info(f"[警告] 设备默认采样率为 {supported_rate}Hz，但将使用 {self.sample_rate}Hz")
            except (IOError, OSError) as e:
                logger.info(f"[警告] 无法获取默认输入设备: {e}")
                logger.info(f"[音频] 将尝试使用第一个可用的输入设备")
                # 尝试查找第一个可用的输入设备
                for i in range(self.audio_interface.get_device_count()):
                    device_info = self.audio_interface.get_device_info_by_index(i)
                    if device_info.get('maxInputChannels', 0) > 0:
                        logger.info(f"[音频] 使用音频设备: {device_info.get('name', '未知设备')}")
                        break
                else:
                    logger.info("[错误] 未找到可用的音频输入设备")
                    return False

            self.is_initialized = True
            return True

        except Exception as e:
            logger.info(f"[错误] 初始化音频设备失败: {e}")
            traceback.print_exc()
            return False

    def capture_audio_chunk(self, duration_seconds: float = 2.0) -> Optional[bytes]:
        """
        录制一段音频

        参数:
            duration_seconds: 录制时长（秒）

        返回:
            包含PCM音频数据的bytes对象，如果失败则返回None
        """
        if not self.is_initialized and not self.initialize():
            return None

        if not PYAUDIO_AVAILABLE:
            # 模拟模式：返回静音数据
            return self._generate_silence_chunk(duration_seconds)

        try:
            # 计算需要读取的总帧数
            total_frames = int(self.sample_rate * duration_seconds)
            frames_per_chunk = self.chunk_size
            chunks_to_read = total_frames // frames_per_chunk

            # 确保至少读取一个块
            if chunks_to_read == 0:
                chunks_to_read = 1
                frames_per_chunk = total_frames

            logger.info(f"[音频] 开始录制，时长: {duration_seconds:.1f}秒，总帧数: {total_frames}")
            # 打开音频流
            self.stream = self.audio_interface.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=frames_per_chunk,
                stream_callback=None
            )

            audio_data = bytearray()

            # 分块读取音频数据
            for i in range(chunks_to_read):
                try:
                    data = self.stream.read(frames_per_chunk, exception_on_overflow=False)
                    audio_data.extend(data)

                    # 可选：显示录制进度
                    if i % 10 == 0:  # 每10个块打印一次进度
                        progress = (i + 1) / chunks_to_read * 100
                        logger.info(f"[音频] 录制进度: {progress:.0f}%")
                except IOError as e:
                    logger.info(f"[警告] 音频读取错误: {e}")
                    break

            logger.info("")  # 换行
            logger.info(f"[音频] 录制完成，数据大小: {len(audio_data)} 字节")
            # 关闭流
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
            self.stream = None

            captured = bytes(audio_data)
            self.last_volume = calculate_audio_volume(captured)
            return captured

        except Exception as e:
            logger.info(f"[错误] 录制音频时发生异常: {e}")
            traceback.print_exc()

            # 确保流被关闭
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            return None

    def _generate_silence_chunk(self, duration_seconds: float) -> bytes:
        """生成模拟的静音数据（用于测试）"""
        if np is None:
            # 如果没有numpy，返回空的bytes
            logger.info(f"[音频] 模拟录制 {duration_seconds:.1f}秒静音")
            return b'\x00' * int(self.sample_rate * duration_seconds * 2)  # 16位 = 2字节

        # 使用numpy生成静音
        num_samples = int(self.sample_rate * duration_seconds)
        silence = np.zeros(num_samples, dtype=np.int16)
        logger.info(f"[音频] 模拟录制 {duration_seconds:.1f}秒静音 ({num_samples}个样本)")
        return silence.tobytes()

    def close(self):
        """关闭音频设备"""
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.audio_interface:
            self.audio_interface.terminate()
            self.audio_interface = None

        self.is_initialized = False
        logger.info("[音频] 音频设备已关闭")

# 全局音频设备实例（单例模式）
_global_audio_device = None


def get_audio_device(sample_rate: int = 16000, device_index: Optional[int] = None) -> AudioCaptureDevice:
    """获取全局音频设备实例（单例）"""
    global _global_audio_device
    if (
        _global_audio_device is None
        or _global_audio_device.sample_rate != sample_rate
        or _global_audio_device.device_index != device_index
    ):
        _global_audio_device = AudioCaptureDevice(sample_rate=sample_rate, device_index=device_index)
    return _global_audio_device


def capture_audio_chunk(duration_seconds: float = 2.0, sample_rate: int = 16000, device_index: Optional[int] = None) -> Optional[bytes]:
    """
    录制指定时长的音频（便捷函数）

    这是对外的接口函数，与文档中的函数签名一致。

    参数:
        duration_seconds: 录制时长（秒），默认2.0秒
        sample_rate: 采样率（Hz），默认16000

    返回:
        包含PCM音频数据的bytes对象
    """
    device = get_audio_device(sample_rate, device_index=device_index)
    return device.capture_audio_chunk(duration_seconds)


def calculate_audio_volume(audio_data: bytes) -> float:
    """计算16-bit PCM音频的RMS音量。"""
    if not audio_data or np is None:
        return 0.0
    try:
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        if audio_array.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_array ** 2)))
    except Exception:
        traceback.print_exc()
        return 0.0


def select_audio_device(preferred_index: Optional[int] = None) -> Optional[int]:
    """选择可用输入设备，优先使用preferred_index。"""
    if not PYAUDIO_AVAILABLE:
        return None
    p = None
    try:
        p = pyaudio.PyAudio()
        if preferred_index is not None:
            info = p.get_device_info_by_index(preferred_index)
            if info.get("maxInputChannels", 0) > 0:
                return preferred_index
        default_info = p.get_default_input_device_info()
        if default_info and default_info.get("maxInputChannels", 0) > 0:
            return int(default_info["index"])
        for index in range(p.get_device_count()):
            info = p.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0) > 0:
                return index
    except Exception:
        traceback.print_exc()
    finally:
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass
    return None


def test_audio_capture():
    """测试音频采集功能"""
    logger.info("=" * 60)
    logger.info("音频采集模块测试")
    logger.info("=" * 60)

    # 测试1: 初始化设备
    logger.info("1. 测试音频设备初始化...")
    device = AudioCaptureDevice(sample_rate=16000)

    if device.initialize():
        logger.info("✅ 音频设备初始化成功")
    else:
        logger.info("❌ 音频设备初始化失败")
        if not PYAUDIO_AVAILABLE:
            logger.info("   注意: 正在使用模拟模式")
    # 测试2: 录制短音频
    logger.info("\n2. 测试音频录制（0.5秒）...")
    audio_data = device.capture_audio_chunk(duration_seconds=0.5)

    if audio_data is not None:
        data_size_kb = len(audio_data) / 1024
        logger.info(f"✅ 音频录制成功")
        logger.info(f"   数据大小: {data_size_kb:.1f} KB")
        logger.info(f"   数据类型: {type(audio_data).__name__}")
        # 检查是否是静音
        is_silent = True
        if audio_data:
            # 检查前100个字节是否都是0
            test_bytes = audio_data[:100]
            if all(b == 0 for b in test_bytes):
                logger.info("   数据内容: 静音（可能麦克风未连接或静音）")
            else:
                is_silent = False
                logger.info("   数据内容: 检测到音频信号")
        else:
            logger.info("   数据内容: 空数据")
    else:
        logger.info("❌ 音频录制失败")
    # 测试3: 测试便捷函数
    logger.info("\n3. 测试便捷函数 capture_audio_chunk()...")
    test_audio = capture_audio_chunk(0.3, 16000)
    if test_audio:
        logger.info(f"✅ 便捷函数测试通过，数据长度: {len(test_audio)} 字节")
    else:
        logger.info("❌ 便捷函数测试失败")
    # 清理
    device.close()

    logger.info("=" * 60)
    logger.info("音频采集测试完成")
    logger.info("=" * 60)
    return audio_data is not None


# 辅助函数：列出所有音频设备
def list_audio_devices():
    """列出所有可用的音频设备"""
    if not PYAUDIO_AVAILABLE:
        logger.info("pyaudio 不可用，无法列出音频设备")
        return

    try:
        p = pyaudio.PyAudio()
        logger.info("可用的音频设备:")
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            input_channels = dev_info.get('maxInputChannels', 0)
            output_channels = dev_info.get('maxOutputChannels', 0)

            device_type = []
            if input_channels > 0:
                device_type.append(f"输入({input_channels}通道)")
            if output_channels > 0:
                device_type.append(f"输出({output_channels}通道)")

            logger.info(f"  [{i}] {dev_info.get('name', '未知设备')} ({', '.join(device_type)})")
            logger.info(f"      默认采样率: {dev_info.get('defaultSampleRate', '未知')}Hz")
        p.terminate()
    except Exception as e:
        logger.info(f"列出音频设备时出错: {e}")

# 当此文件被直接运行时，执行测试
if __name__ == "__main__":
    # 首先列出音频设备
    list_audio_devices()
    logger.info("")

    # 运行主测试
    test_audio_capture()
