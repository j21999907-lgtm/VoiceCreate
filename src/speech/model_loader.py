

import logging

logger = logging.getLogger("VoiceCreate")

# src/speech/model_loader.py
"""
VOSK语音识别模型加载器
功能：加载离线语音识别模型，为语音转文本功能做好准备。

使用说明：
1. 确保已安装 vosk 库: pip install vosk
2. 从 https://alphacephei.com/vosk/models 下载所需语音模型（如中文模型 vosk-model-small-cn-0.22）
3. 将下载的模型压缩包解压，并将整个模型文件夹放置于项目指定路径下（默认为 ./models/vosk/）
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple, List

# 获取项目根目录的绝对路径
# __file__ 是当前文件路径，.parent.parent 回到 src，.parent 回到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger.info(f"[信息] 项目根目录: {PROJECT_ROOT}")
# 尝试导入VOSK，如果失败会提供明确指引
VOSK_AVAILABLE = False
try:
    import vosk

    VOSK_AVAILABLE = True
except ImportError as e:
    logger.info(f"[警告] 无法导入 vosk 库。语音识别功能将不可用。")
    logger.info(f"       错误详情: {e}")
    logger.info(f"       请执行: pip install vosk")

def find_vosk_model_directory(search_paths: List[str]) -> Optional[str]:
    """
    在给定的多个路径中查找有效的VOSK模型目录。

    VOSK模型目录有效的标志是：目录存在，并且包含必要的模型文件（如 'am/final.mdl'）。

    参数:
        search_paths: 待搜索的路径列表。

    返回:
        第一个找到的有效模型目录的完整路径。如果都未找到，返回None。
    """
    necessary_files = ['am/final.mdl', 'graph/HCLG.fst', 'graph/words.txt']

    for model_path in search_paths:
        if not model_path or not isinstance(model_path, str):
            continue

        # 将相对路径转换为基于项目根目录的绝对路径
        if not os.path.isabs(model_path):
            full_path = PROJECT_ROOT / model_path
        else:
            full_path = Path(model_path)

        full_path = full_path.resolve()
        logger.info(f"[探测] 检查模型路径: {full_path}")
        if not full_path.exists():
            logger.info(f"      -> 路径不存在，跳过。")
            continue

        if not full_path.is_dir():
            logger.info(f"      -> 不是目录，跳过。")
            continue

        # 检查关键文件是否存在
        is_valid = all((full_path / file).exists() for file in necessary_files)
        if is_valid:
            logger.info(f"      -> ✅ 是有效的VOSK模型目录。")
            return str(full_path)
        else:
            # 检查是否有其他常见的VOSK模型文件结构
            if (full_path / "am" / "final.mdl").exists():
                logger.info(f"      -> ✅ 是有效的VOSK模型目录 (已找到 am/final.mdl)。")
                return str(full_path)
            logger.info(f"      -> ⚠️  目录存在，但缺少关键模型文件，可能不是标准VOSK模型。")
    return None


def load_vosk_model(model_config: dict) -> Optional['vosk.Model']:
    """
    从配置中加载VOSK语音识别模型。

    参数:
        model_config: 包含模型配置的字典。例如:
            {
                "model_path": "./models/vosk",  # 主路径
                "language": "zh-cn",            # 语言代码，用于后备路径生成
                "sample_rate": 16000
            }

    返回:
        加载成功的vosk.Model对象。如果加载失败，返回None。
    """
    if not VOSK_AVAILABLE:
        logger.info("[错误] 无法加载模型，因为 'vosk' 库未安装。")
        logger.info("       解决方案: 在终端中执行 'pip install vosk'")
        return None

    # 1. 构建可能的模型搜索路径列表
    base_paths = []
    configured_path = model_config.get("model_path")
    if configured_path:
        base_paths.append(configured_path)

    # 添加基于语言的后备路径（常见的模型命名方式）
    language = model_config.get("language", "zh-cn")
    if language == "zh-cn":
        base_paths.extend([
            "./models/vosk-model-small-cn-0.22",  # 中文小模型常见命名
            "./models/vosk-model-cn-0.22",  # 中文大模型常见命名
            "./models/vosk",  # 通用命名
        ])
    elif language == "en-us":
        base_paths.extend([
            "./models/vosk-model-small-en-us-0.15",
            "./models/vosk-model-en-us-0.22",
        ])
    # 可以为其他语言添加更多后备路径

    # 2. 查找有效的模型目录
    logger.info(f"\n[信息] 开始查找VOSK语音模型（语言: {language}）...")
    logger.info(f"[信息] 项目根目录: {PROJECT_ROOT}")
    effective_model_path = find_vosk_model_directory(base_paths)

    if not effective_model_path:
        logger.info(f"\n[错误] 未能在以下任何路径中找到有效的VOSK模型目录：")
        for p in base_paths:
            # 显示完整的绝对路径以便调试
            if not os.path.isabs(p):
                abs_path = PROJECT_ROOT / p
            else:
                abs_path = Path(p)
            logger.info(f"  - {abs_path.resolve()}")
        logger.info(f"\n请执行以下步骤：")
        logger.info(f"  1. 从 https://alphacephei.com/vosk/models 下载模型（例如 vosk-model-small-cn-0.22.zip）")
        logger.info(f"  2. 将ZIP文件解压。")
        logger.info(f"  3. 将解压出的文件夹（如 'vosk-model-small-cn-0.22'）放置于项目根目录的 'models/' 下。")
        logger.info(f"  4. 您可以选择：")
        logger.info(f"     a) 将文件夹重命名为 'vosk'")
        logger.info(f"     b) 或者修改配置文件 'configs/default.yaml' 中的 'speech.model_path' 为您文件夹的实际名称")
        return None

    # 3. 加载模型
    logger.info(f"\n[信息] 正在从以下路径加载VOSK模型，请稍候...")
    logger.info(f"       {effective_model_path}")
    try:
        # 此步骤可能消耗一定时间和内存（尤其是大模型）
        model = vosk.Model(effective_model_path)
        logger.info(f"[信息] ✅ VOSK模型加载成功！")
        logger.info(f"       模型路径: {effective_model_path}")
        # 获取并显示模型信息（如果接口支持）
        try:
            # 注意：VOSK Model 类不一定有 get_model_info 方法，这里只是示例
            # model_info = model.get_model_info()
            # print(f"       模型信息: {model_info}")
            pass
        except AttributeError:
            # 如果获取模型信息的方法不存在，静默跳过
            pass

        return model

    except Exception as e:
        logger.info(f"\n[错误] 加载VOSK模型时发生异常: {e}")
        logger.info(f"异常详情：")
        traceback.print_exc()
        return None


# src/speech/model_loader.py
# 在 load_vosk_model 函数后添加以下函数

def register_model_to_global_state(global_state: dict, model_obj: 'vosk.Model'):
    """
    将加载好的VOSK模型对象注册到全局状态字典中。

    参数:
        global_state: 全局状态字典
        model_obj: 已加载的VOSK模型对象
    """
    if global_state and 'modules' in global_state:
        # 使用锁确保线程安全
        lock = global_state.get('lock')
        if lock:
            with lock:
                global_state['modules']['speech_recognizer'] = model_obj
        else:
            global_state['modules']['speech_recognizer'] = model_obj
        logger.info("[信息] 语音识别模型已注册到全局状态。")
def test_model_loading_standalone():
    """独立的模块测试函数"""
    logger.info("=" * 60)
    logger.info("VOSK 模型加载器 - 独立测试")
    logger.info("=" * 60)

    # 创建模拟配置（与您的 configs/default.yaml 保持一致）
    test_config = {
        "model_path": "./models/vosk",  # 主配置路径
        "language": "zh-cn",  # 语言
        "sample_rate": 16000
    }

    logger.info(f"测试配置: {test_config}")
    logger.info("-" * 40)

    model = load_vosk_model(test_config)

    if model:
        logger.info("\n" + "=" * 60)
        logger.info("✅ 模型加载测试通过！")
        logger.info("=" * 60)
        return True
    else:
        logger.info("\n" + "=" * 60)
        logger.info("❌ 模型加载测试失败。")
        logger.info("=" * 60)
        return False


# 当此文件被直接运行时，执行测试
if __name__ == "__main__":
    test_model_loading_standalone()