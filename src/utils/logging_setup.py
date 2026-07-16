import logging

logger = logging.getLogger("VoiceCreate")

# src/utils/logging_setup.py
"""
日志系统初始化模块
功能：配置系统的日志记录器，支持控制台和文件输出
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"


def setup_logger(
        name: str = "VoiceCreate",
        log_level: str = "INFO",
        log_to_file: bool = True
) -> logging.Logger:
    """
    配置并返回一个日志记录器

    参数:
        name: 日志记录器的名称
        log_level: 日志级别，可选值: DEBUG, INFO, WARNING, ERROR, CRITICAL
        log_to_file: 是否将日志记录到文件

    返回:
        配置好的 logging.Logger 对象
    """
    # 确保 logs 文件夹存在
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 获取日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 创建或获取日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除现有的处理器，避免重复
    logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 创建控制台处理器（输出到终端）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 创建文件处理器（输出到文件）
    if log_to_file:
        # 生成带时间戳的日志文件名
        log_filename = f"voicecreate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = LOG_DIR / log_filename

        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的日志
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_default_logger() -> logging.Logger:
    """
    获取默认的日志记录器

    返回:
        默认配置的日志记录器
    """
    return setup_logger()


# 测试代码
if __name__ == "__main__":
    logger.info("测试 logging_setup.py 模块")
    # 测试1: 创建默认日志记录器
    try:
        logger = setup_logger()
        logger.info("日志系统测试: 这是一条INFO级别信息")
        logger.debug("这是一条DEBUG级别信息")
        logger.warning("这是一条WARNING级别信息")
        logger.error("这是一条ERROR级别信息")
        logger.info("✓ 日志记录器创建成功")
    except Exception as e:
        logger.info(f"✗ 日志记录器创建失败: {e}")
    # 测试2: 验证日志文件创建
    log_files = list(LOG_DIR.glob("*.log"))
    if log_files:
        logger.info(f"✓ 日志文件已创建: {log_files[-1].name}")
    else:
        logger.info("✗ 未找到日志文件")