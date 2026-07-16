

import logging

logger = logging.getLogger("VoiceCreate")

# src/utils/config_loader.py
"""
配置文件加载器
功能：
1. 加载 YAML 格式的系统配置文件
2. 加载 .env 环境变量文件
3. 提供项目根目录的绝对路径
"""

import os
import sys
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

# 获取项目根目录的绝对路径
# __file__ 是当前文件路径, .parent 是上一级 (src/utils), .parent.parent 是上上一级 (src), .parent 是根目录 (VoiceCreate)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_system_config(config_file: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    加载系统 YAML 配置文件

    参数:
        config_file: 配置文件的相对路径，默认为 "configs/default.yaml"

    返回:
        包含所有配置的字典

    抛出异常:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: 配置文件格式错误
    """
    full_path = os.path.join(BASE_DIR, config_file)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"配置文件不存在: {full_path}")

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        logger.info(f"[INFO] 配置文件加载成功: {full_path}")
        return config

    except yaml.YAMLError as e:
        logger.info(f"[ERROR] 配置文件格式错误: {e}")
        raise
    except Exception as e:
        logger.info(f"[ERROR] 加载配置文件时发生未知错误: {e}")
        raise


def load_config(config_file: str = "configs/default.yaml") -> Dict[str, Any]:
    """Compatibility alias used by newer workflow modules and examples."""
    return load_system_config(config_file)


def load_environment_variables() -> None:
    """
    加载 .env 环境变量文件 (如果存在)

    说明:
        1. 检查项目根目录下的 .env 文件
        2. 如果存在则加载，不存在则跳过
        3. 加载的环境变量可以通过 os.getenv() 访问
    """
    env_path = os.path.join(BASE_DIR, '.env')

    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        logger.info("[INFO] 已加载 .env 环境变量文件")
    else:
        logger.info("[INFO] 未找到 .env 文件，跳过环境变量加载")

# 测试代码
if __name__ == "__main__":
    logger.info("测试 config_loader.py")
    logger.info(f"项目根目录: {BASE_DIR}")
    # 测试环境变量加载
    try:
        load_environment_variables()
        logger.info("环境变量加载测试: 通过 ✓")
    except Exception as e:
        logger.info(f"环境变量加载测试: 失败 ✗ ({e})")
    # 测试配置文件加载
    try:
        config = load_system_config()
        logger.info(f"配置加载测试: 通过 ✓")
        logger.info(f"系统名称: {config.get('system', {}).get('name', '未找到')}")
    except Exception as e:
        logger.info(f"配置加载测试: 失败 ✗ ({e})")