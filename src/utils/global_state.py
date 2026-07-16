

import logging

logger = logging.getLogger("VoiceCreate")

# src/utils/global_state.py
"""
全局状态管理模块
功能：创建并初始化一个集中管理程序运行时所有状态、数据和共享资源的字典。
此状态字典将作为单例在系统各模块间传递。
"""

import datetime
import threading
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, asdict, field
import copy

class SystemStatus(Enum):
    """系统运行状态枚举，定义所有可能的状态。"""
    BOOTING = "booting"          # 启动中
    IDLE = "idle"                # 空闲，等待语音指令
    LISTENING = "listening"      # 正在采集和处理音频
    PROCESSING_COMMAND = "processing_command"  # 正在解析指令
    GENERATING_IMAGE = "generating_image"      # 正在生成图片
    DISPLAYING = "displaying"    # 正在显示图片
    ERROR = "error"              # 发生错误
    SHUTTING_DOWN = "shutting_down"  # 关闭中

@dataclass
class Task:
    """任务数据类，定义单个图片生成任务的结构。"""
    task_id: str                     # 唯一任务标识
    status: str = "pending"          # 状态: pending, running, completed, failed
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat()) # 创建时间
    # 指令解析结果
    raw_text: Optional[str] = None   # 原始识别文本
    keywords: List[str] = field(default_factory=list)  # 提取出的关键词
    position_str: Optional[str] = None  # 位置描述字符串，如“左上角”
    position_xy: Optional[tuple] = None # 转换后的屏幕坐标 (x, y)
    # 生成结果
    image_params: Dict[str, Any] = field(default_factory=dict)  # 生成参数
    image_path: Optional[str] = None # 生成图片的保存路径
    # 显示相关
    window_handle: Any = None        # 显示窗口的句柄或对象
    metadata: Dict[str, Any] = field(default_factory=dict)  # 图片元数据

def initialize_global_state(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据提供的配置，初始化并返回全局状态字典。

    参数：
        config (dict): 系统配置字典，来自 load_system_config。

    返回：
        dict: 一个深度嵌套的字典，包含了系统运行所需的所有状态、计数器和共享资源占位符。
    """
    # 获取当前时间，作为系统启动标志
    current_time = datetime.datetime.now()

    # 构建并返回全局状态字典
    global_state = {
        # --- 1. 系统核心状态与信息 ---
        "system_info": {
            "name": config.get("system", {}).get("name", "VoiceCreate"),
            "version": config.get("system", {}).get("version", "0.1.0"),
            "start_time": current_time,
            "start_time_iso": current_time.isoformat(),
        },
        "runtime_status": {
            "system": SystemStatus.BOOTING.value,  # 初始状态为启动中
            "current_activity": "Initializing",
            "last_error": None,  # 记录最后一次错误信息
        },

        # --- 2. 配置与模块实例（从参数传入或等待初始化）---
        "config": copy.deepcopy(config),  # 深拷贝配置，避免意外修改
        "modules": {
            "speech_recognizer": None,  # 语音识别器实例 (VOSK)
            "image_generator": None,     # 图片生成器实例 (DreamLite)
            "display_manager": None,     # 显示管理器实例
            "audio_capture": None,       # 音频采集流
        },

        # --- 3. 核心数据与缓存 ---
        "resources": {
            "position_mapping_table": None,  # 位置映射表，将由 create_position_mapping_table 填充
            "active_windows": {},  # 活跃窗口映射 {task_id: window_handle}
        },

        # --- 4. 任务队列与管理 ---
        "task_management": {
            "next_task_id": 1,  # 用于生成唯一任务ID的计数器
            "queue": [],        # 等待处理的任务队列 (FIFO)
            "active_tasks": {}, # 正在处理的任务 {task_id: Task对象}
            "completed_tasks": [], # 已完成的任务历史 (Task对象列表)
            "failed_tasks": [],    # 失败的任务历史
        },

        # --- 5. 运行统计与监控 ---
        "statistics": {
            "commands_received": 0,
            "images_generated": 0,
            "errors_occurred": 0,
            "total_uptime_seconds": 0,
        },
        "performance": {
            "last_audio_time": None,
            "last_generation_time": None,
        },

        # --- 6. 线程同步与控制 ---
        "lock": threading.RLock(),  # 可重入锁，用于保护对全局状态的并发访问
        "control_flags": {
            "keep_running": True,  # 主循环控制标志，设为False以优雅退出
            "pause_processing": False,
        },
    }

    return global_state

# 辅助函数：用于安全地更新和访问全局状态
def update_system_status(global_state: Dict[str, Any], new_status: SystemStatus, activity: str = ""):
    """安全地更新系统运行时状态。"""
    with global_state["lock"]:
        global_state["runtime_status"]["system"] = new_status.value
        if activity:
            global_state["runtime_status"]["current_activity"] = activity

def get_system_status(global_state: Dict[str, Any]) -> Dict[str, Any]:
    """安全地获取系统状态快照。"""
    with global_state["lock"]:
        return copy.deepcopy({
            "system": global_state["runtime_status"]["system"],
            "activity": global_state["runtime_status"]["current_activity"],
            "last_error": global_state["runtime_status"]["last_error"]
        })

def register_new_task(global_state: Dict[str, Any], task_data: Dict[str, Any]) -> Task:
    """创建并注册一个新任务到等待队列。返回创建的Task对象。"""
    with global_state["lock"]:
        task_id = f"task_{global_state['task_management']['next_task_id']:06d}"
        global_state["task_management"]["next_task_id"] += 1

        new_task = Task(
            task_id=task_id,
            raw_text=task_data.get("raw_text"),
            keywords=task_data.get("keywords", []),
            position_str=task_data.get("position_str"),
        )
        global_state["task_management"]["queue"].append(new_task)
        global_state["statistics"]["commands_received"] += 1
        return copy.deepcopy(new_task)  # 返回副本

# 模块测试代码
if __name__ == "__main__":
    logger.info("测试 global_state.py 模块")
    logger.info("=" * 40)

    # 1. 模拟一个最小配置
    mock_config = {
        "system": {"name": "VoiceCreate", "version": "0.1.0", "debug": True},
        "storage": {"image_dir": "./test_output"},
    }

    # 2. 测试初始化
    logger.info("1. 测试全局状态初始化...")
    try:
        gs = initialize_global_state(mock_config)
        logger.info(f"   ✅ 成功。系统名称: {gs['system_info']['name']}")
        logger.info(f"      初始状态: {gs['runtime_status']['system']}")
    except Exception as e:
        logger.info(f"   ❌ 失败: {e}")
    # 3. 测试状态更新
    logger.info("\n2. 测试状态更新函数...")
    try:
        update_system_status(gs, SystemStatus.IDLE, "等待指令")
        current_status = get_system_status(gs)
        logger.info(f"   ✅ 成功。当前状态: {current_status}")
    except Exception as e:
        logger.info(f"   ❌ 失败: {e}")
    # 4. 测试任务注册
    logger.info("\n3. 测试任务注册函数...")
    try:
        test_task_data = {"raw_text": "在中心生成一只猫", "keywords": ["猫"], "position_str": "中心"}
        new_task = register_new_task(gs, test_task_data)
        logger.info(f"   ✅ 成功。创建任务ID: {new_task.task_id}")
        logger.info(f"      待处理队列长度: {len(gs['task_management']['queue'])}")
        logger.info(f"      收到指令计数: {gs['statistics']['commands_received']}")
    except Exception as e:
        logger.info(f"   ❌ 失败: {e}")
    logger.info("\n" + "=" * 40)
    logger.info("全局状态模块测试完成。")