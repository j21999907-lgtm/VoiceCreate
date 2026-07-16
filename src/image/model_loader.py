

import logging

logger = logging.getLogger("VoiceCreate")

# src/image/model_loader.py
"""
DreamLite 图片生成模型加载模块
修复版本：修正路径、OpenMP冲突和依赖问题
"""

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings

# 解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 抑制一些警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger.info(f"[DreamLite] 项目根目录: {PROJECT_ROOT}")

class ModelStatus(Enum):
    """模型状态枚举"""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


@dataclass
class GenerationResult:
    """生成结果数据类"""
    success: bool
    image: Optional[Any] = None
    generation_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DreamLiteModel:
    """
    DreamLite 模型包装类。
    修复版本：支持正确的模型路径和详细的错误处理
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.status = ModelStatus.NOT_LOADED

        # 获取并验证模型路径
        raw_path = config.get("model_path", "./models/DreamLite-base")

        # 转换为绝对路径
        if os.path.isabs(raw_path):
            self.model_path = raw_path
        else:
            # 尝试多种可能的路径格式
            possible_paths = [
                PROJECT_ROOT / raw_path,  # 相对于项目根目录
                Path.cwd() / raw_path,  # 相对于当前工作目录
                Path(raw_path)  # 原始路径
            ]

            for path in possible_paths:
                if path.exists():
                    self.model_path = str(path.resolve())
                    break
            else:
                # 如果都不存在，使用相对于项目根目录的路径
                self.model_path = str(PROJECT_ROOT / raw_path)

        # 设备配置
        self.device = config.get("device", "cpu")
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.info("[警告] CUDA不可用，将使用CPU")
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"

        # 其他参数
        self.default_size = config.get("default_size", 512)
        self.steps = config.get("steps", 20)
        self.model_type = config.get("model_type", "base")
        self.dtype_str = config.get("dtype", "float32")

        # 转换dtype
        try:
            import torch
            if self.dtype_str == "float16":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        except ImportError:
            self.dtype = None

        # 检查模型路径
        self._validate_model_path()

    def _validate_model_path(self):
        """验证模型路径"""
        logger.info(f"[DreamLite] 验证模型路径: {self.model_path}")
        if not os.path.exists(self.model_path):
            logger.info(f"[警告] 模型路径不存在: {self.model_path}")
            self.model_exists = False
            return

        # 检查是否是目录
        if not os.path.isdir(self.model_path):
            logger.info(f"[警告] 模型路径不是目录: {self.model_path}")
            self.model_exists = False
            return

        # 检查关键文件
        required_files = ["model_index.json"]
        optional_folders = ["unet", "vae", "text_encoder", "scheduler"]

        missing_files = []
        for file in required_files:
            file_path = os.path.join(self.model_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)

        if missing_files:
            logger.info(f"[警告] 缺少必需文件: {missing_files}")
            self.model_exists = False

            # 列出目录内容以便调试
            logger.info(f"[调试] 目录内容:")
            try:
                for item in os.listdir(self.model_path):
                    item_path = os.path.join(self.model_path, item)
                    if os.path.isdir(item_path):
                        logger.info(f"  📁 {item}/")
                    else:
                        logger.info(f"  📄 {item}")
            except Exception as e:
                logger.info(f"    无法列出目录: {e}")
        else:
            self.model_exists = True
            logger.info(f"[DreamLite] 模型路径验证通过")
    def load(self) -> bool:
        """加载模型（真实或模拟）"""
        logger.info(f"\n[DreamLite] 正在加载模型，路径: {self.model_path}")
        self.status = ModelStatus.LOADING

        if not self.model_exists:
            logger.info(f"[DreamLite] 警告: 模型文件不完整或不存在")
            logger.info(f"[DreamLite] 将使用模拟模式继续开发")
            self.status = ModelStatus.LOADED
            logger.info(f"[DreamLite] 模拟模式加载完成")
            return True

        try:
            # 尝试加载真实模型
            return self._load_real_model()
        except ImportError as e:
            logger.info(f"[DreamLite] 缺少必要的库: {e}")
            logger.info(f"请安装依赖: pip install torch diffusers transformers accelerate")
            return False
        except Exception as e:
            logger.info(f"[DreamLite] 真实模型加载失败: {e}")
            logger.info(f"[DreamLite] 将回退到模拟模式")
            self.status = ModelStatus.LOADED
            return True

    def _load_real_model(self) -> bool:
        """加载真实的 DreamLite 模型"""
        logger.info(f"[DreamLite] 正在加载真实模型，这可能需要一些时间...")
        # 延迟导入，以便在需要时才检查依赖
        try:
            from diffusers import DiffusionPipeline
            import torch
        except ImportError as e:
            logger.info(f"[错误] 缺少必要的库: {e}")
            return False

        try:
            # 检查是否已安装xformers
            xformers_available = False
            try:
                import xformers
                xformers_available = True
            except ImportError:
                pass

            logger.info(f"[DreamLite] PyTorch版本: {torch.__version__}")
            logger.info(f"[DreamLite] CUDA可用: {torch.cuda.is_available()}")
            logger.info(f"[DreamLite] 设备: {self.device}")
            logger.info(f"[DreamLite] 精度: {self.dtype_str}")
            # 根据设备选择dtype
            if self.device == "cuda" and torch.cuda.is_available():
                device_str = "cuda"
                torch_dtype = torch.float16 if self.dtype_str == "float16" else torch.float32
            else:
                device_str = "cpu"
                torch_dtype = torch.float32
                self.device = "cpu"  # 确保更新设备

            # 加载模型
            logger.info(f"[DreamLite] 从本地路径加载模型...")
            # 检查模型文件
            if not os.path.exists(os.path.join(self.model_path, "model_index.json")):
                raise FileNotFoundError(f"找不到 model_index.json 在 {self.model_path}")

            # 加载管道
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            )

            # 移动到设备
            self.pipe = self.pipe.to(device_str)

            # 启用优化
            if xformers_available and self.device == "cuda":
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info(f"[DreamLite] 已启用 xformers 优化")
                except:
                    logger.info(f"[DreamLite] 无法启用 xformers 优化")
            # 启用注意力切片以节省内存
            if self.device == "cuda" and hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
                logger.info(f"[DreamLite] 已启用注意力切片")
            self.status = ModelStatus.LOADED
            logger.info(f"[DreamLite] ✅ 真实模型加载成功！")
            logger.info(f"        设备: {self.device}, 尺寸: {self.default_size}, 类型: {self.model_type}")
            # 测试一次生成
            self._test_generation()

            return True

        except Exception as e:
            logger.info(f"[DreamLite] 模型加载异常: {e}")
            traceback.print_exc()
            return False

    def _test_generation(self):
        """测试生成功能"""
        logger.info(f"[DreamLite] 测试生成功能...")
        try:
            start_time = time.time()

            # 使用小尺寸快速测试
            test_width = 64
            test_height = 64
            test_steps = 2

            result = self._generate_real(
                prompt="测试图片",
                width=test_width,
                height=test_height,
                steps=test_steps,
                seed=42,
                start_time=start_time
            )

            if result.success:
                logger.info(f"[DreamLite] ✅ 生成测试通过 ({result.generation_time:.2f}秒)")
            else:
                logger.info(f"[DreamLite] ⚠️ 生成测试失败: {result.error_message}")
        except Exception as e:
            logger.info(f"[DreamLite] ⚠️ 生成测试异常: {e}")
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """
        生成图片

        参数:
            prompt: 生成图片的提示词
            **kwargs: 其他生成参数

        返回:
            GenerationResult: 生成结果
        """
        if self.status != ModelStatus.LOADED:
            return GenerationResult(
                success=False,
                error_message="模型未加载"
            )

        start_time = time.time()

        try:
            # 获取生成参数
            width = kwargs.get('width', self.default_size)
            height = kwargs.get('height', self.default_size)
            steps = kwargs.get('steps', self.steps)
            seed = kwargs.get('seed', None)

            # 限制最大尺寸以防止内存溢出
            max_size = 1024
            width = min(width, max_size)
            height = min(height, max_size)

            # 限制步数
            steps = min(steps, 50)

            logger.info(f"[DreamLite] 生成提示词: '{prompt[:50]}...'")
            logger.info(f"           尺寸: {width}x{height}, 步数: {steps}")
            if hasattr(self, 'pipe') and self.pipe is not None:
                # 使用真实模型生成
                return self._generate_real(prompt, width, height, steps, seed, start_time)
            else:
                # 使用模拟生成
                return self._generate_mock(prompt, width, height, steps, seed, start_time)

        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = f"生成时发生异常: {str(e)}"
            logger.info(f"[DreamLite] {error_msg}")
            traceback.print_exc()
            return GenerationResult(
                success=False,
                generation_time=generation_time,
                error_message=error_msg
            )

    def _generate_real(self, prompt: str, width: int, height: int, steps: int,
                       seed: int, start_time: float) -> GenerationResult:
        """使用真实模型生成图片"""
        import torch

        try:
            # 设置随机种子
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None

            # 生成图片
            with torch.no_grad():
                image = self.pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    generator=generator,
                    guidance_scale=self.config.get("guidance_scale", 7.5)
                ).images[0]

            generation_time = time.time() - start_time

            return GenerationResult(
                success=True,
                image=image,
                generation_time=generation_time,
                metadata={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "seed": seed,
                    "model": f"DreamLite-{self.model_type}",
                    "device": self.device,
                    "generation_time": generation_time
                }
            )

        except torch.cuda.OutOfMemoryError:
            generation_time = time.time() - start_time
            error_msg = "CUDA内存不足，尝试减小图片尺寸或步数"
            logger.info(f"[DreamLite] {error_msg}")
            return GenerationResult(
                success=False,
                generation_time=generation_time,
                error_message=error_msg
            )
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = f"生成失败: {str(e)}"
            logger.info(f"[DreamLite] {error_msg}")
            return GenerationResult(
                success=False,
                generation_time=generation_time,
                error_message=error_msg
            )

    def _generate_mock(self, prompt: str, width: int, height: int, steps: int,
                       seed: int, start_time: float) -> GenerationResult:
        """模拟生成图片（用于测试）"""
        # 模拟生成时间
        time_per_step = 0.1
        estimated_time = min(steps * time_per_step, 3.0)
        time.sleep(estimated_time)

        generation_time = time.time() - start_time

        # 尝试创建模拟图片
        mock_image = None
        try:
            from PIL import Image, ImageDraw, ImageFont
            import random

            # 创建图片
            img = Image.new('RGB', (width, height), color=(40, 40, 60))
            draw = ImageDraw.Draw(img)

            # 添加边框
            border_margin = width // 20
            draw.rectangle([border_margin, border_margin, width - border_margin, height - border_margin],
                           outline=(100, 150, 200), width=3)

            # 添加文字
            try:
                font = ImageFont.load_default()
                text = prompt[:30] + "..." if len(prompt) > 30 else prompt
                text_width = draw.textlength(text, font=font)
                text_position = ((width - text_width) // 2, height // 2 - 10)
                draw.text(text_position, text, fill=(200, 220, 255), font=font)
            except:
                pass

            # 添加水印
            watermark = "DreamLite 模拟生成"
            draw.text((10, height - 20), watermark, fill=(100, 100, 120))

            mock_image = img

        except ImportError:
            # 如果PIL不可用，返回None
            pass

        return GenerationResult(
            success=True,
            image=mock_image,
            generation_time=generation_time,
            metadata={
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
                "model": f"DreamLite-{self.model_type} (模拟模式)",
                "device": self.device,
                "note": "这是模拟生成，请确保模型文件完整以获得真实图片"
            }
        )


def initialize_dreamlite_model(model_config: Dict[str, Any]) -> Optional[DreamLiteModel]:
    """
    初始化 DreamLite 图片生成模型。

    参数:
        model_config: 包含模型配置的字典

    返回:
        初始化的 DreamLiteModel 实例，如果失败则返回 None。
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("DreamLite 图片生成模型初始化")
    logger.info(f"{'=' * 60}")

    try:
        # 创建模型实例
        model = DreamLiteModel(model_config)

        # 尝试加载模型
        if model.load():
            logger.info(f"[DreamLite] ✅ 模型初始化完成")
            logger.info(f"{'=' * 60}")
            return model
        else:
            logger.info(f"[DreamLite] ❌ 模型加载失败")
            logger.info(f"{'=' * 60}")
            return None

    except Exception as e:
        logger.info(f"[DreamLite] ❌ 初始化时发生异常: {e}")
        traceback.print_exc()
        logger.info(f"{'=' * 60}")
        return None


def register_model_to_global_state(global_state: dict, model_obj: DreamLiteModel):
    """将加载好的模型对象注册到全局状态字典中。"""
    if global_state and 'modules' in global_state:
        lock = global_state.get('lock')
        if lock:
            with lock:
                global_state['modules']['image_generator'] = model_obj
        else:
            global_state['modules']['image_generator'] = model_obj
        logger.info("[DreamLite] 图片生成模型已注册到全局状态。")

def test_model_loading_standalone():
    """独立的模块测试函数"""
    logger.info(f"{'=' * 60}")
    logger.info("DreamLite 模型加载模块测试")
    logger.info(f"{'=' * 60}")

    # 测试配置 - 使用正确的路径
    test_config = {
        "model_path": "./models/DreamLite-base",  # 相对于项目根目录
        "device": "cpu",  # 先用CPU测试
        "default_size": 256,
        "steps": 4,
        "model_type": "base",
        "dtype": "float32"
    }

    logger.info("1. 测试模型初始化...")
    model = initialize_dreamlite_model(test_config)

    if model is not None:
        logger.info("✅ 模型初始化测试通过")
        logger.info(f"   模型状态: {model.status.value}")
        logger.info(f"   模型存在: {model.model_exists}")
        logger.info(f"   设备: {model.device}")
        # 测试生成功能
        logger.info("\n2. 测试图片生成...")
        result = model.generate(
            prompt="一只可爱的猫，数字艺术风格",
            width=128,
            height=128,
            steps=2
        )

        if result.success:
            logger.info(f"✅ 图片生成测试通过")
            logger.info(f"   生成时间: {result.generation_time:.2f}秒")
            logger.info(f"   元数据: {result.metadata}")
            # 保存测试图片
            if result.image is not None:
                try:
                    result.image.save("test_output.png")
                    logger.info("   测试图片已保存为: test_output.png")
                except Exception as e:
                    logger.info(f"   无法保存图片: {e}")
        else:
            logger.info(f"❌ 图片生成测试失败: {result.error_message}")
    else:
        logger.info("❌ 模型初始化测试失败")
    logger.info(f"{'=' * 60}")
    logger.info("模型加载模块测试完成。")
    logger.info(f"{'=' * 60}")


# 当此文件被直接运行时，执行测试
if __name__ == "__main__":
    # 设置环境变量避免OpenMP冲突
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    test_model_loading_standalone()