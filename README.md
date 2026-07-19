# VoiceCreate

VoiceCreate 是一个本地桌面应用：接收中文语音或键盘指令，生成图片，并按“左上角”“中心”等位置描述在屏幕上显示结果。

## 功能

- 使用 VOSK 进行离线语音识别
- 使用本地 Diffusers SD-Turbo 模型生成图片
- 解析中文动作、主体、风格和屏幕位置
- 可选使用本地 Ollama 模型增强提示词
- 保存生成图片及 JSON 元数据

## 环境要求

- Windows 10/11
- Python 3.10 或 3.11
- 建议使用支持 CUDA 的 NVIDIA GPU；CPU 模式可以运行，但生成速度较慢
- Tkinter（通常随 Windows Python 一起安装）
- 麦克风（仅语音输入需要）

## 安装

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`pyaudio` 在部分 Windows 环境中可能需要安装与 Python 版本匹配的预编译 wheel。只使用键盘输入时，仍建议安装完整依赖，避免初始化语音模块时报错。

## 模型目录

默认配置位于 `configs/default.yaml`：

```yaml
speech:
  model_path: ./models/vosk

image:
  model_path: ./models/sd-turbo
  device: cuda
```

模型路径相对于项目根目录解析，因此项目可以移动到其他目录。没有 CUDA 时将 `device` 改为 `cpu`，并建议将 `dtype` 改为 `float32`。

提示词增强默认使用本机 Ollama 的 `qwen2.5:7b`。该功能不可用时会自动保留原始提示词；也可以在配置中设置：

```yaml
ai_prompt_enhancer:
  enabled: false
```

## 启动

在项目根目录执行：

```powershell
.\venv\Scripts\python.exe -m src.main
```

也可以直接启动 GUI：

```powershell
.\venv\Scripts\python.exe src\gui\main_application.py
```

模型在后台加载。界面显示 `Ready` 后，可选择语音或键盘模式提交指令，例如：

```text
在左上角生成一只戴眼镜的猫，水彩风格
```

生成结果默认保存到 `generated_images/`，日志写入 `logs/`。

## 测试

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

## 项目结构

```text
configs/          运行配置
models/           本地语音与图片模型
src/gui/          Tkinter 桌面界面
src/speech/       录音、VAD 和语音识别
src/parser/       指令与位置解析
src/image/        图片模型加载与生成
src/display/      屏幕定位显示
tests/            自动化测试
```
