# CUDA 加速设置指南

## 概述

SuperPicky 现在支持自动 GPU 加速，可以在以下平台上使用：

- **macOS**: Apple Silicon (MPS - Metal Performance Shaders)
- **Windows/Linux**: NVIDIA GPU (CUDA)
- **所有平台**: CPU 降级（当 GPU 不可用时）

## Windows 上使用 CUDA 加速

### 前置要求

1. **NVIDIA GPU**
   - 支持 CUDA 的 NVIDIA 显卡（推荐 GTX 10 系列或更高）
   - 检查您的 GPU 是否支持：https://developer.nvidia.com/cuda-gpus

2. **NVIDIA 驱动程序**
   - 安装最新的 NVIDIA 驱动程序
   - 下载地址：https://www.nvidia.com/drivers

3. **CUDA Toolkit**（可选，但推荐）
   - PyTorch 的预编译版本通常包含 CUDA 运行时
   - 如果需要编译自定义扩展，可能需要完整 CUDA Toolkit
   - 下载地址：https://developer.nvidia.com/cuda-downloads

### 安装步骤

#### 1. 安装 PyTorch（CUDA 版本）

SuperPicky 使用 PyTorch 进行 AI 推理。需要安装支持 CUDA 的 PyTorch 版本。

**方法一：使用 pip（推荐）**

```bash
# 对于 CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 对于 CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**方法二：检查当前 PyTorch 版本**

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果 `torch.cuda.is_available()` 返回 `True`，说明 CUDA 已正确配置。

#### 2. 验证 CUDA 支持

运行以下命令检查 CUDA 是否可用：

```bash
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('CUDA 版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU 名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

预期输出示例：
```
CUDA 可用: True
CUDA 版本: 11.8
GPU 名称: NVIDIA GeForce RTX 3080
```

#### 3. 安装其他依赖

```bash
pip install -r requirements.txt
```

### 自动设备选择

SuperPicky 会自动选择最佳计算设备，优先级如下：

1. **MPS** (macOS Apple Silicon)
2. **CUDA** (Windows/Linux NVIDIA GPU)
3. **CPU** (所有平台，作为降级选项)

### 运行时行为

当程序启动时，您会看到类似以下的消息：

**Windows (CUDA 可用)**:
```
✅ 检测到 NVIDIA GeForce RTX 3080 (CUDA)，启用硬件加速
```

**Windows (CUDA 不可用)**:
```
⚠️  CUDA 不可用，使用 CPU 推理
```

### 性能对比

使用 CUDA 加速可以显著提升处理速度：

| 设备类型 | 单张图片处理时间（估算） |
|---------|----------------------|
| CPU (Intel i7) | ~2-3 秒 |
| CUDA (RTX 3080) | ~0.3-0.5 秒 |
| MPS (M1 Pro) | ~0.5-0.8 秒 |

*实际性能取决于图片大小、模型复杂度和硬件配置*

### 故障排除

#### 问题 1: CUDA 不可用

**症状**: 程序显示 "CUDA 不可用，使用 CPU 推理"

**解决方案**:
1. 检查 NVIDIA 驱动程序是否已安装并更新到最新版本
2. 确认 PyTorch 安装的是 CUDA 版本（不是 CPU 版本）
3. 运行 `nvidia-smi` 命令检查 GPU 是否被系统识别

#### 问题 2: CUDA 版本不匹配

**症状**: 运行时出现 CUDA 版本错误

**解决方案**:
1. 检查 PyTorch 的 CUDA 版本：`python -c "import torch; print(torch.version.cuda)"`
2. 检查系统 CUDA 版本：`nvcc --version`（如果安装了 CUDA Toolkit）
3. 确保 PyTorch 的 CUDA 版本与系统兼容

#### 问题 3: 内存不足

**症状**: CUDA out of memory 错误

**解决方案**:
1. 减少批处理大小（如果支持）
2. 降低图片分辨率（在配置中调整 `TARGET_IMAGE_SIZE`）
3. 关闭其他占用 GPU 的程序

### 手动指定设备

如果需要手动指定设备，可以修改代码中的设备选择逻辑：

```python
from utils import get_best_device

# 强制使用 CUDA
device = get_best_device('cuda')

# 强制使用 CPU
device = get_best_device('cpu')

# 自动选择（推荐）
device = get_best_device('auto')
```

### 技术细节

- **YOLO 模型**: 使用 Ultralytics YOLO，支持 CUDA 加速
- **TOPIQ 模型**: 美学评分模型，支持 CUDA 加速
- **飞版检测模型**: EfficientNet 模型，支持 CUDA 加速
- **关键点检测**: ResNet50 模型，支持 CUDA 加速

所有模型都会自动使用检测到的最佳设备进行推理。

### 相关文件

- `utils.py`: 包含 `get_best_device()` 函数
- `ai_model.py`: YOLO 模型加载和推理
- `iqa_scorer.py`: TOPIQ 美学评分器
- `core/flight_detector.py`: 飞版检测器
- `core/keypoint_detector.py`: 关键点检测器

