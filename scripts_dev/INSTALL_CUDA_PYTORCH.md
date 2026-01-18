# PyTorch CUDA 版本安装指南

## 问题

当前安装的 PyTorch 是 CPU 版本（`2.9.1+cpu`），无法使用 GPU 加速。

## 解决方案

### 方法 1: 使用自动安装脚本（推荐）

#### Windows 批处理脚本
```bash
install_pytorch_cuda.bat
```

#### Python 脚本
```bash
python install_pytorch_cuda.py
```

脚本会自动：
1. 检测 NVIDIA 驱动和 CUDA 版本
2. 选择对应的 PyTorch CUDA 版本
3. 卸载旧版本并安装新版本
4. 验证安装

### 方法 2: 手动安装

#### 步骤 1: 检查 NVIDIA 驱动

运行以下命令检查驱动是否安装：
```bash
nvidia-smi
```

如果命令不存在，请先安装 NVIDIA 驱动程序：
- 访问：https://www.nvidia.com/drivers
- 下载并安装最新的驱动程序

#### 步骤 2: 卸载旧版本

```bash
pip uninstall torch torchvision torchaudio -y
```

#### 步骤 3: 安装 CUDA 版本

**CUDA 11.8 版本（推荐，兼容性最好）：**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1 版本（如果系统支持）：**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 步骤 4: 验证安装

```bash
python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('CUDA 版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU 数量:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
```

预期输出：
```
PyTorch 版本: 2.x.x+cu118
CUDA 可用: True
CUDA 版本: 11.8
GPU 数量: 1
```

## 常见问题

### Q1: 安装后仍然显示 CPU 版本

**原因：** 可能安装了错误的版本，或者虚拟环境未激活

**解决：**
1. 确保在正确的虚拟环境中安装
2. 检查安装的版本：`pip show torch`
3. 如果版本号包含 `+cpu`，说明安装的是 CPU 版本，需要重新安装

### Q2: CUDA 版本不匹配

**原因：** PyTorch 的 CUDA 版本与系统 CUDA 版本不匹配

**解决：**
- PyTorch CUDA 11.8 可以运行在 CUDA 11.8+ 的系统上
- PyTorch CUDA 12.1 可以运行在 CUDA 12.1+ 的系统上
- 如果系统 CUDA 版本较低，使用 cu118 版本（向后兼容）

### Q3: 安装失败

**可能原因：**
1. 网络问题
2. pip 版本过低
3. 权限问题

**解决：**
1. 升级 pip: `python -m pip install --upgrade pip`
2. 使用国内镜像（如果网络慢）:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
3. 使用管理员权限运行

### Q4: 找不到 nvidia-smi

**原因：** NVIDIA 驱动程序未安装

**解决：**
1. 访问 https://www.nvidia.com/drivers
2. 下载并安装最新的驱动程序
3. 重启电脑
4. 重新运行安装脚本

## 验证 CUDA 加速

安装完成后，重新运行程序，应该会看到：

```
🔍 设备选择过程 (首选: auto)
============================================================
✅ PyTorch 已导入，版本: 2.x.x+cu118
   CUDA (NVIDIA GPU): ✅ 可用
      CUDA 版本: 11.8
      GPU 数量: 1
      GPU 0: NVIDIA GeForce RTX XXX

✅ 选择设备: CUDA (NVIDIA GPU)
```

而不是：
```
⚠️  PyTorch 未编译 CUDA 支持，可能是 CPU 版本
⚠️  选择设备: CPU (所有 GPU 都不可用)
```

## 相关文件

- `install_pytorch_cuda.bat` - Windows 批处理安装脚本
- `install_pytorch_cuda.py` - Python 安装脚本
- `CUDA_SETUP.md` - CUDA 设置详细文档
