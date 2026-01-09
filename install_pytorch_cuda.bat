@echo off
REM 自动安装 PyTorch CUDA 版本
chcp 65001 >nul
echo ============================================================
echo   PyTorch CUDA 版本自动安装脚本
echo ============================================================
echo.

REM 检查是否在虚拟环境中
if exist ".venv\Scripts\activate.bat" (
    echo [1/5] 激活虚拟环境...
    call .venv\Scripts\activate.bat
    echo ✅ 虚拟环境已激活
) else (
    echo ⚠️  未找到虚拟环境，将在系统 Python 中安装
    echo    建议先创建虚拟环境: python -m venv .venv
    pause
)

echo.
echo [2/5] 检测 NVIDIA GPU 和 CUDA 驱动...
echo.

REM 检查 nvidia-smi 是否可用
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未找到 nvidia-smi 命令
    echo.
    echo 💡 请先安装 NVIDIA 驱动程序:
    echo    1. 访问 https://www.nvidia.com/drivers
    echo    2. 下载并安装最新的驱动程序
    echo    3. 安装完成后重新运行此脚本
    echo.
    pause
    exit /b 1
)

echo ✅ 找到 NVIDIA 驱动程序
echo.

REM 获取 CUDA 版本
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits ^| head -n 1') do set DRIVER_VERSION=%%i
echo    驱动版本: %DRIVER_VERSION%

REM 检测 CUDA 版本（通过 nvidia-smi）
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits ^| head -n 1') do set CUDA_VERSION=%%i
echo    CUDA 版本: %CUDA_VERSION%

echo.
echo [3/5] 检测当前 PyTorch 版本...
python -c "import torch; print(f'当前版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️  PyTorch 未安装
)

echo.
echo [4/5] 选择 PyTorch CUDA 版本...
echo.

REM 根据检测到的 CUDA 版本选择 PyTorch 版本
REM 如果 CUDA 版本 >= 12.1，使用 cu121
REM 如果 CUDA 版本 >= 11.8，使用 cu118
REM 否则使用 cu118（兼容性最好）

set PYTORCH_CUDA=cu118
set PYTORCH_INDEX=https://download.pytorch.org/whl/cu118

REM 尝试解析 CUDA 版本号
echo    检测到的 CUDA 版本: %CUDA_VERSION%
echo    推荐使用 PyTorch CUDA 11.8 版本（兼容性最好）
echo.

echo [5/5] 卸载旧版本并安装 PyTorch CUDA 版本...
echo.

REM 卸载旧版本
echo    正在卸载旧版本...
pip uninstall torch torchvision torchaudio -y >nul 2>&1

REM 安装 CUDA 版本
echo    正在安装 PyTorch CUDA 11.8 版本...
echo    这可能需要几分钟，请耐心等待...
echo.

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if %errorlevel% neq 0 (
    echo.
    echo ❌ 安装失败
    echo.
    echo 💡 如果安装失败，可以尝试:
    echo    1. 手动安装: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo    2. 或使用 CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   验证安装...
echo ============================================================
echo.

python -c "import torch; print(f'✅ PyTorch 版本: {torch.__version__}'); print(f'✅ CUDA 可用: {torch.cuda.is_available()}'); print(f'✅ CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'✅ GPU 数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); [print(f'✅ GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

if %errorlevel% neq 0 (
    echo.
    echo ⚠️  验证失败，但安装可能已成功
    echo    请手动运行: python -c "import torch; print(torch.cuda.is_available())"
) else (
    echo.
    echo ============================================================
    echo   ✅ 安装完成！
    echo ============================================================
    echo.
    echo   现在可以重新运行程序，应该会使用 CUDA 加速了
    echo.
)

pause
