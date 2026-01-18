cd ..

@echo off
REM 重建虚拟环境 - 解决Python版本升级后的依赖问题
echo ========================================
echo 重建虚拟环境
echo ========================================
echo.
echo [WARN] 这将删除现有的虚拟环境并重新创建
echo        所有已安装的包都需要重新安装
echo.
set /p confirm="确定继续? (y/N): "
if /i not "%confirm%"=="y" (
    echo 已取消
    exit /b 0
)

cd /d "%~dp0"

echo.
echo [1/6] 检查Python版本...
python --version
if errorlevel 1 (
    echo [ERROR] Python未安装或不在PATH中
    pause
    exit /b 1
)

echo.
echo [2/6] 删除旧虚拟环境...
if exist ".venv" (
    echo    正在删除 .venv 目录...
    rmdir /s /q .venv
    if exist ".venv" (
        echo [ERROR] 无法删除 .venv 目录，请手动删除后重试
        pause
        exit /b 1
    )
    echo    [OK] 旧虚拟环境已删除
) else (
    echo    [INFO] 未找到旧虚拟环境
)

echo.
echo [3/6] 创建新虚拟环境...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] 虚拟环境创建失败
    pause
    exit /b 1
)
echo    [OK] 虚拟环境创建成功

echo.
echo [4/6] 激活虚拟环境...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] 虚拟环境激活失败
    pause
    exit /b 1
)

echo.
echo [5/6] 升级pip和基础工具...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [WARN] pip升级失败，继续尝试安装依赖...
)

echo.
echo [6/6] 安装项目依赖...
echo    这可能需要几分钟时间...

REM 检查Python版本，选择对应的requirements文件
python -c "import sys; ver = sys.version_info; exit(0 if ver.major == 3 and ver.minor >= 14 else 1)" 2>nul
if errorlevel 1 (
    REM Python < 3.14，使用标准requirements.txt
    echo    使用标准依赖列表（支持rawpy）...
    pip install -r requirements.txt
) else (
    REM Python >= 3.14，使用兼容版本（rawpy需要特殊处理）
    echo    检测到Python 3.14+，使用兼容依赖列表...
    REM 先安装不包含rawpy的依赖
    pip install ultralytics>=8.0.0 opencv-python>=4.0.0 imageio>=2.0.0 numpy>=1.20.0 Pillow>=9.0.0 pillow-heif>=0.13.0
    pip install torch>=2.0.0 torchvision>=0.15.0 timm>=0.9.0
    pip install PySide6>=6.6.0
    
    REM 尝试安装rawpy（从源码编译）
    echo.
    echo    尝试安装rawpy（从源码编译，需要Visual C++ Build Tools）...
    pip install --no-binary :all: rawpy 2>nul
    if errorlevel 1 (
        echo     [WARN] rawpy安装失败，RAW转换功能将不可用
        echo     [INFO] 程序仍可运行，但无法转换RAW文件
        echo     [INFO] 建议：降级到Python 3.12或使用已有JPG文件
    ) else (
        echo     [OK] rawpy从源码编译成功
    )
)

if errorlevel 1 (
    echo [WARN] 部分依赖安装失败，但继续验证...
)

echo.
echo ========================================
echo 验证安装...
echo ========================================
echo.

echo [测试] numpy...
python -c "import numpy; print('  [OK] numpy', numpy.__version__)"
if errorlevel 1 (
    echo "  [ERROR] numpy导入失败"
)

echo [测试] rawpy...
python -c "import rawpy; import rawpy._rawpy; print('  [OK] rawpy')" 2>nul
if errorlevel 1 (
    echo "  [WARN] rawpy导入失败（可能需要Visual C++ Build Tools）"
)

echo [测试] PySide6...
python -c "from PySide6.QtWidgets import QApplication; print('  [OK] PySide6')"
if errorlevel 1 (
    echo "  [ERROR] PySide6导入失败"
)

echo [测试] PIL/Pillow...
python -c "from PIL import Image; print('  [OK] Pillow', Image.__version__)" 2>nul
if errorlevel 1 (
    echo "  [ERROR] Pillow导入失败"
    echo "  尝试修复: pip uninstall Pillow -y && pip install --no-cache-dir Pillow"
)

echo [测试] torch...
python -c "import torch; print('  [OK] torch', torch.__version__)" 2>nul
if errorlevel 1 (
    echo "  [WARN] torch导入失败（可选，用于AI模型）"
)

echo.
echo ========================================
echo 虚拟环境重建完成！
echo ========================================
echo.
echo 现在可以运行: run.bat
echo.
pause
