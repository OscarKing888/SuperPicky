@echo off
REM 修复PySide6安装问题（Python 3.14兼容性）
echo ========================================
echo 修复PySide6安装
echo ========================================
echo.

cd /d "%~dp0"

REM 检查虚拟环境
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] 虚拟环境不存在，请先运行 setup.bat
    pause
    exit /b 1
)

echo [1/4] 激活虚拟环境...
call .venv\Scripts\activate.bat

echo [2/4] 检查Python版本...
.venv\Scripts\python.exe --version
echo.

echo [3/4] 卸载PySide6...
pip uninstall PySide6 shiboken6 PySide6-Essentials PySide6-Addons -y

echo [4/4] 重新安装PySide6（强制重新安装）...
REM 使用--force-reinstall确保重新编译/安装
REM 使用--no-cache-dir避免使用缓存的旧版本
pip install --force-reinstall --no-cache-dir PySide6

echo.
echo ========================================
echo 测试PySide6导入...
echo ========================================
.venv\Scripts\python.exe -c "from PySide6.QtWidgets import QApplication; print('[OK] PySide6导入成功！')"

if errorlevel 1 (
    echo.
    echo [ERROR] PySide6安装失败
    echo.
    echo Python 3.14 可能还不完全支持 PySide6
    echo.
    echo 建议解决方案:
    echo 1. 降级到 Python 3.12 或 3.13 (推荐)
    echo    重新创建虚拟环境: python3.12 -m venv .venv
    echo.
    echo 2. 或者等待 PySide6 更新支持 Python 3.14
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo [OK] PySide6安装成功！
    echo.
    pause
)
