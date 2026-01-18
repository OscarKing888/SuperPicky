@echo off
REM 修复numpy安装问题
echo ========================================
echo 修复numpy安装
echo ========================================
echo.

cd /d "%~dp0"

REM 检查虚拟环境
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] 虚拟环境不存在，请先运行 setup.bat
    pause
    exit /b 1
)

echo [1/3] 激活虚拟环境...
call .venv\Scripts\activate.bat

echo [2/3] 卸载numpy...
pip uninstall numpy -y

echo [3/3] 重新安装numpy...
REM 使用--no-cache-dir避免缓存问题
REM 使用--upgrade确保安装最新版本
pip install --upgrade --no-cache-dir numpy

echo.
echo ========================================
echo 测试numpy导入...
echo ========================================
.venv\Scripts\python.exe -c "import numpy; print('[OK] Numpy版本:', numpy.__version__); print('[OK] Numpy位置:', numpy.__file__)"

if errorlevel 1 (
    echo.
    echo [ERROR] Numpy安装失败
    echo.
    echo 可能的解决方案:
    echo 1. 检查Python版本兼容性 (当前: Python 3.14)
    echo 2. 尝试安装特定版本的numpy: pip install numpy==1.26.4
    echo 3. 检查是否有编译工具 (Visual C++ Build Tools)
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo [OK] Numpy安装成功！
    echo.
    pause
)
