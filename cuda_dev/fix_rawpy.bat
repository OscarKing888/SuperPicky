@echo off
REM 修复rawpy安装问题（Python 3.14兼容性）
echo ========================================
echo 修复rawpy安装
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

echo [3/4] 卸载rawpy...
pip uninstall rawpy -y

echo [4/4] 重新安装rawpy（强制重新安装）...
REM 使用--force-reinstall确保重新编译/安装
REM 使用--no-cache-dir避免使用缓存的旧版本
REM 注意：如果预编译版本不兼容Python 3.14，可能需要从源码编译
echo    尝试安装预编译版本...
pip install --force-reinstall --no-cache-dir rawpy

REM 如果失败，尝试从源码编译（需要Visual C++ Build Tools）
if errorlevel 1 (
    echo.
    echo [WARN] 预编译版本安装失败，尝试从源码编译...
    echo    注意：这需要安装 Visual C++ Build Tools
    pip install --force-reinstall --no-cache-dir --no-binary :all: rawpy
)

echo.
echo ========================================
echo 测试rawpy导入...
echo ========================================
.venv\Scripts\python.exe -c "import rawpy; import rawpy._rawpy; print('[OK] rawpy导入成功！')"

if errorlevel 1 (
    echo.
    echo [ERROR] rawpy安装失败
    echo.
    echo Python 3.14 可能还不完全支持 rawpy
    echo.
    echo 建议解决方案:
    echo 1. 降级到 Python 3.12 或 3.13 (推荐)
    echo    重新创建虚拟环境: python3.12 -m venv .venv
    echo    然后运行: setup.bat
    echo.
    echo 2. 或者尝试安装特定版本的rawpy:
    echo    pip install rawpy==0.20.0
    echo.
    echo 3. 如果仍然失败，可能需要安装 Visual C++ Build Tools
    echo    用于编译rawpy的C扩展模块
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo [OK] rawpy安装成功！
    echo.
    pause
)
