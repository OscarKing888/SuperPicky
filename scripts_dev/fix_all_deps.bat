@echo off
REM 修复所有依赖问题（numpy, rawpy, PySide6等）
echo ========================================
echo 修复所有依赖包
echo ========================================
echo.

cd /d "%~dp0"

REM 检查虚拟环境
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] 虚拟环境不存在，请先运行 setup.bat
    pause
    exit /b 1
)

echo [1/5] 激活虚拟环境...
call .venv\Scripts\activate.bat

echo [2/5] 检查Python版本...
.venv\Scripts\python.exe --version
echo.
echo [WARN] Python 3.14 是较新版本，某些包可能不完全支持
echo        如果修复失败，建议降级到 Python 3.12 或 3.13
echo.

echo [3/5] 修复numpy...
call fix_numpy.bat
if errorlevel 1 (
    echo [ERROR] numpy修复失败
    pause
    exit /b 1
)

echo.
echo [4/5] 修复rawpy...
call fix_rawpy.bat
if errorlevel 1 (
    echo [WARN] rawpy修复失败，但程序可能仍可运行（仅RAW转换功能不可用）
)

echo.
echo [5/5] 修复PySide6...
call fix_pyside6.bat
if errorlevel 1 (
    echo [ERROR] PySide6修复失败，GUI无法运行
    pause
    exit /b 1
)

echo.
echo ========================================
echo 所有依赖修复完成！
echo ========================================
echo.
pause
