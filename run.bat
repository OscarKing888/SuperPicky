@echo off
REM 切换到脚本所在目录
cd /d "%~dp0"

REM 激活虚拟环境（如果存在）
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    REM 使用虚拟环境的Python
    .venv\Scripts\python.exe main.py
) else (
    REM 使用系统Python
    python main.py
)

REM 如果出错，暂停以便查看错误信息
if errorlevel 1 (
    echo.
    echo ========================================
    echo 程序运行出错，请查看上方错误信息
    echo.
    echo 如果是numpy导入错误，请运行: fix_numpy.bat
    echo 如果是rawpy导入错误，请运行: fix_rawpy.bat
    echo ========================================
    pause
)