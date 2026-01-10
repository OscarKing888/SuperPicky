@echo off
REM SuperPicky 环境设置脚本 (Windows)
REM 用于创建 Python 虚拟环境并安装依赖

setlocal enabledelayedexpansion

REM 设置错误处理
set "ERRORLEVEL=0"

REM 获取脚本所在目录
cd /d "%~dp0"

echo.
echo ============================================================
echo   SuperPicky 环境设置脚本 (Windows)
echo ============================================================
echo.

REM 检查 Python 版本
echo [信息] 检查 Python 版本...
set "PYTHON_CMD="
set "PYTHON_VERSION="

REM 尝试不同的 Python 命令
for %%P in (python3.12 python3.13 python3.11 python3 python py) do (
    where %%P >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2" %%V in ('%%P --version 2^>^&1') do set "VERSION=%%V"
        for /f "tokens=1,2 delims=." %%A in ("!VERSION!") do (
            set "MAJOR=%%A"
            set "MINOR=%%B"
        )
        
        REM 检查版本是否 >= 3.12
        if !MAJOR! equ 3 (
            if !MINOR! geq 12 (
                set "PYTHON_CMD=%%P"
                set "PYTHON_VERSION=!VERSION!"
                goto :found_python
            )
        )
    )
)

:found_python
if "%PYTHON_CMD%"=="" (
    echo [错误] 未找到 Python 3.12 或更高版本
    echo.
    echo [信息] 请安装 Python 3.12+：
    echo   - 从官网下载: https://www.python.org/downloads/
    echo   - 或使用 Microsoft Store 安装 Python
    echo.
    pause
    exit /b 1
)

echo [成功] 找到 Python: %PYTHON_VERSION% (%PYTHON_CMD%)
echo.

REM 检查虚拟环境目录
set "VENV_DIR=.venv"
if exist "%VENV_DIR%" (
    echo [警告] 虚拟环境目录已存在: %VENV_DIR%
    set /p "REPLY=是否删除并重新创建? [y/N]: "
    if /i "!REPLY!"=="y" (
        echo [信息] 删除现有虚拟环境...
        rmdir /s /q "%VENV_DIR%"
        echo [成功] 已删除旧虚拟环境
    ) else (
        echo [信息] 使用现有虚拟环境
    )
    echo.
)

REM 创建虚拟环境
if not exist "%VENV_DIR%" (
    echo [信息] 创建 Python 虚拟环境...
    "%PYTHON_CMD%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo [成功] 虚拟环境创建成功: %VENV_DIR%
    echo.
)

REM 激活虚拟环境
echo [信息] 激活虚拟环境...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [错误] 虚拟环境激活失败
    pause
    exit /b 1
)

REM 升级 pip
echo [信息] 升级 pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [警告] pip 升级失败，继续安装依赖...
) else (
    echo [成功] pip 已升级
)
echo.

REM 检查 requirements.txt
if not exist "requirements.txt" (
    echo [错误] 未找到 requirements.txt 文件
    pause
    exit /b 1
)

REM 安装依赖
echo [信息] 安装依赖包（这可能需要几分钟）...
echo ============================================================
echo.

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [成功] 所有依赖安装完成！
echo ============================================================
echo.

REM 检查 CUDA 支持（可选）
echo [信息] 检查 CUDA 支持...
python -c "import torch; cuda_available = torch.cuda.is_available(); print('[成功] CUDA 可用' if cuda_available else '[信息] CUDA 不可用，将使用 CPU'); print('GPU 名称:', torch.cuda.get_device_name(0) if cuda_available else 'N/A')" 2>nul
if errorlevel 1 (
    echo [信息] PyTorch 未安装或 CUDA 检查失败
    echo [提示] 如需 CUDA 加速，请安装支持 CUDA 的 PyTorch：
    echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)
echo.

echo ============================================================
echo [成功] 环境设置完成！
echo ============================================================
echo.
echo [信息] 使用方法：
echo   1. 激活虚拟环境:
echo      .venv\Scripts\activate.bat
echo.
echo   2. 运行 CLI 工具:
echo      python superpicky_cli.py process C:\path\to\photos
echo.
echo   3. 退出虚拟环境:
echo      deactivate
echo.
echo   4. 或者直接运行（无需激活）:
echo      .venv\Scripts\python.exe superpicky_cli.py process C:\path\to\photos
echo.

pause
