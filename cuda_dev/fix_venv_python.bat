@echo off
REM 修复虚拟环境中的 Python 路径配置
REM 用于解决 "No Python at 'xxx'" 错误

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   修复虚拟环境 Python 路径配置
echo ============================================================
echo.

REM 获取脚本所在目录
cd /d "%~dp0"

REM 检查虚拟环境目录是否存在
if not exist ".venv" (
    echo [错误] 未找到虚拟环境目录 .venv
    echo.
    pause
    exit /b 1
)

echo [信息] 正在检测当前可用的 Python 路径...

REM 查找当前可用的 Python
set "PYTHON_CMD="
set "PYTHON_PATH="
set "PYTHON_VERSION="

REM 尝试不同的 Python 命令
for %%P in (python3.12 python3.13 python3.11 python3.10 python3 python py) do (
    where %%P >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=*" %%V in ('%%P --version 2^>^&1') do set "VERSION_LINE=%%V"
        for /f "tokens=2" %%V in ("!VERSION_LINE!") do set "VERSION=%%V"
        
        REM 获取完整路径
        for /f "tokens=*" %%F in ('where %%P') do set "PYTHON_PATH=%%F"
        
        if defined PYTHON_PATH (
            set "PYTHON_CMD=%%P"
            set "PYTHON_VERSION=!VERSION!"
            goto :found_python
        )
    )
)

:found_python
if not defined PYTHON_PATH (
    echo [错误] 未找到可用的 Python 解释器
    echo.
    pause
    exit /b 1
)

echo [成功] 找到 Python: !PYTHON_VERSION!
echo [成功] Python 路径: !PYTHON_PATH!
echo.

REM 获取 Python 的目录（去掉 python.exe）
for %%F in ("!PYTHON_PATH!") do set "PYTHON_DIR=%%~dpF"
set "PYTHON_DIR=!PYTHON_DIR:~0,-1!"

echo [信息] Python 目录: !PYTHON_DIR!
echo.

REM 检查 pyvenv.cfg 文件
set "PYVENV_CFG=.venv\pyvenv.cfg"
if not exist "!PYVENV_CFG!" (
    echo [错误] 未找到配置文件: !PYVENV_CFG!
    echo.
    pause
    exit /b 1
)

echo [信息] 正在更新配置文件: !PYVENV_CFG!
echo.

REM 备份原文件
set "BACKUP_FILE=!PYVENV_CFG!.backup"
copy "!PYVENV_CFG!" "!BACKUP_FILE!" >nul 2>&1
if !errorlevel! equ 0 (
    echo [信息] 已备份原文件到: !BACKUP_FILE!
)

REM 获取 Python 版本号
for /f "tokens=2" %%V in ('!PYTHON_CMD! --version 2^>^&1') do set "PYTHON_VERSION_FULL=%%V"

REM 获取当前目录（用于 command 行）
set "CURRENT_DIR=%~dp0"
set "CURRENT_DIR=!CURRENT_DIR:~0,-1!"

REM 读取原文件并更新
set "TEMP_FILE=%TEMP%\pyvenv_cfg_%RANDOM%.tmp"
(
    for /f "usebackq tokens=*" %%L in ("!PYVENV_CFG!") do (
        set "LINE=%%L"
        set "LINE=!LINE: =!"
        
        REM 检查是否是 home 行
        if "!LINE:~0,5!"=="home=" (
            echo home = !PYTHON_DIR!
        ) else if "!LINE:~0,11!"=="executable=" (
            echo executable = !PYTHON_PATH!
        ) else if "!LINE:~0,8!"=="command=" (
            echo command = !PYTHON_PATH! -m venv !CURRENT_DIR!\.venv
        ) else if "!LINE:~0,8!"=="version=" (
            echo version = !PYTHON_VERSION_FULL!
        ) else (
            REM 保持其他行不变（包括空行和 include-system-site-packages）
            echo %%L
        )
    )
) > "!TEMP_FILE!"

REM 替换原文件
move /y "!TEMP_FILE!" "!PYVENV_CFG!" >nul

if !errorlevel! equ 0 (
    echo [成功] 配置文件已更新
) else (
    echo [错误] 更新配置文件失败
    pause
    exit /b 1
)

echo.
echo [信息] 更新后的配置内容:
echo ----------------------------------------
type "!PYVENV_CFG!"
echo ----------------------------------------
echo.

REM 检查并更新 python.exe（如果需要）
set "VENV_PYTHON=.venv\Scripts\python.exe"
if exist "!VENV_PYTHON!" (
    echo [信息] 检查虚拟环境中的 python.exe...
    
    REM 尝试运行测试
    "!VENV_PYTHON!" --version >nul 2>&1
    if !errorlevel! equ 0 (
        echo [成功] 虚拟环境中的 Python 可以正常工作
    ) else (
        echo [警告] 虚拟环境中的 python.exe 可能有问题
        echo [信息] 尝试重新创建 python.exe...
        
        REM 删除旧的 python.exe
        del /f /q "!VENV_PYTHON!" >nul 2>&1
        
        REM 创建新的 python.exe（使用 mklink 或 copy）
        REM 注意：在某些系统上可能需要管理员权限
        copy "!PYTHON_PATH!" "!VENV_PYTHON!" >nul 2>&1
        if !errorlevel! equ 0 (
            echo [成功] 已重新创建 python.exe
        ) else (
            echo [警告] 无法自动重新创建 python.exe
            echo [信息] 请手动复制: !PYTHON_PATH! 到 !VENV_PYTHON!
        )
    )
)

echo.
echo ============================================================
echo   ✅ 修复完成！
echo ============================================================
echo.
echo [信息] 现在可以尝试运行: python install_pytorch_cuda.py
echo.

pause

