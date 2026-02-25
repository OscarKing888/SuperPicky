@echo off
chcp 65001 >nul
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
set "ROOT_DIR=%ROOT_DIR:~0,-1%"
cd /d "%ROOT_DIR%"

if "%~1"=="" goto :usage
if "%~2"=="" goto :usage

set "OLD_DIR=%~1"
set "NEW_DIR=%~2"
set "OUT_DIR=%~3"
set "ZIP_PATH=%~4"

if "%OUT_DIR%"=="" (
    set "OUT_DIR=patches\incremental_patch"
)

set "CMD=py -3 scripts_dev\incremental_patch_tool.py make --old \"%OLD_DIR%\" --new \"%NEW_DIR%\" --out \"%OUT_DIR%\" --zip"

if not "%ZIP_PATH%"=="" (
    set "CMD=%CMD% --zip-path \"%ZIP_PATH%\""
)

echo [INFO] Running incremental patch build...
echo [INFO] Old: %OLD_DIR%
echo [INFO] New: %NEW_DIR%
echo [INFO] Out: %OUT_DIR%
if not "%ZIP_PATH%"=="" echo [INFO] Zip: %ZIP_PATH%

call %CMD%
exit /b %ERRORLEVEL%

:usage
echo.
echo Usage:
echo   %~nx0 ^<old_dir^> ^<new_dir^> [out_dir] [zip_path]
echo.
echo Examples:
echo   %~nx0 "dist_cpu\SuperPicky" "dist_cuda\SuperPicky"
echo   %~nx0 "d:\_SuperPickyVersions\SuperPicky_4.1.0" "dist_cuda\SuperPicky" "patches\v4.1.1_gpu"
echo.
echo Apply patch:
echo   py -3 scripts_dev\incremental_patch_tool.py apply --patch "patches\v4.1.1_gpu" --target "C:\Apps\SuperPicky"
echo.
exit /b 1

