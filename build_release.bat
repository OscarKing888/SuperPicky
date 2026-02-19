@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

set "APP_NAME=SuperPicky"
set "SPEC_FILE=SuperPicky_win64.spec"
set "ROOT_DIR=%~dp0"
set "ROOT_DIR=%ROOT_DIR:~0,-1%"
cd /d "%ROOT_DIR%"

set "VERSION_ARG="
set "OUT_DIST_DIR=dist"
set "BUILD_ZIP=1"

call :parse_args %*
if errorlevel 1 exit /b 1
if defined SHOW_HELP goto :show_help

goto :start

:show_help
echo SuperPicky Windows build script
echo.
echo Usage:
echo   %~nx0 [version]
echo   %~nx0 [version] --dist-dir DIR
echo   %~nx0 [version] [--no-zip]
echo.
echo Options:
echo   [version]       Override version ^(default: from ui/about_dialog.py^)
echo   --dist-dir DIR  Output directory ^(default: dist^)
echo   --zip           Force create ZIP ^(default^)
echo   --no-zip        Skip ZIP creation
echo.
echo Environment:
echo   PYTHON_EXE       Python to use ^(default: current python in PATH, e.g. venv^)
echo.
echo Output ^(default: always build EXE + one ZIP^):
echo   %OUT_DIST_DIR%\%APP_NAME%\SuperPicky.exe
echo   %OUT_DIST_DIR%\%APP_NAME%_vVERSION_Win64.zip
echo.
exit /b 0

:parse_args
:parse_args_loop
if "%~1"=="" exit /b 0

if /i "%~1"=="--help" (
    set "SHOW_HELP=1"
    exit /b 0
)
if /i "%~1"=="-h" (
    set "SHOW_HELP=1"
    exit /b 0
)
if /i "%~1"=="--dist-dir" (
    if "%~2"=="" (
        echo [ERROR] --dist-dir requires a directory path
        exit /b 1
    )
    set "OUT_DIST_DIR=%~2"
    shift
    shift
    goto :parse_args_loop
)
if /i "%~1"=="--zip" (
    set "BUILD_ZIP=1"
    shift
    goto :parse_args_loop
)
if /i "%~1"=="--no-zip" (
    set "BUILD_ZIP=0"
    shift
    goto :parse_args_loop
)

if "%VERSION_ARG%"=="" (
    set "VERSION_ARG=%~1"
) else (
    echo [WARNING] Ignored extra argument: %~1
)
shift
goto :parse_args_loop

:start
echo.
echo [========================================]
echo Step 0: Environment check
echo [========================================]

if not exist "%SPEC_FILE%" (
    echo [ERROR] Missing spec file: %SPEC_FILE%
    exit /b 1
)

echo [SUCCESS] Spec file found: %SPEC_FILE%

if "!PYTHON_EXE!"=="" set "PYTHON_EXE=python"
rem Prefer current env Python (e.g. activated venv): use first "python" in PATH
if "!PYTHON_EXE!"=="python" (
    where python >nul 2>nul && for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%i"
)
if "!PYTHON_EXE!"=="" set "PYTHON_EXE=python"
call :check_python "!PYTHON_EXE!" "default"
if errorlevel 1 exit /b 1

echo.
echo [========================================]
echo Step 1: Resolve version
echo [========================================]

set "VERSION=4.0.5_sp3"
if not "!VERSION_ARG!"=="" (
    set "VERSION=!VERSION_ARG!"
    echo [SUCCESS] Use version from args: !VERSION!
) else (
    for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$c=Get-Content -Path 'ui/about_dialog.py' -Raw -Encoding UTF8; if($c -match 'v([0-9A-Za-z._-]+)'){ $matches[1] }"`) do (
        set "VERSION=%%i"
    )
    if "!VERSION!"=="" set "VERSION=0.0.0"
    echo [SUCCESS] Detected version: v!VERSION!
)

echo.
echo [========================================]
echo Step 1.5: Inject build metadata
echo [========================================]

set "COMMIT_HASH=unknown"
for /f "tokens=*" %%i in ('git rev-parse --short HEAD 2^>nul') do set "COMMIT_HASH=%%i"
echo [INFO] Commit hash: %COMMIT_HASH%

set "BUILD_INFO_FILE=core\build_info.py"
set "BUILD_INFO_BACKUP=core\build_info.py.backup"
if exist "%BUILD_INFO_FILE%" copy /y "%BUILD_INFO_FILE%" "%BUILD_INFO_BACKUP%" >nul

powershell -NoProfile -Command "(Get-Content -Path '%BUILD_INFO_FILE%' -Raw -Encoding UTF8) -replace 'COMMIT_HASH\s*=\s*.*', 'COMMIT_HASH = \"%COMMIT_HASH%\"' | Set-Content -Path '%BUILD_INFO_FILE%' -Encoding UTF8"
if errorlevel 1 (
    echo [ERROR] Failed to inject build info
    call :restore_build_info >nul
    exit /b 1
)

echo [SUCCESS] Build info injected

call :build_single
set "RET=%ERRORLEVEL%"
call :restore_build_info >nul
exit /b %RET%

:check_python
set "CHECK_PY=%~1"
set "CHECK_LABEL=%~2"

echo [INFO] Checking Python (%CHECK_LABEL%): %CHECK_PY%
"%CHECK_PY%" -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not available: %CHECK_PY%
    exit /b 1
)
for /f "tokens=*" %%i in ('"%CHECK_PY%" -c "import sys; print(sys.executable)" 2^>nul') do set "_PY_RESOLVED=%%i"
echo [SUCCESS] Python (%CHECK_LABEL%): !_PY_RESOLVED!

echo [INFO] Checking PyInstaller (%CHECK_LABEL%)...
"%CHECK_PY%" -c "import PyInstaller" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] PyInstaller missing in %CHECK_LABEL% environment
    exit /b 1
)
echo [SUCCESS] PyInstaller is available (%CHECK_LABEL%)
exit /b 0

:build_with_python
set "B_PY=%~1"
set "B_WORK=%~2"
set "B_DIST=%~3"
set "B_LABEL=%~4"

echo.
echo [========================================]
echo Build: %B_LABEL%
echo [========================================]

if exist "%B_WORK%" rd /s /q "%B_WORK%"
if exist "%B_DIST%" rd /s /q "%B_DIST%"

"%B_PY%" -m PyInstaller "%SPEC_FILE%" --clean --noconfirm --workpath "%B_WORK%" --distpath "%B_DIST%"
if errorlevel 1 (
    echo [ERROR] PyInstaller failed (%B_LABEL%)
    exit /b 1
)

if not exist "%B_DIST%\%APP_NAME%\SuperPicky.exe" (
    echo [ERROR] Missing output exe: %B_DIST%\%APP_NAME%\SuperPicky.exe
    exit /b 1
)

echo [SUCCESS] Build completed (%B_LABEL%)
exit /b 0

rem Zip folder Z_SRC into Z_OUT. Archive contains one top-level folder (e.g. SuperPicky\) so unzip gives one dir.
:zip_dir
set "Z_SRC=%~1"
set "Z_OUT=%~2"

if not exist "%Z_SRC%" (
    echo [ERROR] Zip source not found: %Z_SRC%
    exit /b 1
)

if exist "%Z_OUT%" del /q "%Z_OUT%" >nul 2>&1

where 7z >nul 2>&1
if not errorlevel 1 (
    7z a -tzip "%Z_OUT%" "%Z_SRC%" -r >nul
    if errorlevel 1 (
        echo [ERROR] Failed to create zip with 7z: %Z_OUT%
        exit /b 1
    )
) else (
    powershell -NoProfile -Command "Compress-Archive -Path '%Z_SRC%' -DestinationPath '%Z_OUT%' -Force"
    if errorlevel 1 (
        echo [ERROR] Failed to create zip with Compress-Archive: %Z_OUT%
        exit /b 1
    )
)

echo [SUCCESS] Created zip: %Z_OUT%
exit /b 0

:build_single
set "WORK_DIR=build"
set "DIST_DIR=%OUT_DIST_DIR%"

call :build_with_python "%PYTHON_EXE%" "%WORK_DIR%" "%DIST_DIR%" "release"
if errorlevel 1 exit /b 1

rem Default: always create one release ZIP
if "%BUILD_ZIP%"=="1" (
    set "ZIP_NAME=%APP_NAME%_v%VERSION%_Win64.zip"
    call :zip_dir "%DIST_DIR%\%APP_NAME%" "%DIST_DIR%\%ZIP_NAME%"
    if errorlevel 1 exit /b 1
) else (
    set "ZIP_NAME="
    echo [INFO] ZIP creation skipped ^(--no-zip^)
)

echo.
echo [========================================]
echo Build finished
echo [========================================]
echo EXE: %DIST_DIR%\%APP_NAME%\SuperPicky.exe
if defined ZIP_NAME (
    echo ZIP: %DIST_DIR%\%ZIP_NAME%
) else (
    echo ZIP: ^(skipped^)
)
exit /b 0

:restore_build_info
if exist "%BUILD_INFO_BACKUP%" (
    move /y "%BUILD_INFO_BACKUP%" "%BUILD_INFO_FILE%" >nul
)
exit /b 0
