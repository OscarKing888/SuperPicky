@echo off
REM 下载 ExifTool Windows 版本
chcp 65001 >nul
echo 正在下载 ExifTool Windows 版本...

set TARGET_DIR=exiftool_bundle
set TARGET_FILE=%TARGET_DIR%\exiftool.exe

REM 确保目录存在
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

REM 尝试使用 curl 下载（Windows 10+ 自带）
echo 尝试从 SourceForge 下载...
curl -L -o "%TARGET_DIR%\exiftool.zip" "https://sourceforge.net/projects/exiftool/files/latest/download" 2>nul

if exist "%TARGET_DIR%\exiftool.zip" (
    echo 下载成功，正在解压...
    REM 使用 PowerShell 解压到临时目录
    set TEMP_DIR=%TARGET_DIR%\temp_extract
    powershell -Command "if (Test-Path '%TARGET_DIR%\exiftool.zip') { Expand-Archive -Path '%TARGET_DIR%\exiftool.zip' -DestinationPath '%TEMP_DIR%' -Force }"
    
    REM 查找 exe 文件
    set FOUND_EXE=0
    for /r "%TEMP_DIR%" %%f in (exiftool*.exe) do (
        echo 找到 exe 文件: %%f
        copy "%%f" "%TARGET_FILE%" >nul
        set FOUND_EXE=1
        set "EXE_DIR=%%~dpf"
        
        REM 查找同目录下的 DLL 文件
        if exist "!EXE_DIR!perl5*.dll" (
            echo 找到 DLL 文件，复制到 exiftool_files 目录...
            if not exist "%TARGET_DIR%\exiftool_files" mkdir "%TARGET_DIR%\exiftool_files"
            copy "!EXE_DIR!perl5*.dll" "%TARGET_DIR%\exiftool_files\" >nul
        )
        
        REM 查找 exiftool_files 目录
        for /d /r "%TEMP_DIR%" %%d in (exiftool_files) do (
            if exist "%%d" (
                echo 找到 exiftool_files 目录: %%d
                if not exist "%TARGET_DIR%\exiftool_files" mkdir "%TARGET_DIR%\exiftool_files"
                xcopy "%%d\*" "%TARGET_DIR%\exiftool_files\" /E /I /Y >nul
            )
        )
        
        REM 也检查父目录
        set "PARENT_DIR=!EXE_DIR!.."
        if exist "!PARENT_DIR!\exiftool_files" (
            echo 找到父目录中的 exiftool_files...
            if not exist "%TARGET_DIR%\exiftool_files" mkdir "%TARGET_DIR%\exiftool_files"
            xcopy "!PARENT_DIR!\exiftool_files\*" "%TARGET_DIR%\exiftool_files\" /E /I /Y >nul
        )
        
        goto :found
    )
    
    :found
    REM 清理临时文件
    if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
    if exist "%TARGET_DIR%\exiftool.zip" del "%TARGET_DIR%\exiftool.zip"
    
    if exist "%TARGET_FILE%" (
        echo.
        echo ? 下载成功！文件已保存到: %TARGET_FILE%
        if exist "%TARGET_DIR%\exiftool_files" (
            echo ? DLL 文件已复制到: %TARGET_DIR%\exiftool_files
            dir "%TARGET_DIR%\exiftool_files\*.dll" /b
        ) else (
            echo ??  未找到 DLL 文件，exe 可能需要 DLL 才能运行
            echo    请参考 EXIFTOOL_WINDOWS_SETUP.md 获取完整版本
        )
        echo.
        pause
        exit /b 0
    )
)

echo.
echo ? 自动下载失败
echo.
echo ? 请手动下载:
echo    1. 访问 https://exiftool.org/
echo    2. 点击页面上的下载链接
echo    3. 下载 Windows 版本 (exiftool(-k).exe 或 ZIP 文件)
echo    4. 将 exiftool(-k).exe 重命名为 exiftool.exe
echo    5. 放到目录: %CD%\%TARGET_DIR%
echo.
pause
