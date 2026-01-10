# ExifTool Windows 版本安装说明

## 问题
Windows 版本的 `exiftool.exe` 需要 Perl DLL 文件才能运行。

## 解决方案

### 方案 1: 下载完整版本（推荐）

1. 访问 ExifTool 官方下载页面：
   - https://exiftool.org/
   - 或直接访问：https://sourceforge.net/projects/exiftool/

2. 下载 Windows 版本：
   - 下载 `exiftool-XX.XX.zip`（完整版本，包含 DLL）
   - 或下载 `exiftool(-k).exe`（独立版本，不需要 DLL）

3. 安装到项目目录：

   **如果是 ZIP 文件：**
   - 解压 ZIP 文件
   - 找到 `exiftool(-k).exe` 文件
   - 将其重命名为 `exiftool.exe`
   - 复制到 `exiftool_bundle` 目录
   - 如果 ZIP 中包含 `exiftool_files` 目录，也复制到 `exiftool_bundle` 目录

   **如果是独立的 exe 文件：**
   - 直接重命名为 `exiftool.exe`
   - 复制到 `exiftool_bundle` 目录

### 方案 2: 使用 Perl 脚本版本

1. 安装 Perl：
   - 下载并安装 Strawberry Perl: https://strawberryperl.com/
   - 或使用其他 Perl 发行版

2. 代码会自动检测并使用 Perl 脚本版本（`exiftool_bundle/exiftool`）

### 方案 3: 使用系统安装的 ExifTool

如果系统 PATH 中已有 ExifTool，代码会自动使用。

## 验证安装

运行以下命令验证：

```bash
python -c "from exiftool_manager import ExifToolManager; manager = ExifToolManager()"
```

如果看到 "✅ ExifTool已加载"，说明安装成功。

## 当前状态

- ✅ 已下载 `exiftool.exe` 到 `exiftool_bundle` 目录
- ❌ 缺少 Perl DLL 文件（`perl5*.dll`）
- ❌ 系统未安装 Perl

**建议：** 下载完整版本的 ZIP 文件，解压后复制所有文件到 `exiftool_bundle` 目录。
