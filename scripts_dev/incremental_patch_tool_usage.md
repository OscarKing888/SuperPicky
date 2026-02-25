# SuperPicky 增量补丁工具使用说明

本文说明 `scripts_dev/incremental_patch_tool.py` 与 `build_incremental_patch.bat` 的用法。

适用场景：

- 对比“上一个版本目录”和“当前版本目录”，提取新增/修改文件
- 生成可直接覆盖安装目录的增量补丁（含删除清单）
- 制作 `CPU` 版本到 `CUDA` 版本的 GPU 补丁包


## 1. 工具位置

- Python 工具：`scripts_dev/incremental_patch_tool.py`
- Windows 包装脚本：`build_incremental_patch.bat`


## 2. 补丁目录结构

生成的补丁目录默认包含：

- `payload/`：需要复制到目标目录的新增/修改文件
- `patch_manifest.json`：补丁清单（新增/修改/删除文件列表、统计信息）

可选：

- `xxx.zip`：补丁目录的压缩包（`build_incremental_patch.bat` 默认会生成）


## 3. 快速使用（Windows）

### 3.1 生成补丁（常用）

对比两个目录，生成补丁目录并打包 zip：

```bat
build_incremental_patch.bat "<旧目录>" "<新目录>" [补丁输出目录] [zip输出路径]
```

示例 1：制作 CPU -> CUDA 补丁（仅提取差异文件）

```bat
build_incremental_patch.bat "dist_cpu\SuperPicky" "dist_cuda\SuperPicky" "patches\gpu_patch_cu118"
```

示例 2：制作版本增量补丁（上版本 -> 当前版本）

```bat
build_incremental_patch.bat "E:\_SuperPickyVersions\SuperPicky_4.1.0" "dist_cuda\SuperPicky" "patches\v4.1.1_cuda_patch"
```

说明：

- 第 3 个参数不填时，默认输出到 `patches\incremental_patch`
- 第 4 个参数不填时，默认生成 `补丁目录同级的 .zip`


### 3.2 应用补丁（推荐）

使用 Python 工具应用补丁到安装目录（支持删除旧文件）：

```bat
py -3 scripts_dev\incremental_patch_tool.py apply --patch "patches\v4.1.1_cuda_patch" --target "C:\Apps\SuperPicky"
```

推荐在应用前先关闭 SuperPicky。


### 3.3 预演（不实际写入）

```bat
py -3 scripts_dev\incremental_patch_tool.py apply --patch "patches\v4.1.1_cuda_patch" --target "C:\Apps\SuperPicky" --dry-run
```

适合先确认复制/删除数量是否符合预期。


### 3.4 保留旧文件（跳过删除）

```bat
py -3 scripts_dev\incremental_patch_tool.py apply --patch "patches\v4.1.1_cuda_patch" --target "C:\Apps\SuperPicky" --skip-delete
```

适合保守升级，但可能残留旧版本文件。


## 4. 直接使用 Python 工具（跨平台）

该工具支持 Windows / macOS / Linux。

### 4.1 生成补丁

```bash
python scripts_dev/incremental_patch_tool.py make --old "/path/to/old" --new "/path/to/new" --out "/path/to/patch" --zip
```

可选参数：

- `--zip`：生成 zip
- `--zip-path <path>`：指定 zip 输出路径


### 4.2 应用补丁

```bash
python scripts_dev/incremental_patch_tool.py apply --patch "/path/to/patch" --target "/path/to/install"
```

可选参数：

- `--dry-run`：仅预演
- `--skip-delete`：不删除清单中的文件


## 5. `patch_manifest.json` 说明

清单中主要字段：

- `summary`
- `files.added`
- `files.modified`
- `files.deleted`

`apply` 命令会：

- 复制 `payload/` 中的新增文件
- 覆盖 `payload/` 中的修改文件
- 删除 `files.deleted` 中列出的旧文件（除非指定 `--skip-delete`）


## 6. 推荐工作流

### 场景 A：CPU 基础包 + CUDA 补丁包

1. 先构建 CPU 包（`dist_cpu\SuperPicky`）
2. 再构建 CUDA 包（`dist_cuda\SuperPicky`）
3. 对比生成补丁：

```bat
build_incremental_patch.bat "dist_cpu\SuperPicky" "dist_cuda\SuperPicky" "patches\gpu_patch_cu118"
```

4. 发布：

- `CPU 完整包`
- `GPU 补丁包（gpu_patch_cu118.zip）`

用户安装方式：

1. 安装 CPU 完整包
2. 解压 GPU 补丁包
3. 执行 `apply` 命令，或手工覆盖（手工覆盖不会自动删除旧文件）


### 场景 B：版本增量更新包

1. 准备上一个版本目录（已发布版本）
2. 构建当前版本目录
3. 对比生成增量补丁
4. 发布“完整包 + 增量补丁包”


## 7. 注意事项

- 补丁生成基于文件内容哈希（`SHA-256`），不是只看时间戳，适合发布场景
- 应用补丁前请关闭程序，避免 DLL / EXE 被占用
- `CPU -> CUDA` 补丁必须与目标 CPU 包版本匹配（同一版本号/同一构建基线）
- 如果目标目录不是补丁清单对应的旧版本，应用后可能出现不一致
- 手工解压覆盖不会执行删除清单，建议使用 `apply`


## 8. 常用命令汇总

查看帮助：

```bat
py -3 scripts_dev\incremental_patch_tool.py -h
py -3 scripts_dev\incremental_patch_tool.py make -h
py -3 scripts_dev\incremental_patch_tool.py apply -h
```

Windows 快速入口帮助：

```bat
build_incremental_patch.bat
```


## 9. 接入 `build_release_all.bat`（自动生成 GPU 补丁）

你当前的 `build_release_all.bat` 是先构建 CPU，再构建 CUDA：

- `build_release_cpu.bat` -> 输出 `dist_cpu\SuperPicky`
- `build_release_cuda.bat` -> 输出 `dist_cuda\SuperPicky`

因此最自然的接入点是：

1. CPU 构建成功
2. CUDA 构建成功
3. 自动执行目录对比，生成 `CPU -> CUDA` GPU 补丁包


### 9.1 最小接入（直接在 `build_release_all.bat` 末尾追加）

适合先快速验证流程。

示例（在两次 `call` 之后追加）：

```bat
@echo off
setlocal EnableExtensions

call build_release_cpu.bat %1
if errorlevel 1 exit /b 1

call build_release_cuda.bat %1
if errorlevel 1 exit /b 1

rem 生成 CPU -> CUDA 补丁（仅差异文件）
call build_incremental_patch.bat "dist_cpu\SuperPicky" "dist_cuda\SuperPicky" "patches\gpu_patch_cu118"
if errorlevel 1 exit /b 1

exit /b 0
```

说明：

- 这会在每次 `build_release_all.bat` 成功后生成一个 GPU 补丁目录和 zip
- 输出目录可按版本号命名（见下一节）


### 9.2 带版本号命名的接入（推荐）

如果你传入版本号（例如 `4.1.1`），可以让补丁目录/zip 自动带版本信息，便于归档。

示例思路：

```bat
@echo off
setlocal EnableExtensions

set "VERSION_INPUT=%~1"
if "%VERSION_INPUT%"=="" (
    set "PATCH_TAG=latest"
) else (
    set "PATCH_TAG=%VERSION_INPUT%"
)

call build_release_cpu.bat %1
if errorlevel 1 exit /b 1

call build_release_cuda.bat %1
if errorlevel 1 exit /b 1

call build_incremental_patch.bat ^
  "dist_cpu\SuperPicky" ^
  "dist_cuda\SuperPicky" ^
  "patches\SuperPicky_GPU_Patch_%PATCH_TAG%_cu118" ^
  "patches\SuperPicky_GPU_Patch_%PATCH_TAG%_cu118.zip"
if errorlevel 1 exit /b 1

exit /b 0
```

命名建议：

- `SuperPicky_GPU_Patch_<版本号>_cu118`
- 明确标注 CUDA ABI（例如 `cu118`），避免用户装错补丁


### 9.3 更稳的方案：新增 wrapper，不改现有 `build_release_all.bat`

如果你不想改已有脚本行为，建议新建一个包装脚本（例如 `build_release_all_with_patch.bat`）：

流程：

1. 调用现有 `build_release_all.bat`
2. 成功后调用 `build_incremental_patch.bat`

优点：

- 不影响已有 CI / 习惯命令
- 出问题时回退简单（不用改回主脚本）


### 9.4 发布建议（CPU 包 + GPU 补丁）

建议在发布目录中同时提供：

- `SuperPicky_vX.Y.Z_Win64_CPU.zip`
- `SuperPicky_GPU_Patch_X.Y.Z_cu118.zip`

并在发布说明中写明：

1. 先安装 CPU 包
2. 关闭程序
3. 应用 GPU 补丁（推荐 `apply` 命令）
4. 重启程序验证 CUDA 可用


### 9.5 验证清单（建议每次发布前）

- `dist_cpu\SuperPicky\SuperPicky.exe` 可启动
- `dist_cuda\SuperPicky\SuperPicky.exe` 可启动
- `patches\...\patch_manifest.json` 存在
- GPU 补丁应用后，CPU 安装目录可启动
- 有 NVIDIA 驱动的机器上可正常走 CUDA（至少完成模型预加载）

