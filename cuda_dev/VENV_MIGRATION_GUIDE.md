# 虚拟环境迁移指南

## 🔍 问题诊断

当你从Python 3.12升级到3.14，或从另一台机器复制项目后，可能会遇到：

1. ❌ `No module named 'numpy._core._multiarray_umath'`
2. ❌ `No module named 'rawpy._rawpy'`
3. ❌ `cannot import name '_imaging' from 'PIL'`
4. ❌ `No module named 'PySide6'`（即使pip显示已安装）

**根本原因**：虚拟环境中的二进制扩展模块（.pyd/.so）是为旧Python版本编译的，与新版本不兼容。

## ✅ 彻底解决方案

### 方案1：重建虚拟环境（推荐）

**步骤1：运行诊断脚本**
```bash
python check_venv_health.py
```
这会检查所有依赖包的健康状态。

**步骤2：重建虚拟环境**
```bash
rebuild_venv.bat
```

这个脚本会：
1. 删除旧的 `.venv` 目录
2. 使用当前Python版本创建新虚拟环境
3. 重新安装所有依赖
4. 验证安装结果

### 方案2：手动重建（如果脚本失败）

```bash
# 1. 删除旧虚拟环境
rmdir /s /q .venv

# 2. 创建新虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
.venv\Scripts\activate.bat

# 4. 升级pip
python -m pip install --upgrade pip setuptools wheel

# 5. 安装依赖
pip install -r requirements.txt
```

### 方案3：降级Python版本（最稳定）

如果Python 3.14太新，某些包可能还不支持：

```bash
# 1. 安装Python 3.12或3.13
# 从 https://www.python.org/downloads/ 下载

# 2. 删除旧虚拟环境
rmdir /s /q .venv

# 3. 使用Python 3.12创建虚拟环境
python3.12 -m venv .venv

# 4. 激活并安装依赖
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## 🔧 常见问题修复

### 问题1：PIL._imaging缺失

**症状**：`cannot import name '_imaging' from 'PIL'`

**原因**：Pillow的二进制扩展损坏

**修复**：
```bash
pip uninstall Pillow -y
pip install --no-cache-dir Pillow
```

### 问题2：numpy C扩展缺失

**症状**：`No module named 'numpy._core._multiarray_umath'`

**修复**：
```bash
pip uninstall numpy -y
pip install --no-cache-dir numpy
```

或运行：
```bash
fix_numpy.bat
```

### 问题3：rawpy C扩展缺失

**症状**：`No module named 'rawpy._rawpy'`

**修复**：
```bash
pip uninstall rawpy -y
pip install --no-cache-dir rawpy
```

或运行：
```bash
fix_rawpy.bat
```

**注意**：如果仍然失败，可能需要安装Visual C++ Build Tools来编译C扩展。

### 问题4：PySide6导入失败

**症状**：`No module named 'PySide6'`（即使pip显示已安装）

**修复**：
```bash
pip uninstall PySide6 shiboken6 PySide6-Essentials PySide6-Addons -y
pip install --force-reinstall --no-cache-dir PySide6
```

或运行：
```bash
fix_pyside6.bat
```

## 📋 一键修复所有问题

运行：
```bash
fix_all_deps.bat
```

这会依次修复numpy、rawpy、PySide6等所有依赖。

## ⚠️ 重要提示

### Python 3.14兼容性警告

Python 3.14是一个非常新的版本，某些包可能还没有完全支持：

- ✅ **已测试支持**：numpy, PySide6（部分版本）
- ⚠️ **可能有问题**：rawpy（需要从源码编译）
- ⚠️ **需要验证**：torch, ultralytics

### 推荐方案

1. **生产环境**：使用Python 3.12或3.13（更稳定）
2. **开发环境**：可以尝试Python 3.14，但需要准备处理兼容性问题

### 从其他机器迁移项目

**正确做法**：
1. ❌ **不要**复制 `.venv` 目录
2. ✅ **只复制**源代码和配置文件
3. ✅ 在新机器上重新创建虚拟环境

**应该复制的文件/目录**：
- 源代码（`.py`文件）
- `requirements.txt`
- `models/`（模型文件）
- `img/`（资源文件）
- 配置文件（`advanced_config.json`等）

**不应该复制**：
- `.venv/`（虚拟环境）
- `__pycache__/`（Python缓存）
- `.pyc`文件

## 🧪 验证安装

运行健康检查：
```bash
python check_venv_health.py
```

应该看到所有包都显示 `[OK]`。

## 📝 故障排除清单

如果重建虚拟环境后仍有问题：

1. ✅ 确认Python版本：`python --version`
2. ✅ 确认虚拟环境激活：`where python` 应该指向 `.venv\Scripts\python.exe`
3. ✅ 检查pip版本：`pip --version`
4. ✅ 尝试升级pip：`python -m pip install --upgrade pip`
5. ✅ 检查网络连接（如果需要从PyPI下载）
6. ✅ 检查磁盘空间
7. ✅ 检查杀毒软件是否阻止安装

## 🔗 相关文件

- `rebuild_venv.bat` - 重建虚拟环境脚本
- `check_venv_health.py` - 健康检查脚本
- `fix_all_deps.bat` - 一键修复所有依赖
- `fix_numpy.bat` - 修复numpy
- `fix_rawpy.bat` - 修复rawpy
- `fix_pyside6.bat` - 修复PySide6
