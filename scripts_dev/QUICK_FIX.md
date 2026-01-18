# 快速修复指南

## 🚀 一键解决所有问题

如果你从Python 3.12升级到3.14，或从其他机器复制了项目，运行：

```bash
rebuild_venv.bat
```

这会：
1. 删除旧的虚拟环境
2. 使用当前Python版本创建新虚拟环境
3. 重新安装所有依赖
4. 验证安装结果

## 🔍 检查问题

运行诊断：
```bash
python check_venv_health.py
```

## ⚡ 快速修复单个包

- **numpy问题**：`fix_numpy.bat`
- **rawpy问题**：`fix_rawpy.bat`
- **PySide6问题**：`fix_pyside6.bat`
- **所有依赖**：`fix_all_deps.bat`

## ⚠️ 如果修复失败

**降级到Python 3.12**（最稳定）：

```bash
# 1. 删除虚拟环境
rmdir /s /q .venv

# 2. 使用Python 3.12创建
python3.12 -m venv .venv

# 3. 安装依赖
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## 📋 常见错误

| 错误 | 解决方案 |
|------|---------|
| `No module named 'numpy._core._multiarray_umath'` | `fix_numpy.bat` |
| `No module named 'rawpy._rawpy'` | `fix_rawpy.bat` |
| `cannot import name '_imaging' from 'PIL'` | `pip uninstall Pillow -y && pip install Pillow` |
| `No module named 'PySide6'` | `fix_pyside6.bat` |

## 💡 最佳实践

**从其他机器迁移项目时**：
- ✅ 复制源代码和配置文件
- ❌ **不要**复制 `.venv` 目录
- ✅ 在新机器上运行 `rebuild_venv.bat`
