#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查虚拟环境健康状态
诊断Python版本升级后的依赖问题
"""

import sys
import os
import subprocess

def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("Python环境检查")
    print("=" * 60)
    print(f"\nPython版本: {sys.version}")
    print(f"Python可执行文件: {sys.executable}")
    print(f"Python路径: {sys.executable}")
    
    # 检查是否是虚拟环境
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    print(f"虚拟环境: {'是' if in_venv else '否'}")
    if in_venv:
        print(f"虚拟环境路径: {sys.prefix}")

def check_package(package_name, import_name=None, test_code=None):
    """检查包是否正常安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        if test_code:
            exec(test_code)
        else:
            __import__(import_name)
        return True, None
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"导入错误: {str(e)}"

def main():
    """主函数"""
    check_python_version()
    
    print("\n" + "=" * 60)
    print("依赖包检查")
    print("=" * 60)
    
    # 要检查的包列表
    packages = [
        ("numpy", "numpy", "import numpy; numpy.__version__"),
        ("rawpy", "rawpy", "import rawpy; import rawpy._rawpy"),
        ("PySide6", "PySide6.QtWidgets", "from PySide6.QtWidgets import QApplication"),
        ("Pillow", "PIL", "from PIL import Image; Image.__version__"),
        ("opencv-python", "cv2", "import cv2"),
        ("torch", "torch", "import torch"),
        ("ultralytics", "ultralytics", "import ultralytics"),
    ]
    
    results = []
    for pkg_name, import_name, test_code in packages:
        ok, error = check_package(pkg_name, import_name, test_code)
        status = "[OK]" if ok else "[FAIL]"
        print(f"\n{status} {pkg_name}")
        if not ok:
            print(f"    错误: {error}")
            results.append((pkg_name, False, error))
        else:
            results.append((pkg_name, True, None))
    
    # 统计
    print("\n" + "=" * 60)
    print("检查结果统计")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed
    
    print(f"\n总计: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    
    if failed > 0:
        print("\n失败的包:")
        for pkg_name, ok, error in results:
            if not ok:
                print(f"  - {pkg_name}: {error}")
        
        print("\n建议:")
        print("  1. 运行 rebuild_venv.bat 重建虚拟环境")
        print("  2. 或者运行 fix_all_deps.bat 修复依赖")
        print("  3. 如果Python 3.14太新，考虑降级到3.12或3.13")
    else:
        print("\n✅ 所有依赖包检查通过！")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
