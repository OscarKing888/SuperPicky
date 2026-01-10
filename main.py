#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky - PySide6 版本入口点
Version: 3.7.0 - TOPIQ Aesthetic Model
"""

# ============================================================================
# 关键：必须在导入任何其他模块之前初始化环境
# 这可以避免numpy从错误的位置导入
# ============================================================================
# 注意：不能使用import，必须直接执行代码
import sys
import os

# 获取脚本所在目录的绝对路径
if hasattr(sys, 'frozen'):
    # PyInstaller打包后的情况
    script_dir = os.path.dirname(sys.executable)
else:
    # 开发环境
    script_dir = os.path.dirname(os.path.abspath(__file__))

# 立即切换到脚本目录（在任何导入之前）
if os.path.exists(script_dir):
    os.chdir(script_dir)

# 清理sys.path中可能存在的numpy源码目录
# 注意：只移除明确的numpy源码目录，不要影响site-packages
paths_to_remove = []
for path in sys.path[:]:  # 使用切片复制，避免迭代时修改
    if os.path.exists(path) and os.path.isdir(path):
        # 检查是否是numpy源码目录（包含numpy/__init__.py）
        # 但排除site-packages目录（可能包含已安装的numpy）
        numpy_init = os.path.join(path, 'numpy', '__init__.py')
        # 只移除明确的源码目录（不在site-packages中，且不是脚本目录）
        is_site_packages = 'site-packages' in path or 'Lib' in path
        if os.path.exists(numpy_init) and path != script_dir and not is_site_packages:
            paths_to_remove.append(path)

for path in paths_to_remove:
    if path in sys.path:
        sys.path.remove(path)

# 将脚本目录添加到Python路径末尾（而不是开头）
# 这样可以确保优先从site-packages导入numpy和PySide6
if script_dir not in sys.path:
    sys.path.append(script_dir)

# 现在可以安全导入其他模块
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from ui.main_window import SuperPickyMainWindow


def main():
    """主函数"""
    # 创建应用
    app = QApplication(sys.argv)
    
    # 设置应用属性
    app.setApplicationName("SuperPicky")
    app.setApplicationDisplayName("慧眼选鸟")
    app.setOrganizationName("JamesPhotography")
    app.setOrganizationDomain("jamesphotography.com.au")
    
    # 设置应用图标
    icon_path = os.path.join(os.path.dirname(__file__), "img", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # 注：Qt6/PySide6 默认启用 HiDPI 支持，无需手动设置
    
    # 创建主窗口
    window = SuperPickyMainWindow()
    window.show()
    
    # 运行事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
