#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky - PySide6 版本入口点
Version: 3.6.0 - PySide6 Migration
"""

import sys
import os

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
