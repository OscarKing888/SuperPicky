#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境初始化模块
必须在导入任何其他模块之前执行，确保工作目录和路径正确设置
"""

import sys
import os

def init_environment():
    """初始化Python环境，避免numpy导入错误"""
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
    # 移除当前目录（如果存在），然后重新添加（确保在最后）
    paths_to_remove = []
    for path in sys.path:
        # 检查是否是numpy源码目录
        if os.path.exists(path) and os.path.isdir(path):
            # 检查是否包含numpy的setup.py或__init__.py（numpy源码目录的特征）
            if os.path.exists(os.path.join(path, 'numpy', '__init__.py')):
                # 这是numpy源码目录，不应该在sys.path中
                if path != script_dir:  # 不要移除脚本目录本身
                    paths_to_remove.append(path)
    
    for path in paths_to_remove:
        if path in sys.path:
            sys.path.remove(path)
    
    # 将脚本目录添加到Python路径末尾（而不是开头）
    # 这样可以确保优先从site-packages导入numpy
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    
    return script_dir

# 自动执行初始化
_init_env_script_dir = init_environment()
