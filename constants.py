#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky 常量定义
统一管理全局常量，避免重复定义
"""

# 应用版本号
APP_VERSION = "3.9.3"

# 评分对应的文件夹名称映射
RATING_FOLDER_NAMES = {
    3: "3星_优选",
    2: "2星_良好",
    1: "1星_普通",
    0: "0星_放弃",
    -1: "0星_放弃"  # 无鸟照片也放入0星目录
}

# 支持的 RAW 文件扩展名（小写）
RAW_EXTENSIONS = ['.nef', '.cr2', '.cr3', '.arw', '.raf', '.orf', '.rw2', '.pef', '.dng', '.3fr', '.iiq']

# 支持的 JPG 文件扩展名（小写）
JPG_EXTENSIONS = ['.jpg', '.jpeg']

# oscar:支持的HEIF文件扩展名
HEIF_EXTENSIONS = ['.heic', '.heif', '.hif']

# 所有支持的图片扩展名（用于文件查找，包含大小写）
IMAGE_EXTENSIONS = (
    [ext.lower() for ext in RAW_EXTENSIONS] +
    [ext.upper() for ext in RAW_EXTENSIONS] +
    [ext.lower() for ext in JPG_EXTENSIONS] +
    [ext.upper() for ext in JPG_EXTENSIONS] +
    [ext.lower() for ext in HEIF_EXTENSIONS] +
    [ext.upper() for ext in HEIF_EXTENSIONS]
)
