# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Dict, Any

from core.job_base import JobFileInfo


@dataclass
class RatingInfo:
    """评星信息封装类"""
    job_file_info: JobFileInfo
    rating: int
    pick: int
    reason: str
    exif_data: Dict[str, Any]  # 包含所有EXIF相关数据
