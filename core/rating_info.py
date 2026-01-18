# -*- coding: utf-8 -*-

import queue
from dataclasses import dataclass
from typing import Dict, Any

from core.job_base import JobFileInfo


@dataclass
class RatingInfo:
    """评星信息封装类"""
    job_file_info: JobFileInfo
    rating: int
    pick: int
    reason: str
    exif_data: Dict[str, Any]  # 包含所有EXIF相关数据


class RatingInfoQueue:
    """Thread-safe queue wrapper for rating info."""

    def __init__(self) -> None:
        self._queue: queue.Queue[RatingInfo] = queue.Queue()

    def put(self, rating_info: RatingInfo) -> None:
        self._queue.put(rating_info)

    def get_nowait(self) -> RatingInfo:
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()
