# -*- coding: utf-8 -*-

import time
import os
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict, List, Tuple
from dataclasses import dataclass, field

class JobFileInfo:
    """文件信息封装"""
    
    HEIF_EXTENSIONS = ['.heic', '.heif', '.hif']

    def __init__(self, in_src_file_path: str):
        self.src_file_path = in_src_file_path
        
        file_prefix, file_ext = os.path.splitext(self.src_file_path)
        self.file_ext = file_ext.lower()

        if self.file_ext in self.HEIF_EXTENSIONS:
            self._use_tmp_file = True
            # 临时JPG路径（在temp_dir中）
            file_basename = os.path.splitext(os.path.basename(self.src_file_path))[0]
            self.tmp_file_path = None  # 将在转换时设置
        else:
            self._use_tmp_file = False
            self.tmp_file_path = None

    def needs_tmp_file(self) -> bool:
        """是否需要临时文件（HEIF需要转换）"""
        return self._use_tmp_file
    
    @property
    def file_prefix(self) -> str:
        """文件前缀（不含扩展名）"""
        return os.path.splitext(os.path.basename(self.src_file_path))[0]



class JobBase(ABC):
    """Job基类"""
    
    def __init__(self, job_file_info: JobFileInfo):
        self.job_file_info = job_file_info
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None

    def start_time(self):
        """记录开始时间"""
        self.started_at = time.time()
        
    def end_time(self):
        """记录结束时间"""
        self.completed_at = time.time()

    def reset_job(self):
        """重置任务状态"""
        self.error = None
        self.started_at = None
        self.completed_at = None

    def run_job(self):
        """运行任务"""
        self.start_time()        
        self.do_job()
        self.end_time()
    
    @abstractmethod
    def do_job(self):
        """执行任务逻辑（子类实现）"""
        pass
        
    @property
    def duration(self) -> float:
        """任务耗时（秒）"""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return 0.0
    
    @property
    def job_id(self) -> str:
        """任务ID（基于文件路径）"""
        return self.job_file_info.file_prefix