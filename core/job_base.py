# -*- coding: utf-8 -*-

import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict, List, Tuple
from dataclasses import dataclass, field

class JobBase(ABC):
    job_id: str
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None

    def set_file_path(self, file_path: str):
        self.file_path = file_path

    def start_time(self):
        self.started_at = time.time()
        
    def end_time(self):
        self.completed_at = time.time()

    def reset_job(self):
        self.error = None
        self.started_at = None
        self.completed_at = None

    def run_job(self):
        self.start_time()        
        self.do_job()
        self.end_time()
    
    def do_job(self):
        pass
        
    
    @property
    def duration(self) -> float:
        """任务耗时（秒）"""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return 0.0