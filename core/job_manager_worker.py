# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Dict, Any

from core.job_base import JobFileInfo
from core.job_base import JobBase

class JobWorker(ABC):

    @abstractmethod
    def submit(self, job) -> Future:
        """提交任务到执行器（由JobManager管理，worker不需要实现）"""
        pass
    
    @abstractmethod
    def _run_job(self, job):
        """执行job（由JobManager的线程池调用）"""
        pass

    @abstractmethod
    def create_rate_job(
        self,
        job_file_info: JobFileInfo,
        photo_processor,
        raw_dict: Dict[str, str],
    ) -> JobBase:
        """创建评分任务"""
        pass

    @abstractmethod
    def shutdown(self, wait: bool = True):
        """关闭worker（由JobManager管理，worker不需要实现）"""
        pass
