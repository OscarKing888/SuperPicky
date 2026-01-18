# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from concurrent.futures import Future

from core.job_base import JobFileInfo
from core.job_base import JobBase

class JobWorker(ABC):

    @abstractmethod
    def submit(self, job) -> Future:
        pass
    
    @abstractmethod
    def _run_job(self, job):
        pass

    @abstractmethod
    def create_rate_job(self, job_file_info: JobFileInfo) -> JobBase:
        pass

    @abstractmethod
    def shutdown(self, wait: bool = True):
        pass
