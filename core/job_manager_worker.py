# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from concurrent.futures import Future

class JobWorker(ABC):

    @abstractmethod
    def submit(self, job) -> Future:
        pass
    
    @abstractmethod
    def _run_job(self, job):
        pass

    @abstractmethod
    def shutdown(self, wait: bool = True):
        pass
