# -*- coding: utf-8 -*-

import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict, List, Tuple
from dataclasses import dataclass, field

from core.job_base import JobBase
from core.job_base import JobFileInfo

class JobBaseCPU(JobBase):
    def __init__(self, job_file_info: JobFileInfo):
        super().__init__(job_file_info)

    @abstractmethod
    def do_job(self):
        pass
