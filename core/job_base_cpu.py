# -*- coding: utf-8 -*-

import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict, List, Tuple
from dataclasses import dataclass, field

from core.job_base import JobBase
class JobBaseCPU(JobBase):    
    def __init__(self, job_id: str):
        super().__init__(job_id)

    @abstractmethod
    def do_job(self):
        pass
