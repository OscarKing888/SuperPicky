# -*- coding: utf-8 -*-

import queue
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Optional, Callable, Any, Dict, List, Tuple
from dataclasses import dataclass, field

from core.job_base_cpu_convert_heif import JobBaseCPU_ConvertHEIF
from core.job_base_cpu_rate import JobBaseCPU_Rate
from core.job_base_cpu_write_exif import JobBaseCPU_WriteEXIF

from core.job_manager_worker import JobWorker

class CPUJobWorker(JobWorker):
    """
    CPU Job Worker（线程池封装）

    - 负责CPU线程池调度
    - 负责CPU端模型/检测器/EXIF工具的创建与复用（线程本地）
    - 负责执行 JobBaseCPU_Rate 评分任务以及其他纯CPU job（convert/exif等）
    """

    def __init__(
        self,
        max_workers: int,
        log_callback: Optional[Callable[[str, str], None]] = None,
        device: str = "cpu",
    ):
        self.max_workers = max_workers
        self.device = device
        self.log_callback = log_callback
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._local = threading.local()

    def _log(self, msg: str, level: str = "info"):
        if self.log_callback:
            self.log_callback(msg, level)

    def _get_context(self, photo_processor) -> Dict[str, Any]:
        """
        获取当前线程的推理上下文（懒加载）。
        仅在执行评分任务时才会触发加载。
        """
        ctx = getattr(self._local, "ctx", None)
        if ctx is None:
            ctx = {}
            self._local.ctx = ctx

        if "exiftool_mgr" not in ctx:
            from exiftool_manager import get_exiftool_manager
            ctx["exiftool_mgr"] = get_exiftool_manager()

        if "yolo_model" not in ctx:
            from ai_model import load_yolo_model
            try:
                ctx["yolo_model"] = load_yolo_model(device=self.device)
            except TypeError:
                ctx["yolo_model"] = load_yolo_model()

        if "keypoint_detector" not in ctx:
            from core.keypoint_detector import get_keypoint_detector
            detector = get_keypoint_detector()
            try:
                detector.load_model()
                ctx["keypoint_detector"] = detector
            except FileNotFoundError:
                ctx["keypoint_detector"] = None

        if "flight_detector" not in ctx:
            if getattr(photo_processor.settings, "detect_flight", False):
                from core.flight_detector import get_flight_detector
                detector = get_flight_detector()
                try:
                    detector.load_model()
                    ctx["flight_detector"] = detector
                except FileNotFoundError:
                    ctx["flight_detector"] = None
            else:
                ctx["flight_detector"] = None

        return ctx

    def submit(self, job) -> Future:
        return self._executor.submit(self._run_job, job)

    def _run_job(self, job):
        # 评分任务：由worker注入设备/模型/检测器/EXIF管理器
        if isinstance(job, JobBaseCPU_Rate):
            ctx = self._get_context(job.photo_processor)
            job.device = self.device
            job.model = ctx.get("yolo_model")
            job.keypoint_detector = ctx.get("keypoint_detector")
            job.flight_detector = ctx.get("flight_detector")
            job.exiftool_mgr = ctx.get("exiftool_mgr")

        job.run_job()
        if hasattr(job, "get_result"):
            return job.get_result()
        return None

    def shutdown(self, wait: bool = True):
        self._executor.shutdown(wait=wait)