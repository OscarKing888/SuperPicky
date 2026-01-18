# -*- coding: utf-8 -*-

import queue
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Optional, Callable, Any, Dict, List, Tuple
from dataclasses import dataclass, field

from core.job_manager_worker import JobWorker
from core.job_base_gpu_rate import JobBaseGPU_Rate


class GPUJobWorker(JobWorker):
    """
    GPU Job Worker（线程池封装）

    - 负责GPU线程池调度（并发数由 JobManager 传入）
    - 负责GPU端模型/检测器/EXIF工具的创建与复用（线程本地）
    - 负责执行 JobBaseGPU_Rate 评分任务
    """

    def __init__(
        self,
        max_workers: int,
        log_callback: Optional[Callable[[str, str], None]] = None,
        device: Optional[str] = None,
    ):
        self.max_workers = max_workers
        self.log_callback = log_callback
        self.device = device or self._detect_best_device()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._local = threading.local()

    def _log(self, msg: str, level: str = "info"):
        if self.log_callback:
            self.log_callback(msg, level)

    def _detect_best_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            try:
                if torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
        except Exception:
            pass
        return "cpu"

    def _get_context(self, photo_processor) -> Dict[str, Any]:
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
        if isinstance(job, JobBaseGPU_Rate):
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

