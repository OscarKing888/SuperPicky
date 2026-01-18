# -*- coding: utf-8 -*-

import threading
from typing import Optional, Callable, Any, Dict

from core.job_base import JobFileInfo
from core.job_base import JobBase
from core.job_base_cpu_rate import JobBaseCPU_Rate
from core.job_base_cpu_convert_heif import JobBaseCPU_ConvertHEIF
from core.job_base_cpu_write_exif import JobBaseCPU_WriteEXIF


class CPUJobWorker:
    """
    CPU Job Worker（仅负责模型/设备创建和注入，不管理线程池）

    - 负责CPU端模型/检测器/EXIF工具的创建与复用（线程本地）
    - 负责为 JobBaseCPU_Rate 评分任务注入设备/模型/检测器
    - 线程调度由 JobManager 管理
    """

    def __init__(
        self,
        log_callback: Optional[Callable[[str, str], None]] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.log_callback = log_callback
        self._local = threading.local()

    
    def create_rate_job(
        self,
        job_file_info: JobFileInfo,
        photo_processor,
        raw_dict: Dict[str, str],
    ) -> JobBaseCPU_Rate:
        """创建CPU评分任务"""
        return JobBaseCPU_Rate(
            job_file_info=job_file_info,
            photo_processor=photo_processor,
            raw_dict=raw_dict,
        )

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

    def run_job(self, job):
        """
        执行job（由JobManager的线程池调用）
        
        Args:
            job: 待执行的job对象
            
        Returns:
            job的执行结果
        """
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