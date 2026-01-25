# -*- coding: utf-8 -*-

import os
import threading
from typing import Optional, Callable, Any, Dict

from core.job_base import JobFileInfo
from core.job_base import JobBase
from core.job_base_gpu_rate import JobBaseGPU_Rate
from core.job_manager_worker import JobWorker


class GPUJobWorker(JobWorker):
    """
    GPU Job Worker（仅负责模型/设备创建和注入，不管理线程池）

    - 负责GPU端模型/检测器/EXIF工具的创建与复用（线程本地）
    - 负责为 JobBaseGPU_Rate 评分任务注入设备/模型/检测器
    - 线程调度由 JobManager 管理
    """

    def __init__(
        self,
        log_callback: Optional[Callable[[str, str], None]] = None,
        device: Optional[str] = None,
    ):
        self.log_callback = log_callback
        self.device = device or self._detect_best_device()
        self._ctx = None
        self._ctx_lock = threading.Lock()
        self._mem_log_sizes = set()
        self._mem_samples = {}
        self._mem_est_logged = False

    def create_rate_job(
        self,
        job_file_info: JobFileInfo,
        photo_processor,
        raw_dict: Dict[str, str],
    ) -> JobBaseGPU_Rate:
        """创建GPU评分任务"""
        return JobBaseGPU_Rate(
            job_file_info=job_file_info,
            photo_processor=photo_processor,
            raw_dict=raw_dict,
        )

    def submit(self, job):
        """提交任务（由JobManager管理，worker不需要实现）"""
        raise NotImplementedError("submit should be called by JobManager's executor")

    def _run_job(self, job):
        """
        执行job（由JobManager的线程池调用）
        
        Args:
            job: 待执行的job对象
            
        Returns:
            job的执行结果
        """
        import traceback
        job_file_info = None
        try:
            if isinstance(job, JobBaseGPU_Rate):
                job_file_info = job.job_file_info
                self._log(f"[GPU Worker] 开始处理: {job_file_info.file_prefix if job_file_info else 'unknown'}")
                ctx = self._get_context(job.photo_processor)
                job.device = self.device
                job.model = ctx.get("yolo_model")
                job.keypoint_detector = ctx.get("keypoint_detector")
                job.flight_detector = ctx.get("flight_detector")
                job.exiftool_mgr = ctx.get("exiftool_mgr")

            job.run_job()
            if hasattr(job, "get_result"):
                result = job.get_result()
                self._log(f"[GPU Worker] 完成处理: {job_file_info.file_prefix if job_file_info else 'unknown'}")
                return result
            return None
        except Exception as e:
            error_msg = f"[GPU Worker] 执行任务异常"
            if job_file_info:
                error_msg += f": {job_file_info.file_prefix}"
            error_msg += f" - {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg, "error")
            # 返回错误结果而不是抛出异常
            if hasattr(job, 'job_file_info') and job.job_file_info:
                return {
                    'filename': job.job_file_info.file_prefix,
                    'filepath': job.job_file_info.src_file_path,
                    'rating': -1,
                    'reason': f'Worker异常: {str(e)}',
                    'processing_time': 0.0,
                    'job_file_info': job.job_file_info,
                }
            return None

    def shutdown(self, wait: bool = True):
        """关闭worker（由JobManager管理，worker不需要实现）"""
        pass

    def _run_batch(self, jobs):
        """
        Execute a batch of GPU rate jobs.
        """
        import time
        import traceback
        if not jobs:
            return []

        job_file_info = None
        try:
            job_file_info = jobs[0].job_file_info if hasattr(jobs[0], 'job_file_info') else None
            self._log(f"[GPU Worker] 开始批量处理: {len(jobs)} 张")

            ctx = self._get_context(jobs[0].photo_processor)
            model = ctx.get("yolo_model")
            keypoint_detector = ctx.get("keypoint_detector")
            flight_detector = ctx.get("flight_detector")
            exiftool_mgr = ctx.get("exiftool_mgr")

            image_paths = []
            meta = []
            for job in jobs:
                job_file_info = job.job_file_info
                if job_file_info.needs_tmp_file() and job_file_info.tmp_file_path:
                    inference_path = job_file_info.tmp_file_path
                else:
                    inference_path = job_file_info.src_file_path
                image_paths.append(inference_path)
                meta.append((job, job_file_info, inference_path))

            from ai_model import detect_and_draw_birds_batch
            ui_settings = jobs[0].photo_processor._build_ui_settings()

            batch_start_time = time.time()
            log_mem = len(jobs) in (1, 4) and len(jobs) not in self._mem_log_sizes
            mem_before = self._get_gpu_mem_snapshot() if log_mem else None

            try:
                yolo_results = detect_and_draw_birds_batch(
                    image_paths,
                    model,
                    jobs[0].photo_processor.dir_path,
                    ui_settings,
                    None,
                    skip_nima=True,
                    device=self.device,
                )
            except Exception as e:
                self._log(f"[GPU Worker] 批量YOLO失败，回退单张: {type(e).__name__}: {e}", "warning")
                yolo_results = None

            mem_after = self._get_gpu_mem_snapshot() if log_mem else None
            if log_mem:
                self._log_batch_memory(len(jobs), mem_before, mem_after)
                self._mem_log_sizes.add(len(jobs))

            results = []
            if yolo_results is None or len(yolo_results) != len(jobs):
                for job in jobs:
                    job.device = self.device
                    job.model = model
                    job.keypoint_detector = keypoint_detector
                    job.flight_detector = flight_detector
                    job.exiftool_mgr = exiftool_mgr
                    job.run_job()
                    result = job.get_result() if hasattr(job, "get_result") else None
                    if result:
                        result['device'] = self.device
                        result['job_file_info'] = job.job_file_info
                        results.append(result)
                return results

            for (job, job_info, _), yolo_result in zip(meta, yolo_results):
                result = job.photo_processor.process_single_image_with_yolo_result(
                    filename=os.path.basename(job_info.src_file_path),
                    filepath=job_info.src_file_path,
                    raw_dict=job.raw_dict,
                    yolo_result=yolo_result,
                    keypoint_detector=keypoint_detector,
                    flight_detector=flight_detector,
                    exiftool_mgr=exiftool_mgr,
                    use_keypoints=keypoint_detector is not None,
                    use_flight=flight_detector is not None and job.photo_processor.settings.detect_flight,
                    photo_start_time=batch_start_time,
                )
                if result:
                    result['device'] = self.device
                    result['job_file_info'] = job_info
                results.append(result)

            self._log(f"[GPU Worker] 完成批量处理: {len(results)} 张")
            return results
        except Exception as e:
            error_msg = f"[GPU Worker] 批量任务异常"
            if job_file_info:
                error_msg += f": {job_file_info.file_prefix}"
            error_msg += f" - {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg, "error")
            results = []
            for job in jobs:
                if hasattr(job, 'job_file_info') and job.job_file_info:
                    results.append({
                        'filename': job.job_file_info.file_prefix,
                        'filepath': job.job_file_info.src_file_path,
                        'rating': -1,
                        'reason': f'Worker批量异常: {str(e)}',
                        'processing_time': 0.0,
                        'job_file_info': job.job_file_info,
                    })
            return results

    def _get_gpu_mem_snapshot(self):
        try:
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                return {
                    "free": free_bytes,
                    "total": total_bytes,
                    "allocated": allocated,
                    "reserved": reserved,
                }
            if self.device == "mps" and hasattr(torch, "mps"):
                allocated = torch.mps.current_allocated_memory()
                return {
                    "free": None,
                    "total": None,
                    "allocated": allocated,
                    "reserved": None,
                }
        except Exception:
            return None
        return None

    def _log_batch_memory(self, batch_size, before, after):
        if not before or not after:
            return
        try:
            def _gb(value):
                if value is None:
                    return None
                return value / (1024 ** 3)

            free_before = _gb(before.get("free"))
            free_after = _gb(after.get("free"))
            alloc_before = _gb(before.get("allocated"))
            alloc_after = _gb(after.get("allocated"))
            reserved_before = _gb(before.get("reserved"))
            reserved_after = _gb(after.get("reserved"))

            msg = f"[GPU Batch] size={batch_size}"
            if free_before is not None and free_after is not None:
                msg += f" free_gb={free_before:.2f}->{free_after:.2f}"
            if alloc_before is not None and alloc_after is not None:
                msg += f" alloc_gb={alloc_before:.2f}->{alloc_after:.2f}"
            if reserved_before is not None and reserved_after is not None:
                msg += f" reserved_gb={reserved_before:.2f}->{reserved_after:.2f}"
            self._log(msg)
            self._record_batch_mem_sample(batch_size, before, after)
            self._maybe_log_batch_estimate()
        except Exception:
            pass

    def _record_batch_mem_sample(self, batch_size, before, after):
        sample = {}
        if before:
            sample["free_before"] = before.get("free")
            sample["total_before"] = before.get("total")
            sample["allocated_before"] = before.get("allocated")
            sample["reserved_before"] = before.get("reserved")
        if after:
            sample["free_after"] = after.get("free")
            sample["total_after"] = after.get("total")
            sample["allocated_after"] = after.get("allocated")
            sample["reserved_after"] = after.get("reserved")
        self._mem_samples[batch_size] = sample

    def _maybe_log_batch_estimate(self):
        if self._mem_est_logged:
            return
        sample_1 = self._mem_samples.get(1)
        sample_4 = self._mem_samples.get(4)
        if not sample_1 or not sample_4:
            return

        metric = None
        mem_1 = None
        mem_4 = None
        if sample_1.get("reserved_after") is not None and sample_4.get("reserved_after") is not None:
            metric = "reserved"
            mem_1 = sample_1.get("reserved_after")
            mem_4 = sample_4.get("reserved_after")
        elif sample_1.get("allocated_after") is not None and sample_4.get("allocated_after") is not None:
            metric = "allocated"
            mem_1 = sample_1.get("allocated_after")
            mem_4 = sample_4.get("allocated_after")
        if metric is None or mem_1 is None or mem_4 is None:
            return
        if mem_4 <= mem_1:
            return

        per_item = (mem_4 - mem_1) / 3.0
        if per_item <= 0:
            return
        overhead = max(0.0, mem_1 - per_item)
        free_bytes = sample_4.get("free_after") or sample_1.get("free_after")
        if free_bytes is None:
            return
        raw_batch = int((free_bytes - overhead) / per_item) if free_bytes > overhead else 0
        recommended = max(1, raw_batch)

        def _gb(value):
            return value / (1024 ** 3)

        mem_1_gb = _gb(mem_1)
        mem_4_gb = _gb(mem_4)
        per_item_gb = _gb(per_item)
        overhead_gb = _gb(overhead)
        free_gb = _gb(free_bytes)

        msg = (
            f"[GPU Batch] 估算公式: per_item_gb=({metric}_4_gb {mem_4_gb:.2f} - "
            f"{metric}_1_gb {mem_1_gb:.2f})/3={per_item_gb:.2f}; "
            f"overhead_gb={metric}_1_gb {mem_1_gb:.2f}-per_item_gb {per_item_gb:.2f}="
            f"{overhead_gb:.2f}; "
            f"推荐batch=floor((free_gb {free_gb:.2f}-overhead_gb {overhead_gb:.2f})/"
            f"per_item_gb {per_item_gb:.2f})={recommended}"
        )
        self._log(msg)
        self._mem_est_logged = True

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
        with self._ctx_lock:
            if self._ctx is None:
                self._ctx = {}
            ctx = self._ctx

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

