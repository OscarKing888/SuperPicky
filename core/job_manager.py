# -*- coding: utf-8 -*-

import os
import queue
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Optional, Callable, Any, Dict, List, Tuple, Set

import torch

from constants import RAW_EXTENSIONS, JPG_EXTENSIONS
from core.job_base import JobFileInfo
from core.job_base_cpu_convert_heif import JobBaseCPU_ConvertHEIF
from core.job_base_cpu_rate import JobBaseCPU_Rate
from core.job_base_gpu_rate import JobBaseGPU_Rate
from core.job_base_cpu_write_exif import JobBaseCPU_WriteEXIF
from core.rating_info import RatingInfo, RatingInfoQueue
from core.photo_processor import PhotoProcessor

from core.job_manager_worker_cpu import CPUJobWorker
from core.job_manager_worker_gpu import GPUJobWorker
from advanced_config import get_advanced_config


class JobManager:
    """任务管理器 - 管理并执行各种类型的任务（负责线程池调度）"""
    
    def __init__(
        self,
        dir_path: str,
        photo_processor: PhotoProcessor,
        cpu_worker_count: Optional[int] = None,
        gpu_worker_count: Optional[int] = None,
        log_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        初始化任务管理器
        
        Args:
            dir_path: 处理目录路径
            photo_processor: PhotoProcessor实例
            cpu_worker_count: CPU工作线程数（None=自动）
            gpu_worker_count: GPU工作线程数（None=自动）
            log_callback: 日志回调函数
        """
        self.dir_path = dir_path
        self.photo_processor = photo_processor
        self.log_callback = log_callback
        
        # 计算worker数量
        cpu_total_limit = None
        cpu_rate_count_config = 0
        cpu_io_count = None
        self.cpu_rate_assist_enabled = False
        self.cpu_rate_backlog_threshold = 8
        self.gpu_single_thread_mode = False
        self.gpu_batch_size_config = 0
        self.gpu_batch_min_size = 1
        self.gpu_batch_max_size = 8
        self.gpu_batch_mem_per_item_gb = 1.0
        self.gpu_batch_mem_overhead_gb = 2.0
        self.gpu_batch_max_wait_ms = 0

        try:
            config = get_advanced_config()
            if cpu_worker_count is None:
                # 自动计算：使用配置中的最大限制
                cpu_total_limit = min(config.max_cpu_worker_count, multiprocessing.cpu_count())
                # 应用调整值
                cpu_total_limit += config.cpu_worker_count_adjust
                cpu_total_limit = min(cpu_total_limit, config.max_cpu_worker_count)
                cpu_total_limit = max(1, cpu_total_limit)  # 确保至少为1
            else:
                cpu_total_limit = max(1, cpu_worker_count)

            cpu_io_count = getattr(config, "cpu_io_worker_count", 0)
            cpu_rate_count_config = getattr(config, "cpu_rate_worker_count", 0)
            self.cpu_rate_backlog_threshold = max(0, int(getattr(config, "cpu_rate_backlog_threshold", 8)))
            self.cpu_rate_assist_enabled = bool(getattr(config, "cpu_rate_assist_enabled", True))
            self.gpu_single_thread_mode = bool(getattr(config, "gpu_single_thread_mode", False))
            self.gpu_batch_size_config = max(0, int(getattr(config, "gpu_batch_size", 0)))
            self.gpu_batch_min_size = max(1, int(getattr(config, "gpu_batch_min_size", 1)))
            self.gpu_batch_max_size = max(
                self.gpu_batch_min_size,
                int(getattr(config, "gpu_batch_max_size", 8)),
            )
            self.gpu_batch_mem_per_item_gb = float(getattr(config, "gpu_batch_mem_per_item_gb", 1.0))
            self.gpu_batch_mem_overhead_gb = float(getattr(config, "gpu_batch_mem_overhead_gb", 2.0))
            self.gpu_batch_max_wait_ms = max(0, int(getattr(config, "gpu_batch_max_wait_ms", 0)))
        except Exception as e:
            # 如果配置加载失败，使用默认值
            self._log(f"⚠️  加载高级配置失败，使用默认值: {e}", "warning")
            if cpu_worker_count is None:
                cpu_total_limit = min(64, multiprocessing.cpu_count())
            else:
                cpu_total_limit = max(1, cpu_worker_count)
            cpu_io_count = min(2, cpu_total_limit)
            cpu_rate_count_config = 0
            self.cpu_rate_backlog_threshold = 8
            self.cpu_rate_assist_enabled = False
            self.gpu_single_thread_mode = False
            self.gpu_batch_size_config = 0
            self.gpu_batch_min_size = 1
            self.gpu_batch_max_size = 8
            self.gpu_batch_mem_per_item_gb = 1.0
            self.gpu_batch_mem_overhead_gb = 2.0
            self.gpu_batch_max_wait_ms = 0

        if cpu_io_count is None or cpu_io_count <= 0:
            cpu_io_count = min(2, cpu_total_limit)
        cpu_io_count = max(1, min(cpu_io_count, cpu_total_limit))

        # 检测GPU设备
        gpu_device_str = None
        if torch.backends.mps.is_available():
            gpu_device_str = "mps"
        elif torch.cuda.is_available():
            gpu_device_str = "cuda"
        gpu_available = gpu_device_str is not None
        self.gpu_device_str = gpu_device_str
        self.gpu_available = gpu_available

        if cpu_rate_count_config is None:
            cpu_rate_count_config = 0

        if cpu_rate_count_config > 0:
            cpu_rate_count = cpu_rate_count_config
        else:
            if gpu_available:
                cpu_rate_count = max(0, min(4, cpu_total_limit - cpu_io_count))
            else:
                cpu_rate_count = max(1, cpu_total_limit - cpu_io_count)

        if gpu_available and not self.cpu_rate_assist_enabled:
            cpu_rate_count = 0

        if cpu_rate_count + cpu_io_count > cpu_total_limit:
            cpu_rate_count = max(0, cpu_total_limit - cpu_io_count)

        if not gpu_available and cpu_rate_count < 1:
            cpu_rate_count = 1
            if cpu_rate_count + cpu_io_count > cpu_total_limit:
                cpu_io_count = max(1, cpu_total_limit - cpu_rate_count)

        self.cpu_rate_worker_count = cpu_rate_count
        self.cpu_io_worker_count = cpu_io_count

        # 创建CPU workers列表（每个worker一个实例，用于负载均衡）
        self.cpu_rate_workers: List[CPUJobWorker] = [
            CPUJobWorker(log_callback=log_callback, device="cpu")
            for _ in range(cpu_rate_count)
        ]
        self.cpu_io_workers: List[CPUJobWorker] = [
            CPUJobWorker(log_callback=log_callback, device="cpu")
            for _ in range(cpu_io_count)
        ]
        self.cpu_rate_worker_index = 0  # 轮询索引
        self.cpu_io_worker_index = 0  # 轮询索引
        self.worker_index_lock = threading.Lock()  # 保护worker索引的锁
        self.busy_gpu_workers: Set[int] = set()
        
        # 创建CPU线程池（由JobManager管理）
        self.cpu_rate_executor = ThreadPoolExecutor(max_workers=cpu_rate_count) if cpu_rate_count > 0 else None
        self.cpu_io_executor = ThreadPoolExecutor(max_workers=cpu_io_count)
        
        # 创建GPU worker和线程池（如果GPU可用）
        if gpu_device_str is not None:
            if self.gpu_single_thread_mode:
                gpu_count = 1
            else:
                gpu_count = gpu_worker_count or self._calculate_gpu_workers(gpu_device_str)
            self.gpu_workers: List[GPUJobWorker] = [
                GPUJobWorker(log_callback=log_callback, device=gpu_device_str)
                for _ in range(gpu_count)
            ]
            self.gpu_worker_index = 0  # 轮询索引
            max_gpu_workers = 1 if self.gpu_single_thread_mode else gpu_count
            self.gpu_executor = ThreadPoolExecutor(max_workers=max_gpu_workers)
        else:
            self.gpu_workers = []
            self.gpu_executor = None
        
        self._log(
            f"💻 CPU评分Worker数量: {len(self.cpu_rate_workers)} | "
            f"💾 CPU IO Worker数量: {len(self.cpu_io_workers)} | "
            f"🖥️ GPU Worker数量: {len(self.gpu_workers)}"
        )
        self._debug_log(
            f"[调度] CPU评分辅助: {self.cpu_rate_assist_enabled}, "
            f"CPU评分线程: {self.cpu_rate_worker_count}, "
            f"CPU IO线程: {self.cpu_io_worker_count}, "
            f"评分队列阈值: {self.cpu_rate_backlog_threshold}"
        )
        if gpu_available:
            self._debug_log(
                f"[调度] GPU批量: size={self._resolve_gpu_batch_size()}, "
                f"min={self.gpu_batch_min_size}, max={self.gpu_batch_max_size}, "
                f"per_item_gb={self.gpu_batch_mem_per_item_gb:.2f}, "
                f"overhead_gb={self.gpu_batch_mem_overhead_gb:.2f}, "
                f"wait_ms={self.gpu_batch_max_wait_ms}, "
                f"fixed={self.gpu_batch_size_config}"
            )

        # 评星信息队列（步骤4：评星完成后保存到这里）
        self.rating_info_queue: RatingInfoQueue = RatingInfoQueue()
        
        # 评分任务队列（待评分的JobFileInfo）
        self.rate_job_queue: queue.Queue[JobFileInfo] = queue.Queue()

        self.rating_results_lock = threading.Lock()
        self.file_ratings: Dict[str, int] = {}
        self.star_3_photos: List[Dict[str, Any]] = []
        
        # 统计信息
        self.stats = {
            'convert_success': 0,
            'convert_failed': 0,
            'rate_success': 0,
            'rate_failed': 0,
            'exif_success': 0,
            'exif_failed': 0,
            'total': 0,
            'star_3': 0,
            'picked': 0,
            'star_2': 0,
            'star_1': 0,
            'star_0': 0,
            'no_bird': 0,
            'flying': 0,
            'exposure_issue': 0,
            'start_time': 0,
            'end_time': 0,
            'total_time': 0,
            'avg_time': 0,
        }
        self.stats_lock = threading.Lock()
        
        # 跟踪正在运行的任务（用于判断worker是否空闲）
        self.active_rate_futures: Dict[Future, Tuple[List[JobFileInfo], str, Optional[int]]] = {}
        self.active_convert_futures: Dict[Future, JobFileInfo] = {}
        self.active_exif_futures: List[Future] = []
        self.max_exif_in_flight = max(1, len(self.cpu_io_workers))
        self.futures_lock = threading.Lock()
        
        # 线程控制标志
        self.scan_complete = threading.Event()
        self.rate_complete = threading.Event()
        self.exif_complete = threading.Event()

    def _get_idle_cpu_rate_worker(self) -> Optional[CPUJobWorker]:
        """获取空闲的CPU评分worker（轮询方式，线程安全）"""
        if not self.cpu_rate_workers:
            return None
        with self.worker_index_lock:
            worker = self.cpu_rate_workers[self.cpu_rate_worker_index]
            self.cpu_rate_worker_index = (self.cpu_rate_worker_index + 1) % len(self.cpu_rate_workers)
            return worker

    def _get_idle_cpu_io_worker(self) -> Optional[CPUJobWorker]:
        """获取空闲的CPU IO worker（轮询方式，线程安全）"""
        if not self.cpu_io_workers:
            return None
        with self.worker_index_lock:
            worker = self.cpu_io_workers[self.cpu_io_worker_index]
            self.cpu_io_worker_index = (self.cpu_io_worker_index + 1) % len(self.cpu_io_workers)
            return worker
    
    def _get_idle_gpu_worker(self) -> Optional[Tuple[int, GPUJobWorker]]:
        """获取空闲的GPU worker（轮询方式，线程安全）"""
        if not self.gpu_workers:
            return None
        with self.worker_index_lock:
            for _ in range(len(self.gpu_workers)):
                worker_index = self.gpu_worker_index
                self.gpu_worker_index = (self.gpu_worker_index + 1) % len(self.gpu_workers)
                if worker_index not in self.busy_gpu_workers:
                    self.busy_gpu_workers.add(worker_index)
                    return worker_index, self.gpu_workers[worker_index]
        return None

    def _release_gpu_worker(self, worker_index: int) -> None:
        """释放GPU worker占用标记"""
        with self.worker_index_lock:
            self.busy_gpu_workers.discard(worker_index)

    def _has_idle_gpu_worker(self) -> bool:
        """检查是否有空闲的GPU worker"""
        if not self.gpu_workers:
            return False
        with self.worker_index_lock:
            return len(self.busy_gpu_workers) < len(self.gpu_workers)

    def _calculate_gpu_workers(self, device_str: str) -> int:
        """根据可用显存计算GPU并发数"""
        try:
            if device_str == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_memory_bytes, _ = torch.cuda.mem_get_info(0)
                free_memory_gb = free_memory_bytes / (1024 ** 3)
                gpu_workers = int(free_memory_gb / 1.5 - 2)
                # 应用配置中的调整值
                try:
                    config = get_advanced_config()
                    gpu_workers += config.gpu_worker_count_adjust
                    gpu_workers = min(gpu_workers, config.max_gpu_worker_count)
                except Exception as e:
                    self._log(f"⚠️  加载GPU Worker配置失败，使用默认调整值: {e}", "warning")
                return max(1, gpu_workers)
        except Exception:
            pass
        return 1

    def _log(self, msg: str, level: str = "info"):
        """内部日志方法"""
        if self.log_callback:
            self.log_callback(msg, level)
    
    def _debug_log(self, msg: str, level: str = "info"):
        """调试日志方法（可通过配置开关）"""
        try:
            config = get_advanced_config()
            if config.debug_log:
                self._log(f"[DEBUG] {msg}", level)
        except Exception:
            # 如果配置加载失败，默认输出调试日志
            self._log(f"[DEBUG] {msg}", level)
    
    def _scan_files(self) -> Tuple[Dict[str, str], List[JobFileInfo]]:
        """
        步骤1：扫描处理目录文件生成列表
        
        Returns:
            (raw_dict, job_file_info_list)
        """
        scan_start = time.time()
        raw_dict = {}
        job_file_info_list: List[JobFileInfo] = []
        
        for filename in os.listdir(self.dir_path):
            if filename.startswith('.'):
                continue
            
            file_prefix, file_ext = os.path.splitext(filename)
            ext_lower = file_ext.lower()
            
            # 记录RAW文件
            if ext_lower in RAW_EXTENSIONS:
                raw_dict[file_prefix] = file_ext
                continue
            
            # 处理JPG和HEIF文件
            if ext_lower in JPG_EXTENSIONS or ext_lower in JobFileInfo.HEIF_EXTENSIONS:
                filepath = os.path.join(self.dir_path, filename)
                job_file_info = JobFileInfo(filepath)
                job_file_info_list.append(job_file_info)
        
        scan_time = (time.time() - scan_start) * 1000
        self._log(f"⏱️  文件扫描耗时: {scan_time:.1f}ms (共 {len(job_file_info_list)} 个文件)")
        
        return raw_dict, job_file_info_list

    def _reset_stats(self) -> None:
        with self.stats_lock:
            for key in self.stats:
                self.stats[key] = 0

    def _reset_rating_results(self) -> None:
        with self.rating_results_lock:
            self.file_ratings.clear()
            self.star_3_photos.clear()

    def _clear_rating_queue(self) -> int:
        cleared = 0
        while not self.rating_info_queue.empty():
            try:
                self.rating_info_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        return cleared

    def _update_rating_stats(
        self,
        rating: int,
        pick: int,
        is_flying: bool,
        has_exposure_issue: bool,
    ) -> None:
        with self.stats_lock:
            self.stats['total'] += 1
            if rating == 3:
                self.stats['star_3'] += 1
            elif rating == 2:
                self.stats['star_2'] += 1
            elif rating == 1:
                self.stats['star_1'] += 1
            elif rating == 0:
                self.stats['star_0'] += 1
            else:
                self.stats['no_bird'] += 1

            if pick == 1:
                self.stats['picked'] += 1
            if is_flying:
                self.stats['flying'] += 1
            if has_exposure_issue:
                self.stats['exposure_issue'] += 1

    def _describe_job_infos(self, job_infos: List[JobFileInfo]) -> str:
        if not job_infos:
            return "unknown"
        if len(job_infos) == 1:
            return job_infos[0].file_prefix
        return f"{job_infos[0].file_prefix} +{len(job_infos) - 1}"

    def _handle_single_rate_result(self, result: Dict[str, Any]) -> None:
        if result:
            self._on_rate_complete(result)
            with self.stats_lock:
                if result.get('rating', -1) >= 0:
                    self.stats['rate_success'] += 1
                else:
                    self.stats['rate_failed'] += 1

    def _handle_rate_results(self, result: Any) -> None:
        if isinstance(result, list):
            for item in result:
                self._handle_single_rate_result(item)
        else:
            self._handle_single_rate_result(result)

    def _get_gpu_free_mem_gb(self) -> Optional[float]:
        if self.gpu_device_str != "cuda":
            return None
        try:
            torch.cuda.synchronize()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return free_bytes / (1024 ** 3)
        except Exception:
            return None

    def _resolve_gpu_batch_size(self) -> int:
        if not self.gpu_available:
            return 1
        if self.gpu_batch_size_config > 0:
            return max(self.gpu_batch_min_size, min(self.gpu_batch_size_config, self.gpu_batch_max_size))

        free_gb = self._get_gpu_free_mem_gb()
        if free_gb is None:
            return 1
        per_item = max(0.1, float(self.gpu_batch_mem_per_item_gb))
        overhead = max(0.0, float(self.gpu_batch_mem_overhead_gb))
        available = max(0.0, free_gb - overhead)
        if per_item <= 0:
            return 1
        batch_size = int(available / per_item)
        batch_size = max(self.gpu_batch_min_size, min(batch_size, self.gpu_batch_max_size))
        return max(1, batch_size)

    def _on_rate_complete(self, result: Dict[str, Any]):
        """
        步骤4：评星任务完成回调，保存评星信息到队列
        """
        if not result:
            return
        
        job_file_info = result.get('job_file_info')
        if not job_file_info:
            return
        
        rating_value = result.get('rating', 0)
        if rating_value is None:
            rating_value = 0
        pick_value = result.get('pick', 0)
        if pick_value is None:
            pick_value = 0
        is_flying = bool(result.get('is_flying', False))
        has_exposure_issue = bool(result.get('is_overexposed', False)) or bool(
            result.get('is_underexposed', False)
        )
        adj_sharpness = result.get('adj_sharpness')
        adj_topiq = result.get('adj_topiq')

        # 构建EXIF数据
        exif_data = {
            'rating': rating_value,
            'pick': pick_value,
            'reason': result.get('reason', ''),
            'confidence': result.get('confidence', 0.0),
            'head_sharpness': result.get('head_sharpness', 0.0),
            'topiq': result.get('topiq'),
            'adj_sharpness': adj_sharpness,
            'adj_topiq': adj_topiq,
            'is_flying': is_flying,
            'focus_status': result.get('focus_status'),
            'focus_sharpness_weight': result.get('focus_sharpness_weight', 1.0),
            'focus_topiq_weight': result.get('focus_topiq_weight', 1.0),
            'best_eye_visibility': result.get('best_eye_visibility', 0.0),
        }
        
        rating_info = RatingInfo(
            job_file_info=job_file_info,
            rating=rating_value,
            pick=pick_value,
            reason=result.get('reason', ''),
            exif_data=exif_data,
        )

        self._update_rating_stats(
            rating=rating_value,
            pick=pick_value,
            is_flying=is_flying,
            has_exposure_issue=has_exposure_issue,
        )
        with self.rating_results_lock:
            self.file_ratings[job_file_info.file_prefix] = rating_value
            if rating_value == 3 and adj_topiq is not None:
                self.star_3_photos.append(
                    {
                        'file': job_file_info.src_file_path,
                        'nima': adj_topiq,
                        'sharpness': adj_sharpness if adj_sharpness is not None else 0,
                    }
                )
        
        self.rating_info_queue.put(rating_info)

    def _file_scan_thread(self, job_file_info_list: List[JobFileInfo], temp_dir: str):
        """
        文件扫描线程：
        1. 如果是HEIF文件，创建转换任务，转换完成后将JobFileInfo放入评分队列
        2. 否则直接将JobFileInfo放入评分队列
        """
        self._debug_log(f"[文件扫描线程] 开始，共 {len(job_file_info_list)} 个文件")
        try:
            if not job_file_info_list:
                self._log("没有文件需要处理")
                self.scan_complete.set()
                self._debug_log("[文件扫描线程] 无文件，结束")
                return
            
            # 处理文件，创建转换任务或直接放入评分队列
            convert_count = 0
            direct_count = 0
            for job_file_info in job_file_info_list:
                if job_file_info.needs_tmp_file():
                    # 需要转换HEIF，创建转换任务
                    self._debug_log(f"[文件扫描线程] 创建HEIF转换任务: {job_file_info.file_prefix}")
                    convert_job = JobBaseCPU_ConvertHEIF(
                        job_file_info=job_file_info,
                        output_dir=temp_dir,
                    )
                    worker = self._get_idle_cpu_io_worker()
                    if worker is None:
                        worker = CPUJobWorker(log_callback=self.log_callback, device="cpu")
                    future = self.cpu_io_executor.submit(worker._run_job, convert_job)
                    
                    with self.futures_lock:
                        self.active_convert_futures[future] = job_file_info
                    convert_count += 1
                else:
                    # 直接放入评分队列
                    self._debug_log(f"[文件扫描线程] 直接放入评分队列: {job_file_info.file_prefix}")
                    self.rate_job_queue.put(job_file_info)
                    direct_count += 1
            self._debug_log(f"[文件扫描线程] 处理完成: {convert_count} 个转换任务, {direct_count} 个直接评分")
            
            # 等待所有转换任务完成，完成后将JobFileInfo放入评分队列
            if convert_count > 0:
                self._log(f"🔄 开始转换 {convert_count} 个HEIF文件...")
                while True:
                    # 在锁内检查是否还有未完成的任务
                    with self.futures_lock:
                        if not self.active_convert_futures:
                            break
                        completed_futures = []
                        for future in list(self.active_convert_futures.keys()):
                            if future.done():
                                completed_futures.append(future)
                    
                    for future in completed_futures:
                        with self.futures_lock:
                            job_file_info = self.active_convert_futures.pop(future)
                        try:
                            result = future.result()
                            if result and result.get('success'):
                                # 转换成功，更新job_file_info的tmp_file_path
                                job_file_info.tmp_file_path = result.get('temp_jpg_path')
                                # 放入评分队列
                                self.rate_job_queue.put(job_file_info)
                                
                                with self.stats_lock:
                                    self.stats['convert_success'] += 1
                            else:
                                with self.stats_lock:
                                    self.stats['convert_failed'] += 1
                        except Exception as e:
                            self._log(f"转换任务异常: {job_file_info.file_prefix} - {e}", "error")
                            with self.stats_lock:
                                self.stats['convert_failed'] += 1
                    
                    if not completed_futures:
                        time.sleep(0.01)  # 避免CPU空转
                
                self._log(f"✅ HEIF转换完成: 成功 {self.stats['convert_success']}, 失败 {self.stats['convert_failed']}")
            
            self.scan_complete.set()
            self._debug_log("[文件扫描线程] 正常结束")
        except Exception as e:
            import traceback
            self._log(f"文件扫描线程异常: {e}", "error")
            self._debug_log(f"[文件扫描线程] 异常: {type(e).__name__}: {str(e)}")
            self._debug_log(f"[文件扫描线程] 堆栈:\n{traceback.format_exc()}")
            self.scan_complete.set()
    
    def _rate_worker_thread(self, raw_dict: Dict[str, str]):
        """
        评分工作线程：
        从评分队列中取任务，当有空闲的计算资源时，启动新的评分Job
        """
        self._debug_log("[评分工作线程] 开始")
        try:
            self._log(f"🤖 开始评分工作线程...")
            gpu_available = bool(self.gpu_workers and self.gpu_executor)
            cpu_rate_enabled = bool(
                self.cpu_rate_workers
                and self.cpu_rate_executor
                and (self.cpu_rate_assist_enabled or not gpu_available)
            )
            loop_count = 0
            while True:
                loop_count += 1
                if loop_count % 1000 == 0:
                    self._debug_log(f"[评分工作线程] 循环 {loop_count} 次")
                # 检查已完成的任务（持续检查，不阻塞）
                completed_futures = []
                with self.futures_lock:
                    for future in list(self.active_rate_futures.keys()):
                        if future.done():
                            completed_futures.append(future)
                
                for future in completed_futures:
                    with self.futures_lock:
                        job_file_infos_done, device_kind, worker_index = self.active_rate_futures.pop(future)
                    if device_kind == "gpu" and worker_index is not None:
                        self._release_gpu_worker(worker_index)
                    try:
                        result = future.result()
                        self._handle_rate_results(result)
                    except Exception as e:
                        self._log(f"评分任务异常: {self._describe_job_infos(job_file_infos_done)} - {e}", "error")
                        with self.stats_lock:
                            self.stats['rate_failed'] += len(job_file_infos_done)
                
                # 检查是否有新任务且有空闲资源
                active_cpu = 0
                active_gpu = 0
                with self.futures_lock:
                    for job_infos, device_kind, _ in self.active_rate_futures.values():
                        if device_kind == "gpu":
                            active_gpu += 1
                        elif device_kind == "cpu":
                            active_cpu += len(job_infos)
                    total_workers = (len(self.gpu_workers) if gpu_available else 0)
                    if cpu_rate_enabled:
                        total_workers += self.cpu_rate_worker_count
                    active_count = active_cpu + active_gpu
                    has_idle_resource = active_count < total_workers
                
                if has_idle_resource:
                    try:
                        # 非阻塞获取任务
                        backlog = 0
                        try:
                            backlog = self.rate_job_queue.qsize()
                        except Exception:
                            backlog = 0

                        gpu_idle = gpu_available and self._has_idle_gpu_worker()
                        cpu_idle = cpu_rate_enabled and active_cpu < self.cpu_rate_worker_count
                        cpu_allowed = False
                        if cpu_idle:
                            if not gpu_available:
                                cpu_allowed = True
                            elif self.cpu_rate_assist_enabled and backlog >= self.cpu_rate_backlog_threshold:
                                cpu_allowed = True

                        if gpu_idle:
                            target_device = "gpu"
                        elif cpu_allowed:
                            target_device = "cpu"
                        else:
                            time.sleep(0.01)
                            continue

                        job_file_info = self.rate_job_queue.get_nowait()
                        job_file_infos = [job_file_info]
                        if target_device == "gpu":
                            batch_size = self._resolve_gpu_batch_size()
                            if batch_size > 1:
                                deadline = time.time() + (self.gpu_batch_max_wait_ms / 1000.0)
                                while len(job_file_infos) < batch_size:
                                    try:
                                        job_file_infos.append(self.rate_job_queue.get_nowait())
                                    except queue.Empty:
                                        if self.gpu_batch_max_wait_ms <= 0 or time.time() >= deadline:
                                            break
                                        time.sleep(0.005)
                            if len(job_file_infos) > 1:
                                self._debug_log(f"[评分工作线程] GPU批量: {len(job_file_infos)}")

                        self._debug_log(f"[评分工作线程] 获取任务: {job_file_infos[0].file_prefix}")
                        
                        # 有空闲资源，创建评分任务
                        worker = None
                        executor = None
                        worker_index = None
                        device_kind = target_device

                        if target_device == "gpu":
                            gpu_worker_info = self._get_idle_gpu_worker()
                            if gpu_worker_info is None:
                                for info in reversed(job_file_infos):
                                    self.rate_job_queue.put(info)
                                time.sleep(0.01)
                                continue
                            worker_index, worker = gpu_worker_info
                            executor = self.gpu_executor
                        else:
                            worker = self._get_idle_cpu_rate_worker()
                            executor = self.cpu_rate_executor
                        
                        if worker is None or executor is None:
                            time.sleep(0.01)
                            continue

                        self._debug_log(f"[评分工作线程] 使用worker: {type(worker).__name__}, executor: {type(executor).__name__}")
                        if target_device == "gpu":
                            rate_jobs = [
                                worker.create_rate_job(
                                    job_file_info=info,
                                    photo_processor=self.photo_processor,
                                    raw_dict=raw_dict,
                                )
                                for info in job_file_infos
                            ]
                        else:
                            rate_jobs = [
                                worker.create_rate_job(
                                    job_file_info=job_file_infos[0],
                                    photo_processor=self.photo_processor,
                                    raw_dict=raw_dict,
                                )
                            ]
                        
                        try:
                            if target_device == "gpu":
                                future = executor.submit(worker._run_batch, rate_jobs)
                            else:
                                future = executor.submit(worker._run_job, rate_jobs[0])
                        except Exception as e:
                            if device_kind == "gpu" and worker_index is not None:
                                self._release_gpu_worker(worker_index)
                            self._log(f"评分任务提交失败: {self._describe_job_infos(job_file_infos)} - {e}", "error")
                            continue
                        self._debug_log(f"[评分工作线程] 任务已提交: {self._describe_job_infos(job_file_infos)}")
                        
                        with self.futures_lock:
                            self.active_rate_futures[future] = (job_file_infos, device_kind, worker_index)
                            active_count = len(self.active_rate_futures)
                            self._debug_log(f"[评分工作线程] 活跃任务数: {active_count}")
                            if active_count > 50:
                                self._log(f"⚠️  警告: 活跃任务数过多 ({active_count})，可能存在性能问题", "warning")
                    except queue.Empty:
                        # 队列为空，检查是否应该退出（需要在锁内检查active_rate_futures）
                        if self.scan_complete.is_set() and self.rate_job_queue.empty():
                            with self.futures_lock:
                                # 如果扫描完成且队列为空，等待所有任务完成
                                if not self.active_rate_futures:
                                    self._debug_log("[评分工作线程] 队列为空且无活跃任务，准备退出")
                                    break
                        time.sleep(0.01)  # 避免CPU空转
                else:
                    # 没有空闲资源，等待一下
                    time.sleep(0.01)
            
            # 等待所有正在运行的任务完成
            while True:
                # 在锁内检查是否还有未完成的任务
                with self.futures_lock:
                    if not self.active_rate_futures:
                        break
                    completed_futures = []
                    for future in list(self.active_rate_futures.keys()):
                        if future.done():
                            completed_futures.append(future)
                
                for future in completed_futures:
                    with self.futures_lock:
                        job_file_infos_done, device_kind, worker_index = self.active_rate_futures.pop(future)
                    if device_kind == "gpu" and worker_index is not None:
                        self._release_gpu_worker(worker_index)
                    try:
                        result = future.result()
                        self._handle_rate_results(result)
                    except Exception as e:
                        self._log(f"评分任务异常: {self._describe_job_infos(job_file_infos_done)} - {e}", "error")
                        with self.stats_lock:
                            self.stats['rate_failed'] += len(job_file_infos_done)
                
                if not completed_futures:
                    time.sleep(0.01)
            
            self._log(f"✅ 评分完成: 成功 {self.stats['rate_success']}, 失败 {self.stats['rate_failed']}")
            self.rate_complete.set()
            self._debug_log("[评分工作线程] 正常结束")
            self._debug_log(f"[评分工作线程] 最终统计 - 成功: {self.stats['rate_success']}, 失败: {self.stats['rate_failed']}")
        except Exception as e:
            import traceback
            self._log(f"评分工作线程异常: {e}", "error")
            self._debug_log(f"[评分工作线程] 异常: {type(e).__name__}: {str(e)}")
            self._debug_log(f"[评分工作线程] 堆栈:\n{traceback.format_exc()}")
            self.rate_complete.set()
    
    def _exif_write_thread(self, raw_dict: Dict[str, str]):
        """
        EXIF写入线程：
        从rating_info_queue中取任务，创建EXIF写入任务并执行
        """
        self._debug_log("[EXIF写入线程] 开始")
        try:
            self._log(f"📝 开始EXIF写入线程...")
            loop_count = 0
            while True:
                loop_count += 1
                if loop_count % 1000 == 0:
                    self._debug_log(f"[EXIF写入线程] 循环 {loop_count} 次")
                # 检查已完成的任务（持续检查，不阻塞）
                completed_futures = []
                with self.futures_lock:
                    for future in list(self.active_exif_futures):
                        if future.done():
                            completed_futures.append(future)
                
                for future in completed_futures:
                    with self.futures_lock:
                        self.active_exif_futures.remove(future)
                    try:
                        result = future.result()
                        with self.stats_lock:
                            if result:
                                # 检查结果，即使部分失败也算部分成功
                                if result.get('success'):
                                    self.stats['exif_success'] += 1
                                elif result.get('error'):
                                    # 有错误信息，但可能部分成功
                                    error_msg = result.get('error', '')
                                    if '部分' in error_msg or '部分EXIF写入失败' in error_msg:
                                        # 部分成功，仍然计数为成功
                                        self.stats['exif_success'] += 1
                                        self._debug_log(f"[EXIF写入线程] 部分成功: {error_msg}")
                                    else:
                                        self.stats['exif_failed'] += 1
                                else:
                                    self.stats['exif_failed'] += 1
                            else:
                                self.stats['exif_failed'] += 1
                    except Exception as e:
                        self._log(f"EXIF写入任务异常: {e}", "error")
                        self._debug_log(f"[EXIF写入线程] 任务异常详情: {type(e).__name__}: {str(e)}")
                        with self.stats_lock:
                            self.stats['exif_failed'] += 1
                
                # 检查是否有新任务
                try:
                    with self.futures_lock:
                        if len(self.active_exif_futures) >= self.max_exif_in_flight:
                            time.sleep(0.01)
                            continue
                    rating_info = self.rating_info_queue._queue.get_nowait()
                    self._debug_log(f"[EXIF写入线程] 获取任务: {rating_info.job_file_info.file_prefix}")
                    
                    # 创建EXIF写入任务
                    exif_job = JobBaseCPU_WriteEXIF(
                        job_file_info=rating_info.job_file_info,
                        exif_data=rating_info.exif_data,
                        raw_dict=raw_dict,
                        dir_path=self.dir_path,
                    )
                    
                    worker = self._get_idle_cpu_io_worker()
                    if worker is None:
                        worker = CPUJobWorker(log_callback=self.log_callback, device="cpu")
                    future = self.cpu_io_executor.submit(worker._run_job, exif_job)
                    self._debug_log(f"[EXIF写入线程] 任务已提交: {rating_info.job_file_info.file_prefix}")
                    
                    with self.futures_lock:
                        self.active_exif_futures.append(future)
                        self._debug_log(f"[EXIF写入线程] 活跃任务数: {len(self.active_exif_futures)}")
                except queue.Empty:
                    # 队列为空，检查是否应该退出（需要在锁内检查active_exif_futures）
                    if self.rate_complete.is_set() and self.rating_info_queue.empty():
                        with self.futures_lock:
                            # 如果评分完成且队列为空，等待所有任务完成
                            if not self.active_exif_futures:
                                self._debug_log("[EXIF写入线程] 队列为空且无活跃任务，准备退出")
                                break
                    time.sleep(0.01)  # 避免CPU空转
            
            # 等待所有正在运行的任务完成
            while True:
                # 在锁内检查是否还有未完成的任务
                with self.futures_lock:
                    if not self.active_exif_futures:
                        break
                    completed_futures = []
                    for future in list(self.active_exif_futures):
                        if future.done():
                            completed_futures.append(future)
                
                for future in completed_futures:
                    with self.futures_lock:
                        self.active_exif_futures.remove(future)
                    try:
                        result = future.result()
                        with self.stats_lock:
                            if result:
                                # 检查结果，即使部分失败也算部分成功
                                if result.get('success'):
                                    self.stats['exif_success'] += 1
                                elif result.get('error'):
                                    # 有错误信息，但可能部分成功
                                    error_msg = result.get('error', '')
                                    if '部分' in error_msg or '部分EXIF写入失败' in error_msg:
                                        # 部分成功，仍然计数为成功
                                        self.stats['exif_success'] += 1
                                        self._debug_log(f"[EXIF写入线程] 部分成功: {error_msg}")
                                    else:
                                        self.stats['exif_failed'] += 1
                                else:
                                    self.stats['exif_failed'] += 1
                            else:
                                self.stats['exif_failed'] += 1
                    except Exception as e:
                        self._log(f"EXIF写入任务异常: {e}", "error")
                        self._debug_log(f"[EXIF写入线程] 任务异常详情: {type(e).__name__}: {str(e)}")
                        with self.stats_lock:
                            self.stats['exif_failed'] += 1
                
                if not completed_futures:
                    time.sleep(0.01)
            
            self._log(f"✅ EXIF写入完成: 成功 {self.stats['exif_success']}, 失败 {self.stats['exif_failed']}")
            self.exif_complete.set()
            self._debug_log("[EXIF写入线程] 正常结束")
            self._debug_log(f"[EXIF写入线程] 最终统计 - 成功: {self.stats['exif_success']}, 失败: {self.stats['exif_failed']}")
        except Exception as e:
            import traceback
            self._log(f"EXIF写入线程异常: {e}", "error")
            self._debug_log(f"[EXIF写入线程] 异常: {type(e).__name__}: {str(e)}")
            self._debug_log(f"[EXIF写入线程] 堆栈:\n{traceback.format_exc()}")
            self.exif_complete.set()
    
    def run(self):
        """
        运行完整工作流程（阻塞式）：
        1. 启动文件扫描线程，将JobFileInfo送到评分Job队列
        2. 启动评分Job线程，等有新的评分Job及有空闲的计算资源就启动新的评分Job
        3. 启动EXIF写入线程，评分Job完成后将结果送入写入exif线程
        4. 等待所有任务都结束，输出统计信息
        """
        start_time = time.time()
        self._debug_log("=== JobManager.run() 开始 ===")
        
        try:
            self._debug_log("步骤1: 重置线程控制标志")
            # 重置线程控制标志
            self.scan_complete.clear()
            self.rate_complete.clear()
            self.exif_complete.clear()

            self._reset_stats()
            self._reset_rating_results()
            with self.stats_lock:
                self.stats['start_time'] = start_time
            
            self._debug_log("步骤2: 清空队列和任务跟踪")
            # 清空队列和任务跟踪
            queue_count = 0
            while not self.rate_job_queue.empty():
                try:
                    self.rate_job_queue.get_nowait()
                    queue_count += 1
                except queue.Empty:
                    break
            if queue_count > 0:
                self._debug_log(f"清空了 {queue_count} 个队列项")

            rating_queue_count = self._clear_rating_queue()
            if rating_queue_count > 0:
                self._debug_log(f"Cleared {rating_queue_count} rating queue items")
            
            with self.futures_lock:
                futures_count = len(self.active_rate_futures) + len(self.active_convert_futures) + len(self.active_exif_futures)
                self.active_rate_futures.clear()
                self.active_convert_futures.clear()
                self.active_exif_futures.clear()
                if futures_count > 0:
                    self._debug_log(f"清空了 {futures_count} 个未来任务")
            with self.worker_index_lock:
                self.busy_gpu_workers.clear()
            
            self._debug_log("步骤3: 创建临时目录")
            # 临时目录
            temp_dir = os.path.join(self.dir_path, '.superpicky', 'temp_jpg')
            os.makedirs(temp_dir, exist_ok=True)
            self._debug_log(f"临时目录: {temp_dir}")
            
            self._debug_log("步骤4: 扫描文件")
            # 扫描文件获取raw_dict和job_file_info_list（需要在所有线程中使用）
            raw_dict, job_file_info_list = self._scan_files()
            self._debug_log(f"扫描完成: {len(job_file_info_list)} 个文件, {len(raw_dict)} 个RAW文件")
            
            if not job_file_info_list:
                self._log("没有文件需要处理")
                self._debug_log("=== JobManager.run() 结束（无文件） ===")
                end_time = time.time()
                with self.stats_lock:
                    self.stats['end_time'] = end_time
                    self.stats['total_time'] = end_time - start_time
                    self.stats['avg_time'] = 0
                return {
                    'stats': self.stats.copy(),
                    'total_time': self.stats['total_time'],
                    'file_ratings': {},
                    'star_3_photos': [],
                }
            
            self._debug_log("步骤5: 启动工作线程")
            # 启动文件扫描线程
            scan_thread = threading.Thread(
                target=self._file_scan_thread,
                args=(job_file_info_list, temp_dir),
                daemon=False
            )
            scan_thread.start()
            self._debug_log(f"文件扫描线程已启动 (ID: {scan_thread.ident})")
            
            # 启动评分工作线程
            rate_thread = threading.Thread(
                target=self._rate_worker_thread,
                args=(raw_dict,),
                daemon=False
            )
            rate_thread.start()
            self._debug_log(f"评分工作线程已启动 (ID: {rate_thread.ident})")
            
            # 启动EXIF写入线程
            exif_thread = threading.Thread(
                target=self._exif_write_thread,
                args=(raw_dict,),
                daemon=False
            )
            exif_thread.start()
            self._debug_log(f"EXIF写入线程已启动 (ID: {exif_thread.ident})")
            
            self._debug_log("步骤6: 等待所有线程完成")
            # 等待所有线程完成
            self._debug_log("等待文件扫描线程...")
            scan_thread.join()
            self._debug_log("文件扫描线程已完成")
            
            self._debug_log("等待评分工作线程...")
            rate_thread.join()
            self._debug_log("评分工作线程已完成")
            
            self._debug_log("等待EXIF写入线程...")
            exif_thread.join()
            self._debug_log("EXIF写入线程已完成")
            
            self._debug_log("步骤7: 收集file_ratings和star_3_photos")
            with self.rating_results_lock:
                file_ratings = dict(self.file_ratings)
                star_3_photos = list(self.star_3_photos)
            self._debug_log(f"收集完成: {len(file_ratings)} 个file_ratings, {len(star_3_photos)} 个star_3_photos")
            
            # 输出统计信息
            end_time = time.time()
            total_time = end_time - start_time
            with self.stats_lock:
                self.stats['end_time'] = end_time
                self.stats['total_time'] = total_time
                total_count = self.stats.get('total', 0)
                self.stats['avg_time'] = total_time / total_count if total_count > 0 else 0
            self._log(f"\n⏱️  总耗时: {total_time:.1f}秒")
            self._log(f"📊 统计: 转换({self.stats['convert_success']}/{self.stats['convert_failed']}), "
                      f"评分({self.stats['rate_success']}/{self.stats['rate_failed']}), "
                      f"EXIF({self.stats['exif_success']}/{self.stats['exif_failed']})")
            
            self._debug_log("步骤9: 释放workers")
            # 释放workers
            self._debug_log("开始关闭CPU评分线程池...")
            try:
                if self.cpu_rate_executor is not None:
                    self.cpu_rate_executor.shutdown(wait=True)
                    self._debug_log("CPU评分线程池已关闭")
                else:
                    self._debug_log("CPU评分线程池不存在，跳过")
            except Exception as e:
                self._debug_log(f"关闭CPU评分线程池时出错: {e}", "error")

            self._debug_log("开始关闭CPU IO线程池...")
            try:
                if self.cpu_io_executor is not None:
                    self.cpu_io_executor.shutdown(wait=True)
                    self._debug_log("CPU IO线程池已关闭")
                else:
                    self._debug_log("CPU IO线程池不存在，跳过")
            except Exception as e:
                self._debug_log(f"关闭CPU IO线程池时出错: {e}", "error")
            
            self._debug_log("开始关闭GPU线程池...")
            try:
                if self.gpu_executor is not None:
                    self.gpu_executor.shutdown(wait=True)
                    self._debug_log("GPU线程池已关闭")
                else:
                    self._debug_log("GPU线程池不存在，跳过")
            except Exception as e:
                self._debug_log(f"关闭GPU线程池时出错: {e}", "error")
            
            self._debug_log("Workers已释放")
            self._debug_log(f"最终统计: {self.stats}")
            self._debug_log(f"返回结果: file_ratings={len(file_ratings)}, star_3_photos={len(star_3_photos)}")
            self._debug_log("=== JobManager.run() 正常结束 ===")
            return {
                'stats': self.stats.copy(),
                'total_time': total_time,
                'file_ratings': file_ratings,
                'star_3_photos': star_3_photos,
            }
        except Exception as e:
            # 捕获所有异常，记录日志并确保资源清理
            import traceback
            error_msg = f"JobManager运行异常: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg, "error")
            self._debug_log(f"异常类型: {type(e).__name__}")
            self._debug_log(f"异常消息: {str(e)}")
            self._debug_log(f"异常堆栈:\n{traceback.format_exc()}")
            # 确保设置完成标志，避免线程无限等待
            self._debug_log("设置完成标志以避免线程无限等待")
            self.scan_complete.set()
            self.rate_complete.set()
            self.exif_complete.set()
            end_time = time.time()
            with self.stats_lock:
                self.stats['end_time'] = end_time
                self.stats['total_time'] = end_time - start_time
                total_count = self.stats.get('total', 0)
                self.stats['avg_time'] = self.stats['total_time'] / total_count if total_count > 0 else 0
            # 释放workers
            self._debug_log("尝试释放workers")
            try:
                self.shutdown(wait=True)
                self._debug_log("Workers释放成功")
            except Exception as shutdown_error:
                self._debug_log(f"Workers释放失败: {shutdown_error}", "error")
            # 返回错误结果
            self._debug_log("=== JobManager.run() 异常结束 ===")
            return {
                'stats': self.stats.copy(),
                'total_time': self.stats['total_time'],
                'file_ratings': {},
                'star_3_photos': [],
                'error': str(e),
            }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.stats_lock:
            return self.stats.copy()

    def shutdown(self, wait: bool = True):
        """关闭线程池（可选调用）"""
        self._debug_log(f"shutdown() 被调用，wait={wait}")
        try:
            self._debug_log("关闭CPU评分线程池...")
            if self.cpu_rate_executor is not None:
                self.cpu_rate_executor.shutdown(wait=wait)
                self._debug_log("CPU评分线程池已关闭")
            else:
                self._debug_log("CPU评分线程池不存在，跳过")
        except Exception as e:
            self._debug_log(f"关闭CPU评分线程池时出错: {e}", "error")
        try:
            self._debug_log("关闭CPU IO线程池...")
            if self.cpu_io_executor is not None:
                self.cpu_io_executor.shutdown(wait=wait)
                self._debug_log("CPU IO线程池已关闭")
            else:
                self._debug_log("CPU IO线程池不存在，跳过")
        except Exception as e:
            self._debug_log(f"关闭CPU IO线程池时出错: {e}", "error")
        try:
            if self.gpu_executor is not None:
                self._debug_log("关闭GPU线程池...")
                self.gpu_executor.shutdown(wait=wait)
                self._debug_log("GPU线程池已关闭")
            else:
                self._debug_log("GPU线程池不存在，跳过")
        except Exception as e:
            self._debug_log(f"关闭GPU线程池时出错: {e}", "error")
        with self.worker_index_lock:
            self.busy_gpu_workers.clear()
        self._debug_log("shutdown() 完成")
