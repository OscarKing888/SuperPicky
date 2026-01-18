# -*- coding: utf-8 -*-

import os
import queue
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Optional, Callable, Any, Dict, List, Tuple

import torch

from constants import RAW_EXTENSIONS, JPG_EXTENSIONS
from core.job_base import JobFileInfo
from core.job_base_cpu_convert_heif import JobBaseCPU_ConvertHEIF
from core.job_base_cpu_rate import JobBaseCPU_Rate
from core.job_base_gpu_rate import JobBaseGPU_Rate
from core.job_base_cpu_write_exif import JobBaseCPU_WriteEXIF
from core.rating_info import RatingInfo
from core.photo_processor import PhotoProcessor

from core.job_manager_worker_cpu import CPUJobWorker
from core.job_manager_worker_gpu import GPUJobWorker


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
        cpu_count = cpu_worker_count or min(4, multiprocessing.cpu_count())
        
        # 创建CPU workers列表（每个worker一个实例，用于负载均衡）
        self.cpu_workers: List[CPUJobWorker] = [
            CPUJobWorker(log_callback=log_callback, device="cpu")
            for _ in range(cpu_count)
        ]
        self.cpu_worker_index = 0  # 轮询索引
        
        # 创建CPU线程池（由JobManager管理）
        self.cpu_executor = ThreadPoolExecutor(max_workers=cpu_count)
        
        # 检测GPU设备
        gpu_device_str = None
        if torch.backends.mps.is_available():
            gpu_device_str = "mps"
        elif torch.cuda.is_available():
            gpu_device_str = "cuda"
        
        # 创建GPU worker和线程池（如果GPU可用）
        if gpu_device_str is not None:
            gpu_count = gpu_worker_count or self._calculate_gpu_workers(gpu_device_str)
            self.gpu_workers: List[GPUJobWorker] = [
                GPUJobWorker(log_callback=log_callback, device=gpu_device_str)
                for _ in range(gpu_count)
            ]
            self.gpu_worker_index = 0  # 轮询索引
            self.gpu_executor = ThreadPoolExecutor(max_workers=gpu_count)
        else:
            self.gpu_workers = []
            self.gpu_executor = None
        
        # 评星信息队列（步骤4：评星完成后保存到这里）
        self.rating_info_queue: queue.Queue[RatingInfo] = queue.Queue()
        self.rating_info_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'convert_success': 0,
            'convert_failed': 0,
            'rate_success': 0,
            'rate_failed': 0,
            'exif_success': 0,
            'exif_failed': 0,
        }
        self.stats_lock = threading.Lock()
        
        # 跟踪正在运行的任务（用于判断worker是否空闲）
        self.active_futures: List[Future] = []
        self.futures_lock = threading.Lock()

    def _get_idle_cpu_worker(self) -> CPUJobWorker:
        """获取空闲的CPU worker（轮询方式）"""
        worker = self.cpu_workers[self.cpu_worker_index]
        self.cpu_worker_index = (self.cpu_worker_index + 1) % len(self.cpu_workers)
        return worker
    
    def _get_idle_gpu_worker(self) -> Optional[GPUJobWorker]:
        """获取空闲的GPU worker（轮询方式）"""
        if not self.gpu_workers:
            return None
        worker = self.gpu_workers[self.gpu_worker_index]
        self.gpu_worker_index = (self.gpu_worker_index + 1) % len(self.gpu_workers)
        return worker
    
    def _get_idle_worker_for_rate(self) -> Tuple[Any, ThreadPoolExecutor]:
        """为评分任务选择worker和executor（自动选择CPU或GPU）"""
        # 优先使用GPU（如果可用）
        if self.gpu_workers and self.gpu_executor:
            return self._get_idle_gpu_worker(), self.gpu_executor
        return self._get_idle_cpu_worker(), self.cpu_executor

    def _calculate_gpu_workers(self, device_str: str) -> int:
        """根据可用显存计算GPU并发数"""
        try:
            if device_str == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_memory_bytes, _ = torch.cuda.mem_get_info(0)
                free_memory_gb = free_memory_bytes / (1024 ** 3)
                gpu_workers = int(free_memory_gb / 2 - 2)
                return max(1, gpu_workers)
        except Exception:
            pass
        return 1

    def _log(self, msg: str, level: str = "info"):
        """内部日志方法"""
        if self.log_callback:
            self.log_callback(msg, level)
    
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
    
    def _on_rate_complete(self, result: Dict[str, Any]):
        """
        步骤4：评星任务完成回调，保存评星信息到队列
        """
        if not result:
            return
        
        job_file_info = result.get('job_file_info')
        if not job_file_info:
            return
        
        # 构建EXIF数据
        exif_data = {
            'rating': result.get('rating', 0),
            'pick': result.get('pick', 0),
            'reason': result.get('reason', ''),
            'confidence': result.get('confidence', 0.0),
            'head_sharpness': result.get('head_sharpness', 0.0),
            'topiq': result.get('topiq'),
            'adj_sharpness': result.get('adj_sharpness'),
            'adj_topiq': result.get('adj_topiq'),
            'is_flying': result.get('is_flying', False),
            'focus_status': result.get('focus_status'),
            'focus_sharpness_weight': result.get('focus_sharpness_weight', 1.0),
            'focus_topiq_weight': result.get('focus_topiq_weight', 1.0),
            'best_eye_visibility': result.get('best_eye_visibility', 0.0),
        }
        
        rating_info = RatingInfo(
            job_file_info=job_file_info,
            rating=result.get('rating', 0),
            pick=result.get('pick', 0),
            reason=result.get('reason', ''),
            exif_data=exif_data,
        )
        
        with self.rating_info_lock:
            self.rating_info_queue.put(rating_info)

    def run(self):
        """
        运行完整工作流程：
        1. 扫描文件生成JobFileInfo列表
        2. 根据use_tmp_file创建转换任务或评分任务
        3. 评星完成后保存到队列
        4. 所有rate完成后执行EXIF写入
        5. 输出统计并释放workers
        """
        start_time = time.time()
        
        # 步骤1：扫描文件
        raw_dict, job_file_info_list = self._scan_files()
        
        if not job_file_info_list:
            self._log("没有文件需要处理")
            return {
                'stats': self.stats.copy(),
                'total_time': time.time() - start_time,
            }
        
        # 步骤2和3：处理转换和评分任务
        convert_futures: Dict[Future, JobFileInfo] = {}
        rate_futures: Dict[Future, JobFileInfo] = {}
        
        # 临时目录
        temp_dir = os.path.join(self.dir_path, '.superpicky', 'temp_jpg')
        os.makedirs(temp_dir, exist_ok=True)
        
        for job_file_info in job_file_info_list:
            if job_file_info.needs_tmp_file():
                # 步骤2：需要转换HEIF，创建转换任务
                convert_job = JobBaseCPU_ConvertHEIF(
                    job_file_info=job_file_info,
                    output_dir=temp_dir,
                )
                worker = self._get_idle_cpu_worker()
                future = self.cpu_executor.submit(worker.run_job, convert_job)
                convert_futures[future] = job_file_info
            else:
                # 步骤3：直接评分，从空闲worker中选择（自动CPU/GPU）
                worker, executor = self._get_idle_worker_for_rate()
                rate_job = worker.create_rate_job(
                    job_file_info=job_file_info,
                    photo_processor=self.photo_processor,
                    raw_dict=raw_dict,
                )
                future = executor.submit(worker.run_job, rate_job)
                rate_futures[future] = job_file_info
        
        # 等待转换任务完成，完成后创建评分任务
        self._log(f"🔄 开始转换 {len(convert_futures)} 个HEIF文件...")
        for future in as_completed(convert_futures):
            job_file_info = convert_futures[future]
            try:
                result = future.result()
                if result and result.get('success'):
                    # 转换成功，更新job_file_info的tmp_file_path
                    job_file_info.tmp_file_path = result.get('temp_jpg_path')
                    # 创建评分任务
                    worker, executor = self._get_idle_worker_for_rate()
                    rate_job = worker.create_rate_job(
                        job_file_info=job_file_info,
                        photo_processor=self.photo_processor,
                        raw_dict=raw_dict,
                    )
                    rate_future = executor.submit(worker.run_job, rate_job)
                    rate_futures[rate_future] = job_file_info
                    
                    with self.stats_lock:
                        self.stats['convert_success'] += 1
                else:
                    with self.stats_lock:
                        self.stats['convert_failed'] += 1
            except Exception as e:
                self._log(f"转换任务异常: {job_file_info.file_prefix} - {e}", "error")
                with self.stats_lock:
                    self.stats['convert_failed'] += 1
        
        self._log(f"✅ HEIF转换完成: 成功 {self.stats['convert_success']}, 失败 {self.stats['convert_failed']}")
        
        # 等待所有评分任务完成
        self._log(f"🤖 开始评分 {len(rate_futures)} 个文件...")
        for future in as_completed(rate_futures):
            job_file_info = rate_futures[future]
            try:
                result = future.result()
                if result:
                    # 步骤4：评星完成，保存到队列
                    self._on_rate_complete(result)
                    with self.stats_lock:
                        if result.get('rating', -1) >= 0:
                            self.stats['rate_success'] += 1
                        else:
                            self.stats['rate_failed'] += 1
            except Exception as e:
                self._log(f"评分任务异常: {job_file_info.file_prefix} - {e}", "error")
                with self.stats_lock:
                    self.stats['rate_failed'] += 1
        
        self._log(f"✅ 评分完成: 成功 {self.stats['rate_success']}, 失败 {self.stats['rate_failed']}")
        
        # 步骤5：所有rate完成后，执行EXIF写入
        self._log(f"📝 开始写入EXIF {self.rating_info_queue.qsize()} 个文件...")
        exif_futures: List[Future] = []
        
        while not self.rating_info_queue.empty():
            try:
                rating_info = self.rating_info_queue.get_nowait()
            except queue.Empty:
                break
            
            # 创建EXIF写入任务
            exif_job = JobBaseCPU_WriteEXIF(
                job_file_info=rating_info.job_file_info,
                exif_data=rating_info.exif_data,
                raw_dict=raw_dict,
                dir_path=self.dir_path,
            )
            
            worker = self._get_idle_cpu_worker()
            future = self.cpu_executor.submit(worker.run_job, exif_job)
            exif_futures.append(future)
        
        # 等待所有EXIF写入完成
        for future in as_completed(exif_futures):
            try:
                result = future.result()
                with self.stats_lock:
                    if result and result.get('success'):
                        self.stats['exif_success'] += 1
                    else:
                        self.stats['exif_failed'] += 1
            except Exception as e:
                self._log(f"EXIF写入任务异常: {e}", "error")
                with self.stats_lock:
                    self.stats['exif_failed'] += 1
        
        self._log(f"✅ EXIF写入完成: 成功 {self.stats['exif_success']}, 失败 {self.stats['exif_failed']}")
        
        # 步骤6：输出统计信息
        total_time = time.time() - start_time
        self._log(f"\n⏱️  总耗时: {total_time:.1f}秒")
        self._log(f"📊 统计: 转换({self.stats['convert_success']}/{self.stats['convert_failed']}), "
                  f"评分({self.stats['rate_success']}/{self.stats['rate_failed']}), "
                  f"EXIF({self.stats['exif_success']}/{self.stats['exif_failed']})")
        
        # 释放workers
        self.shutdown(wait=True)
        
        return {
            'stats': self.stats.copy(),
            'total_time': total_time,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.stats_lock:
            return self.stats.copy()

    def shutdown(self, wait: bool = True):
        """关闭线程池（可选调用）"""
        try:
            self.cpu_executor.shutdown(wait=wait)
        except Exception:
            pass
        try:
            if self.gpu_executor is not None:
                self.gpu_executor.shutdown(wait=wait)
        except Exception:
            pass