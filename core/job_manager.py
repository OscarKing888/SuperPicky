# -*- coding: utf-8 -*-

import queue
import time
import threading
import multiprocessing
from typing import Optional, Callable, Any, Dict, List, Tuple
from dataclasses import dataclass, field

from core.job_base_cpu_convert_heif import JobBaseCPU_ConvertHEIF
from core.job_base_cpu_rate import JobBaseCPU_Rate
from core.job_base_gpu_rate import JobBaseGPU_Rate
from core.job_base_cpu_write_exif import JobBaseCPU_WriteEXIF

from core.job_manager_worker_cpu import CPUJobWorker
from core.job_manager_worker_gpu import GPUJobWorker


class JobManager:
    """任务管理器 - 管理并执行各种类型的任务"""
    
    def __init__(
        self,
        cpu_workers: Optional[int] = None,
        gpu_workers: Optional[int] = None,
        log_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        初始化任务管理器
        
        Args:
            cpu_workers: CPU工作线程数（None=自动）
            gpu_workers: GPU并发数（None=自动）
            log_callback: 日志回调函数
        """
        cpu_worker_count = cpu_workers or min(4, multiprocessing.cpu_count())

        self.cpu_workers = 


        gpu_worker_count = gpu_workers or self._calculate_gpu_workers()
        self.gpu_workers = 

        
        self.log_callback = log_callback

        
        # 任务队列
        self.convert_jobs: List[JobBaseCPU_ConvertHEIF] = []
        self.rate_jobs_cpu: List[JobBaseCPU_Rate] = []
        self.rate_jobs_gpu: List[JobBaseGPU_Rate] = []
        self.write_exif_jobs: List[JobBaseCPU_WriteEXIF] = []
        
        # 结果队列
        self.convert_results = queue.Queue()
        self.rate_results = queue.Queue()
        self.exif_results = queue.Queue()
        
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

    def _dispatch(self, job):
        """根据job类型选择对应的worker（调度入口）"""
        if isinstance(job, JobBaseGPU_Rate):
            return self.gpu_worker
        return self.cpu_worker

    def _calculate_gpu_workers(self) -> int:
        """根据可用显存计算GPU并发数"""
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
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

    def add_convert_job(self, job: JobBaseCPU_ConvertHEIF):
        """添加HEIF转换任务"""
        self.convert_jobs.append(job)

    def add_rate_job_cpu(self, job: JobBaseCPU_Rate):
        """添加CPU评分任务"""
        self.rate_jobs_cpu.append(job)

    def add_rate_job_gpu(self, job: JobBaseGPU_Rate):
        """添加GPU评分任务"""
        self.rate_jobs_gpu.append(job)

    def add_write_exif_job(self, job: JobBaseCPU_WriteEXIF):
        """添加EXIF写入任务"""
        self.write_exif_jobs.append(job)

    def run_convert_jobs(self) -> List[Dict[str, Any]]:
        """运行HEIF转换任务"""
        if not self.convert_jobs:
            return []
        
        self._log(f"🔄 开始转换 {len(self.convert_jobs)} 个HEIF文件...")
        results = []

        futures = {self._dispatch(job).submit(job): job for job in self.convert_jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                with self.stats_lock:
                    if result and result.get('success'):
                        self.stats['convert_success'] += 1
                    else:
                        self.stats['convert_failed'] += 1
            except Exception as e:
                self._log(f"转换任务异常: {job.job_id} - {e}", "error")
                with self.stats_lock:
                    self.stats['convert_failed'] += 1
        
        self._log(f"✅ HEIF转换完成: 成功 {self.stats['convert_success']}, 失败 {self.stats['convert_failed']}")
        return results

    def run_rate_jobs(self) -> List[Dict[str, Any]]:
        """运行评分任务（CPU和GPU并行）"""
        all_jobs = self.rate_jobs_cpu + self.rate_jobs_gpu
        if not all_jobs:
            return []
        
        self._log(f"🤖 开始评分 {len(all_jobs)} 个文件 (CPU: {len(self.rate_jobs_cpu)}, GPU: {len(self.rate_jobs_gpu)})...")
        results = []

        futures: Dict[Any, Any] = {}
        for job in self.rate_jobs_cpu:
            futures[self._dispatch(job).submit(job)] = job
        for job in self.rate_jobs_gpu:
            futures[self._dispatch(job).submit(job)] = job

        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    with self.stats_lock:
                        if result.get('rating', -1) >= 0:
                            self.stats['rate_success'] += 1
                        else:
                            self.stats['rate_failed'] += 1
            except Exception as e:
                self._log(f"评分任务异常: {job.job_id} - {e}", "error")
                with self.stats_lock:
                    self.stats['rate_failed'] += 1
        
        self._log(f"✅ 评分完成: 成功 {self.stats['rate_success']}, 失败 {self.stats['rate_failed']}")
        return results

    def run_write_exif_jobs(self) -> List[Dict[str, Any]]:
        """运行EXIF写入任务"""
        if not self.write_exif_jobs:
            return []
        
        self._log(f"📝 开始写入EXIF {len(self.write_exif_jobs)} 个文件...")
        results = []

        futures = {self._dispatch(job).submit(job): job for job in self.write_exif_jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                with self.stats_lock:
                    if result and result.get('success'):
                        self.stats['exif_success'] += 1
                    else:
                        self.stats['exif_failed'] += 1
            except Exception as e:
                self._log(f"EXIF写入任务异常: {job.job_id} - {e}", "error")
                with self.stats_lock:
                    self.stats['exif_failed'] += 1
        
        self._log(f"✅ EXIF写入完成: 成功 {self.stats['exif_success']}, 失败 {self.stats['exif_failed']}")
        return results

    def run(self):
        """运行所有任务队列，按顺序执行：转换 → 评分 → EXIF写入"""
        start_time = time.time()
        
        # 阶段1: HEIF转换
        convert_results = self.run_convert_jobs()
        
        # 阶段2: 评分（CPU和GPU并行）
        rate_results = self.run_rate_jobs()
        
        # 阶段3: EXIF写入
        exif_results = self.run_write_exif_jobs()
        
        total_time = time.time() - start_time
        self._log(f"\n⏱️  总耗时: {total_time:.1f}秒")
        self._log(f"📊 统计: 转换({self.stats['convert_success']}/{self.stats['convert_failed']}), "
                  f"评分({self.stats['rate_success']}/{self.stats['rate_failed']}), "
                  f"EXIF({self.stats['exif_success']}/{self.stats['exif_failed']})")
        
        return {
            'convert_results': convert_results,
            'rate_results': rate_results,
            'exif_results': exif_results,
            'stats': self.stats.copy(),
            'total_time': total_time,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.stats_lock:
            return self.stats.copy()

    def shutdown(self, wait: bool = True):
        """关闭worker线程池（可选调用）"""
        try:
            self.cpu_worker.shutdown(wait=wait)
        except Exception:
            pass
        try:
            self.gpu_worker.shutdown(wait=wait)
        except Exception:
            pass