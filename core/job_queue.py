#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发Job队列框架
支持流水线式处理、动态并发控制、多设备并行

核心设计：
- Job: 基础任务单元
- JobQueue: 线程安全的任务队列
- PipelineStage: 流水线阶段抽象
- Pipeline: 流水线管理器
- DeviceManager: 设备管理器（支持CPU+GPU同时工作）
"""

import os
import time
import queue
import threading
from typing import Optional, Callable, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class JobStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """基础任务单元"""
    job_id: str
    data: Any
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def start(self):
        """标记任务开始"""
        self.status = JobStatus.RUNNING
        self.started_at = time.time()
    
    def complete(self, result: Any = None):
        """标记任务完成"""
        self.status = JobStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()
    
    def fail(self, error: str):
        """标记任务失败"""
        self.status = JobStatus.FAILED
        self.error = error
        self.completed_at = time.time()
    
    def cancel(self):
        """取消任务"""
        self.status = JobStatus.CANCELLED
        self.completed_at = time.time()
    
    @property
    def duration(self) -> float:
        """任务耗时（秒）"""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return 0.0


class JobQueue:
    """线程安全的任务队列"""
    
    def __init__(self, maxsize: int = 0):
        """
        初始化任务队列
        
        Args:
            maxsize: 队列最大容量（0=无限制）
        """
        self._queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._stats = {
            'total_put': 0,
            'total_get': 0,
            'total_done': 0,
        }
    
    def put(self, job: Job, block: bool = True, timeout: Optional[float] = None):
        """添加任务到队列"""
        self._queue.put(job, block=block, timeout=timeout)
        with self._lock:
            self._stats['total_put'] += 1
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Job]:
        """从队列获取任务"""
        try:
            job = self._queue.get(block=block, timeout=timeout)
            with self._lock:
                self._stats['total_get'] += 1
            return job
        except queue.Empty:
            return None
    
    def task_done(self):
        """标记任务完成"""
        self._queue.task_done()
        with self._lock:
            self._stats['total_done'] += 1
    
    def join(self):
        """等待所有任务完成"""
        self._queue.join()
    
    def qsize(self) -> int:
        """获取队列大小"""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """队列是否为空"""
        return self._queue.empty()
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        with self._lock:
            return self._stats.copy()


class PipelineStage(ABC):
    """流水线阶段抽象基类"""
    
    def __init__(
        self,
        name: str,
        input_queue: JobQueue,
        output_queue: Optional[JobQueue] = None,
        max_workers: int = 1,
        device: str = 'cpu',
        log_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        初始化流水线阶段
        
        Args:
            name: 阶段名称
            input_queue: 输入队列
            output_queue: 输出队列（None表示最终阶段）
            max_workers: 最大并发工作线程数
            device: 计算设备
            log_callback: 日志回调函数
        """
        self.name = name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.max_workers = max_workers
        self.device = device
        self.log_callback = log_callback
        
        self._workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._stats = {
            'processed': 0,
            'failed': 0,
            'total_time': 0.0,
        }
        self._stats_lock = threading.Lock()
    
    def _log(self, msg: str, level: str = "info"):
        """内部日志方法"""
        if self.log_callback:
            self.log_callback(f"[{self.name}] {msg}", level)
    
    @abstractmethod
    def process_job(self, job: Job) -> Any:
        """
        处理单个任务（子类实现）
        
        Args:
            job: 待处理的任务
            
        Returns:
            处理结果
        """
        pass
    
    def _worker(self):
        """工作线程主循环"""
        while not self._stop_event.is_set():
            job = self.input_queue.get(timeout=0.1)
            if job is None:
                continue
            
            # 检查是否已取消
            if self._stop_event.is_set():
                # 将未处理的任务标记为取消
                if job.status == JobStatus.PENDING:
                    job.cancel()
                self.input_queue.task_done()
                break
            
            if job.status == JobStatus.CANCELLED:
                self.input_queue.task_done()
                continue
            
            try:
                job.start()
                result = self.process_job(job)
                if isinstance(result, dict):
                    if isinstance(job.data, dict):
                        job.data = {**job.data, **result}
                    else:
                        job.data = result
                job.complete(result)
                
                # 如果有输出队列，将结果传递给下一阶段
                if self.output_queue is not None:
                    self.output_queue.put(job)
                
                # 更新统计
                with self._stats_lock:
                    self._stats['processed'] += 1
                    self._stats['total_time'] += job.duration
                
            except Exception as e:
                job.fail(str(e))
                with self._stats_lock:
                    self._stats['failed'] += 1
                self._log(f"任务失败: {job.job_id} - {e}", "error")
            
            finally:
                # 清理 GPU 显存缓存（如果使用 GPU）
                # 这有助于释放推理过程中分配的显存
                try:
                    import torch
                    if torch.cuda.is_available() and self.device in ('cuda', 'all'):
                        torch.cuda.empty_cache()
                except Exception:
                    pass  # 静默失败，不影响主流程
                
                self.input_queue.task_done()
    
    def start(self):
        """启动工作线程"""
        self._stop_event.clear()
        self._workers = []
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker,
                name=f"{self.name}-Worker-{i+1}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
        self._log(f"启动 {self.max_workers} 个工作线程")
    
    def stop(self, timeout: float = 5.0):
        """停止工作线程"""
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=timeout)
        self._workers = []
        
        # 清理 GPU 显存缓存（如果使用 GPU）
        # 在阶段停止时清理，释放所有模型占用的显存
        try:
            import torch
            if torch.cuda.is_available() and self.device in ('cuda', 'all'):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 确保所有 CUDA 操作完成
        except Exception:
            pass  # 静默失败，不影响主流程
        
        self._log("工作线程已停止")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._stats_lock:
            stats = self._stats.copy()
            if stats['processed'] > 0:
                stats['avg_time'] = stats['total_time'] / stats['processed']
            else:
                stats['avg_time'] = 0.0
            return stats


class Pipeline:
    """流水线管理器"""
    
    def __init__(
        self,
        stages: List[PipelineStage],
        log_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        初始化流水线
        
        Args:
            stages: 流水线阶段列表
            log_callback: 日志回调函数
        """
        self.stages = stages
        self.log_callback = log_callback
        self._running = False
    
    def _log(self, msg: str, level: str = "info"):
        """内部日志方法"""
        if self.log_callback:
            self.log_callback(msg, level)
    
    def start(self):
        """启动所有阶段"""
        self._running = True
        for stage in self.stages:
            stage.start()
        self._log("流水线已启动")
    
    def stop(self, timeout: float = 10.0, skip_log: bool = False):
        """停止所有阶段"""
        self._running = False
        if not skip_log:
            self._log("正在停止流水线...")
        
        # 停止额外的线程（如队列合并器）
        if hasattr(self, '_merger'):
            self._merger.running = False
        
        for stage in reversed(self.stages):  # 从后往前停止
            stage.stop(timeout=timeout)
        if not skip_log:
            self._log("流水线已停止")
    
    def wait_complete(self):
        """等待所有阶段完成（支持中断）"""
        import time
        for stage in self.stages:
            if stage.input_queue:
                # 轮询等待，允许中断检查
                while True:
                    # 检查队列是否为空且所有任务已完成
                    if stage.input_queue.empty() and stage.input_queue.qsize() == 0:
                        # 尝试join，但使用超时以便中断
                        try:
                            # 使用短超时，允许中断检查
                            import queue
                            # 等待队列完成，但允许中断
                            while not stage.input_queue.empty():
                                time.sleep(0.1)
                            stage.input_queue.join()
                            break
                        except:
                            break
                    else:
                        time.sleep(0.1)  # 短暂等待，避免CPU占用过高
        # 立即输出日志，避免感觉卡死
        self._log("所有任务已完成")
        self._log("正在停止流水线...")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取所有阶段的统计信息"""
        return {
            stage.name: stage.get_stats()
            for stage in self.stages
        }


class DeviceManager:
    """设备管理器 - 支持CPU和GPU同时工作"""
    
    def __init__(
        self,
        preferred_device: str = 'auto',
        cpu_threads: int = 0,
        gpu_concurrent: int = 1,
        use_all_devices: bool = False
    ):
        """
        初始化设备管理器
        
        Args:
            preferred_device: 首选设备 ('auto', 'cpu', 'cuda', 'mps', 'all')
            cpu_threads: CPU线程数（0=自动）
            gpu_concurrent: GPU并发数
            use_all_devices: 是否同时使用所有可用设备
        """
        self.preferred_device = preferred_device
        self.cpu_threads = cpu_threads
        self.gpu_concurrent = gpu_concurrent
        self.use_all_devices = use_all_devices or (preferred_device == 'all')
        
        # 检测可用设备
        self.available_devices = self._detect_devices()
        self.active_devices = self._select_active_devices()
    
    def _detect_devices(self) -> Dict[str, bool]:
        """检测可用设备"""
        devices = {'cpu': True, 'cuda': False, 'mps': False}
        
        try:
            import torch
            if torch.cuda.is_available():
                devices['cuda'] = True
            try:
                if torch.backends.mps.is_available():
                    devices['mps'] = True
            except:
                pass
        except ImportError:
            pass
        
        return devices
    
    def _select_active_devices(self) -> List[str]:
        """选择激活的设备"""
        if self.use_all_devices:
            # 使用所有可用设备
            active = []
            if self.available_devices['cpu']:
                active.append('cpu')
            if self.available_devices['cuda']:
                active.append('cuda')
            if self.available_devices['mps']:
                active.append('mps')
            return active if active else ['cpu']
        
        # 单个设备模式
        if self.preferred_device == 'auto':
            if self.available_devices['cuda']:
                return ['cuda']
            elif self.available_devices['mps']:
                return ['mps']
            else:
                return ['cpu']
        
        if self.preferred_device in self.available_devices:
            if self.available_devices[self.preferred_device]:
                return [self.preferred_device]
        
        return ['cpu']  # 降级到CPU
    
    def get_device_config(self, device: str) -> Dict[str, Any]:
        """获取设备配置"""
        if device == 'cpu':
            import multiprocessing
            threads = self.cpu_threads if self.cpu_threads > 0 else multiprocessing.cpu_count()
            return {
                'device': 'cpu',
                'max_workers': threads,
                'concurrent': threads
            }
        elif device in ['cuda', 'mps']:
            return {
                'device': device,
                'max_workers': self.gpu_concurrent,
                'concurrent': self.gpu_concurrent
            }
        else:
            return {
                'device': 'cpu',
                'max_workers': 1,
                'concurrent': 1
            }
    
    def get_all_configs(self) -> List[Dict[str, Any]]:
        """获取所有激活设备的配置"""
        return [self.get_device_config(device) for device in self.active_devices]
    
    def __repr__(self) -> str:
        return f"DeviceManager(active={self.active_devices}, use_all={self.use_all_devices})"


# Job type identifiers used by the parallel pipeline.
JOB_TYPE_HEIF_CONVERT = "heif_convert"
JOB_TYPE_INFER = "inference"


class QueueGroup:
    """Lightweight wrapper to expose group queue stats/join semantics."""

    def __init__(self, queues: List[JobQueue]):
        self._queues = list(queues)

    def empty(self) -> bool:
        return all(queue.empty() for queue in self._queues)

    def qsize(self) -> int:
        return sum(queue.qsize() for queue in self._queues)

    def join(self):
        for queue in self._queues:
            queue.join()


class QueueMonitorStage:
    """No-op stage used to let a pipeline wait on a shared queue."""

    def __init__(
        self,
        name: str,
        input_queue: JobQueue,
        log_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.name = name
        self.input_queue = input_queue
        self.log_callback = log_callback

    def _log(self, msg: str, level: str = "info"):
        if self.log_callback:
            self.log_callback(f"[{self.name}] {msg}", level)

    def start(self):
        self._log("monitor started")

    def stop(self, timeout: float = 0.0):
        self._log("monitor stopped")

    def get_stats(self) -> Dict[str, Any]:
        return {}
