#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流水线构建器
根据配置自动构建最优化的处理流水线
"""

import os
import multiprocessing
from typing import List, Optional, Dict, Any, Callable
from core.job_queue import JobQueue, Pipeline, DeviceManager, Job, JobStatus
from core.pipeline_stages import (
    HEIFConversionStage,
    RAWConversionStage,
    ImageProcessingStage,
    EXIFWriteStage
)


class PipelineBuilder:
    """流水线构建器"""
    
    def __init__(
        self,
        dir_path: str,
        settings: Any,  # ProcessingSettings
        raw_dict: Dict[str, str],
        log_callback: Optional[Callable[[str, str], None]] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        stats_callback: Optional[Callable[[Dict], None]] = None
    ):
        """
        初始化构建器
        
        Args:
            dir_path: 工作目录
            settings: 处理设置
            raw_dict: RAW文件字典
            log_callback: 日志回调
            progress_callback: 进度回调
            stats_callback: 统计回调
        """
        self.dir_path = dir_path
        self.settings = settings
        self.raw_dict = raw_dict
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.stats_callback = stats_callback
        
        # 设备管理器
        self.device_mgr = DeviceManager(
            preferred_device=settings.device if hasattr(settings, 'device') else 'auto',
            cpu_threads=settings.cpu_threads if hasattr(settings, 'cpu_threads') else 0,
            gpu_concurrent=settings.gpu_concurrent if hasattr(settings, 'gpu_concurrent') else 1,
            use_all_devices=(settings.device == 'all' if hasattr(settings, 'device') else False)
        )
    
    def _calculate_conversion_workers(
        self,
        inference_workers: int,
        speed_ratio: float = 10.0,
        min_workers: int = 1,
        max_workers: Optional[int] = None
    ) -> int:
        """
        根据推理速度计算转换线程数
        
        Args:
            inference_workers: 推理工作线程数
            speed_ratio: 推理比转换慢的倍数（默认10倍）
            
        Returns:
            转换线程数
        """
        # 推理慢10倍，转换线程应该是推理线程的1/10，但至少1个
        conversion_workers = max(min_workers, int(inference_workers / speed_ratio))
        # 但也不要太多，避免占用过多CPU
        if max_workers is None:
            max_workers = min(4, multiprocessing.cpu_count() // 2)
        return min(conversion_workers, max_workers)
    
    def build_heif_conversion_stage(
        self,
        heif_files: List[tuple],
        output_queue: JobQueue
    ) -> Pipeline:
        """
        构建HEIF转换阶段（仅转换，不包含AI处理）
        转换完成后立即将结果放入output_queue，供统一的AI处理队列使用
        
        Args:
            heif_files: HEIF文件列表 [(filename, filepath), ...]
            output_queue: 输出队列（转换结果会立即放入此队列，供AI处理使用）
            
        Returns:
            构建好的流水线（仅包含HEIF转换阶段）
        """
        # 创建输入队列
        heif_input_queue = JobQueue()
        
        # 计算并发数（至少4个线程用于预处理，优先喂给推理队列）
        device_configs = self.device_mgr.get_all_configs()
        total_inference_workers = sum(cfg['max_workers'] for cfg in device_configs)
        cpu_threads = self.device_mgr.get_device_config('cpu')['max_workers']
        min_workers = min(4, cpu_threads)
        max_workers = min(cpu_threads, max(min_workers, multiprocessing.cpu_count() // 2))
        conversion_workers = self._calculate_conversion_workers(
            total_inference_workers,
            min_workers=min_workers,
            max_workers=max_workers
        )
        
        # 创建HEIF转换阶段，输出直接进入统一的AI队列
        heif_stage = HEIFConversionStage(
            input_queue=heif_input_queue,
            output_queue=output_queue,  # 直接输出到统一的AI队列
            dir_path=self.dir_path,
            max_workers=conversion_workers,
            log_callback=self.log_callback
        )
        
        # 构建流水线（仅包含HEIF转换阶段）
        pipeline = Pipeline([heif_stage], log_callback=self.log_callback)
        
        # 添加HEIF文件到输入队列
        from core.job_queue import Job
        for filename, filepath in heif_files:
            file_prefix, _ = os.path.splitext(filename)
            job = Job(
                job_id=f"heif_{file_prefix}",
                data={
                    'filename': filename,
                    'filepath': filepath,
                    'file_prefix': file_prefix
                }
            )
            heif_input_queue.put(job)

        return pipeline
    
    def build_raw_pipeline(
        self,
        raw_files_to_convert: List[tuple],
        files_tbr: List[str]
    ) -> Pipeline:
        """
        构建RAW转换流水线
        
        Args:
            raw_files_to_convert: RAW文件列表 [(file_prefix, raw_path), ...]
            files_tbr: 待处理文件列表（会添加转换后的JPG）
            
        Returns:
            构建好的流水线
        """
        # 创建队列
        raw_input_queue = JobQueue()
        raw_output_queue = JobQueue()
        
        # 计算并发数
        device_configs = self.device_mgr.get_all_configs()
        total_inference_workers = sum(cfg['max_workers'] for cfg in device_configs)
        conversion_workers = self._calculate_conversion_workers(total_inference_workers)
        
        # 创建RAW转换阶段
        raw_stage = RAWConversionStage(
            input_queue=raw_input_queue,
            output_queue=raw_output_queue,
            dir_path=self.dir_path,
            max_workers=conversion_workers,
            log_callback=self.log_callback
        )
        
        # 添加RAW文件到输入队列
        from core.job_queue import Job
        for file_prefix, raw_path in raw_files_to_convert:
            job = Job(
                job_id=f"raw_{file_prefix}",
                data={
                    'file_prefix': file_prefix,
                    'raw_ext': os.path.splitext(raw_path)[1]
                }
            )
            raw_input_queue.put(job)
        
        # RAW转换是独立的，不需要后续阶段
        # 转换完成后，JPG文件会被添加到files_tbr
        pipeline = Pipeline([raw_stage], log_callback=self.log_callback)
        
        return pipeline
    
    def build_unified_ai_processing_pipeline(
        self,
        regular_files: List[str],
        shared_ai_queue: JobQueue
    ) -> Pipeline:
        """
        构建统一的AI处理流水线
        支持从共享队列中获取任务（包括HEIF转换输出和常规文件）
        同时支持将常规文件直接加入队列
        
        Args:
            regular_files: 常规文件列表（JPG等，不包括HEIF）
            shared_ai_queue: 共享的AI处理队列（HEIF转换输出也会进入此队列）
            
        Returns:
            构建好的流水线
        """
        # 创建EXIF写入阶段
        exif_input_queue = JobQueue()
        exif_stage = EXIFWriteStage(
            input_queue=exif_input_queue,
            dir_path=self.dir_path,
            raw_dict=self.raw_dict,
            settings=self.settings,
            max_workers=2,
            log_callback=self.log_callback,
            stats_callback=self.stats_callback
        )
        
        # 创建AI处理阶段（多设备）
        # 所有设备共享同一个输入队列，实现负载均衡
        device_configs = self.device_mgr.get_all_configs()
        ai_stages = []
        cpu_threads = self.device_mgr.get_device_config('cpu')['max_workers']
        preprocess_workers = min(4, cpu_threads)
        has_cpu_stage = False
        
        for device_config in device_configs:
            device = device_config['device']
            max_workers = device_config['max_workers']
            if device == 'cpu':
                has_cpu_stage = True
                if preprocess_workers > max_workers:
                    max_workers = preprocess_workers
            
            # 所有设备共享同一个队列，实现真正的负载均衡
            # CPU在转换完成后可以立即参与推理
            ai_stage = ImageProcessingStage(
                input_queue=shared_ai_queue,  # 共享队列
                output_queue=exif_input_queue,  # 所有设备输出到同一个EXIF队列
                dir_path=self.dir_path,
                raw_dict=self.raw_dict,
                settings=self.settings,
                device=device,
                max_workers=max_workers,
                log_callback=self.log_callback,
                stats_callback=self.stats_callback,
                progress_callback=self.progress_callback
            )
            ai_stages.append(ai_stage)
        
        if not has_cpu_stage and preprocess_workers > 0:
            preprocess_stage = ImageProcessingStage(
                input_queue=shared_ai_queue,
                output_queue=exif_input_queue,
                dir_path=self.dir_path,
                raw_dict=self.raw_dict,
                settings=self.settings,
                device='cpu',
                max_workers=preprocess_workers,
                log_callback=self.log_callback,
                stats_callback=self.stats_callback,
                progress_callback=self.progress_callback
            )
            ai_stages.insert(0, preprocess_stage)
        
        # 构建流水线
        stages = ai_stages + [exif_stage]
        pipeline = Pipeline(stages, log_callback=self.log_callback)
        
        # 添加常规文件到共享队列
        from core.job_queue import Job
        for filename in regular_files:
            filepath = os.path.join(self.dir_path, filename)
            file_prefix, _ = os.path.splitext(filename)
            
            job = Job(
                job_id=f"ai_{file_prefix}",
                data={
                    'filename': filename,
                    'filepath': filepath,
                    'file_prefix': file_prefix,
                    'is_heif': False
                }
            )
            shared_ai_queue.put(job)
        
        return pipeline
