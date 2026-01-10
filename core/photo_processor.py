#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Photo Processor - 核心照片处理器
提取自 GUI 和 CLI 的共享业务逻辑

职责：
- 文件扫描和 RAW 转换
- 调用 AI 检测
- 调用 RatingEngine 评分
- 写入 EXIF 元数据
- 文件移动和清理
"""

import os
import time
import json
import shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

# 现有模块
from find_bird_util import raw_to_jpeg
from ai_model import load_yolo_model, detect_and_draw_birds
from exiftool_manager import get_exiftool_manager
from advanced_config import get_advanced_config
from core.rating_engine import RatingEngine, create_rating_engine_from_config
from core.keypoint_detector import KeypointDetector, get_keypoint_detector
from core.flight_detector import FlightDetector, get_flight_detector, FlightResult
from core.exposure_detector import ExposureDetector, get_exposure_detector, ExposureResult
from core.focus_point_detector import get_focus_detector, verify_focus_in_bbox

from constants import RATING_FOLDER_NAMES, RAW_EXTENSIONS, JPG_EXTENSIONS


@dataclass
class ProcessingSettings:
    """处理参数配置"""
    ai_confidence: int = 50
    sharpness_threshold: int = 400   # 头部区域锐度达标阈值 (200-600)
    nima_threshold: float = 5.2  # TOPIQ 美学达标阈值 (4.0-7.0)
    save_crop: bool = False
    normalization_mode: str = 'log_compression'  # 默认使用log_compression，与GUI一致
    detect_flight: bool = True  # V3.4: 飞版检测开关
    detect_exposure: bool = False  # V3.8: 曝光检测开关（默认关闭）
    exposure_threshold: float = 0.10  # V3.8: 曝光阈值 (0.05-0.20)
    device: str = 'auto'  # 计算设备选择: 'auto', 'cuda', 'cpu', 'mps', 'all'
    stop_event: Optional[Any] = None  # 停止事件（用于取消处理）
    keep_temp_jpg: bool = True  # 是否保留临时转换的JPG文件
    cpu_threads: int = 0  # CPU推理线程数（0=自动，使用CPU逻辑核心数）
    gpu_concurrent: int = 10  # GPU推理并发数（1=串行，>1=并发队列，需考虑显存）
    use_pipeline: bool = True  # 是否使用新的流水线框架（默认启用）


@dataclass
class ProcessingCallbacks:
    """回调函数（用于进度更新和日志输出）"""
    log: Optional[Callable[[str, str], None]] = None
    progress: Optional[Callable[[int], None]] = None


@dataclass
class ProcessingResult:
    """处理结果数据"""
    stats: Dict[str, any] = field(default_factory=dict)
    file_ratings: Dict[str, int] = field(default_factory=dict)
    star_3_photos: List[Dict] = field(default_factory=list)
    total_time: float = 0.0
    avg_time: float = 0.0


class PhotoProcessor:
    """
    核心照片处理器
    
    封装所有业务逻辑，GUI 和 CLI 都调用这个类
    """
    
    def __init__(
        self,
        dir_path: str,
        settings: ProcessingSettings,
        callbacks: Optional[ProcessingCallbacks] = None
    ):
        """
        初始化处理器
        
        Args:
            dir_path: 处理目录路径
            settings: 处理参数
            callbacks: 回调函数（进度、日志）
        """
        self.dir_path = dir_path
        self.settings = settings
        self.callbacks = callbacks or ProcessingCallbacks()
        self.config = get_advanced_config()
        
        # 初始化评分引擎
        self.rating_engine = create_rating_engine_from_config(self.config)
        # 使用 UI 设置更新达标阈值
        self.rating_engine.update_thresholds(
            sharpness_threshold=settings.sharpness_threshold,
            nima_threshold=settings.nima_threshold
        )
        
        # DEBUG: 输出参数
        self._log(f"\n🔍 DEBUG - 处理参数:")
        self._log(f"  📊 AI置信度: {settings.ai_confidence}")
        self._log(f"  📏 锐度阈值: {settings.sharpness_threshold}")
        self._log(f"  🎨 NIMA阈值: {settings.nima_threshold}")
        self._log(f"  🔧 归一化模式: {settings.normalization_mode}")
        self._log(f"  🦅 飞鸟检测: {'开启' if settings.detect_flight else '关闭'}")
        self._log(f"  📸 曝光检测: {'开启' if settings.detect_exposure else '关闭'}")
        self._log(f"  ⚙️  高级配置 - min_sharpness: {self.config.min_sharpness}")
        self._log(f"  ⚙️  高级配置 - min_nima: {self.config.min_nima}\n")
        
        # 统计数据（支持 0/1/2/3 星）
        self.stats = {
            'total': 0,
            'star_3': 0,
            'picked': 0,
            'star_2': 0,
            'star_1': 0,  # 普通照片（合格）
            'star_0': 0,  # 普通照片（问题）
            'no_bird': 0,
            'flying': 0,  # V3.6: 飞鸟照片计数
            'exposure_issue': 0,  # V3.8: 曝光问题计数
            'start_time': 0,
            'end_time': 0,
            'total_time': 0,
            'avg_time': 0,
            # 新增统计字段
            'photo_times': [],  # 每张图片的处理时间列表 [(filename, time_ms, detected)]
            'with_bird_times': [],  # 带鸟图片的处理时间
            'no_bird_times': [],  # 不带鸟图片的处理时间
            'longest_photo': None,  # (filename, time_ms)
            'shortest_photo': None,  # (filename, time_ms)
            'avg_with_bird_time': 0.0,  # 带鸟图片平均处理时间
            'avg_no_bird_time': 0.0,  # 不带鸟图片平均处理时间
            'cancelled': False  # 是否被取消
        }
        
        # 停止事件（用于取消处理）
        self.stop_event = settings.stop_event
        
        # 内部状态
        self.file_ratings = {}
        self.star2_reasons = {}  # 记录2星原因: 'sharpness' 或 'nima'
        self.star_3_photos = []
        self.heif_temp_map = {}  # HEIF 文件到临时 JPG 的映射
        self.picked_files = set()  # 精选文件集合（用于判断是否精选）
        
        # 线程安全锁（用于并行处理）
        import threading
        self._stats_lock = threading.Lock()
        
        # 流水线模式下的进度跟踪
        self._pipeline_total_files = 0  # 总文件数
        self._pipeline_processed_files = 0  # 已处理文件数
        self._pipeline_progress_lock = threading.Lock()  # 进度锁
        
        # 流水线实例（用于UI监控）
        self._pipelines = []  # 保存流水线实例列表
    
    def _log(self, msg: str, level: str = "info"):
        """内部日志方法"""
        if self.callbacks.log:
            self.callbacks.log(msg, level)
    
    def _progress(self, percent: int = -1):
        """
        内部进度更新
        
        Args:
            percent: 进度百分比 (0-100)，-1 表示基于已处理文件数自动计算
        """
        if self.callbacks.progress:
            # 如果传递 -1，表示流水线模式下的进度更新（基于已处理文件数计算）
            if percent == -1:
                with self._pipeline_progress_lock:
                    if self._pipeline_total_files > 0:
                        # 基于已处理文件数计算进度
                        calculated_percent = int((self._pipeline_processed_files / self._pipeline_total_files) * 100)
                        calculated_percent = min(100, max(0, calculated_percent))  # 限制在 0-100
                        self.callbacks.progress(calculated_percent)
                    # 如果总文件数为0，不更新进度（避免除零错误）
            else:
                self.callbacks.progress(percent)
    
    def process(
        self,
        organize_files: bool = True,
        cleanup_temp: bool = True
    ) -> ProcessingResult:
        """
        主处理流程
        
        Args:
            organize_files: 是否移动文件到分类文件夹
            cleanup_temp: 是否清理临时JPG文件
            
        Returns:
            ProcessingResult 包含统计数据和处理结果
        """
        start_time = time.time()
        self.stats['start_time'] = start_time
        
        # 阶段1: 文件扫描
        raw_dict, jpg_dict, files_tbr = self._scan_files()
        
        # 阶段2: RAW转换
        raw_files_to_convert = self._identify_raws_to_convert(raw_dict, jpg_dict, files_tbr)
        if raw_files_to_convert:
            self._convert_raws(raw_files_to_convert, files_tbr)
        
        # 阶段2.5: HEIF/HIF 并行转换（仅非流水线模式）
        # 流水线模式下，HEIF转换会在流水线中处理，转换一张立即进入推理队列
        use_pipeline = getattr(self.settings, 'use_pipeline', True)  # 默认启用
        if not use_pipeline:
            # 非流水线模式：提前转换所有HEIF文件
            heif_files_to_convert = self._identify_heif_to_convert(files_tbr)
            if heif_files_to_convert:
                self._convert_heif_files(heif_files_to_convert)
        
        # 阶段3: AI检测与评分
        # 使用新的流水线框架（如果启用）
        if use_pipeline:
            self._process_images_with_pipeline(files_tbr, raw_dict)
        else:
            self._process_images(files_tbr, raw_dict)
        
        # 阶段4: 精选旗标计算
        self._calculate_picked_flags()
        
        # 阶段5: 文件组织
        if organize_files:
            self._move_files_to_rating_folders(raw_dict)
        
        # 阶段6: 清理临时文件
        if cleanup_temp:
            self._cleanup_temp_files(files_tbr, raw_dict)
        
        # 记录结束时间
        end_time = time.time()
        self.stats['end_time'] = end_time
        self.stats['total_time'] = end_time - start_time
        self.stats['avg_time'] = (
            self.stats['total_time'] / self.stats['total']
            if self.stats['total'] > 0 else 0
        )
        
        # 计算详细统计信息
        if self.stats['photo_times']:
            # 最长/最短处理时间
            longest = max(self.stats['photo_times'], key=lambda x: x[1])
            shortest = min(self.stats['photo_times'], key=lambda x: x[1])
            self.stats['longest_photo'] = (longest[0], longest[1])
            self.stats['shortest_photo'] = (shortest[0], shortest[1])
            
            # 带鸟/不带鸟平均时间
            if self.stats['with_bird_times']:
                self.stats['avg_with_bird_time'] = sum(self.stats['with_bird_times']) / len(self.stats['with_bird_times'])
            if self.stats['no_bird_times']:
                self.stats['avg_no_bird_time'] = sum(self.stats['no_bird_times']) / len(self.stats['no_bird_times'])
        
        return ProcessingResult(
            stats=self.stats.copy(),
            file_ratings=self.file_ratings.copy(),
            star_3_photos=self.star_3_photos.copy(),
            total_time=self.stats['total_time'],
            avg_time=self.stats['avg_time']
        )
    
    def _scan_files(self) -> Tuple[dict, dict, list]:
        """扫描目录文件"""
        scan_start = time.time()
        
        raw_dict = {}
        jpg_dict = {}
        files_tbr = []
        
        for filename in os.listdir(self.dir_path):
            if filename.startswith('.'):
                continue

            
            file_prefix, file_ext = os.path.splitext(filename)
            if file_ext.lower() in RAW_EXTENSIONS:
                raw_dict[file_prefix] = file_ext
            if file_ext.lower() in JPG_EXTENSIONS:
                jpg_dict[file_prefix] = file_ext
                files_tbr.append(filename)
        
        scan_time = (time.time() - scan_start) * 1000
        self._log(f"⏱️  文件扫描耗时: {scan_time:.1f}ms")
        
        return raw_dict, jpg_dict, files_tbr
    
    def _identify_raws_to_convert(self, raw_dict, jpg_dict, files_tbr):
        """识别需要转换的RAW文件"""
        raw_files_to_convert = []
        
        for key, value in raw_dict.items():
            if key in jpg_dict:
                jpg_dict.pop(key)
                continue
            else:
                raw_file_path = os.path.join(self.dir_path, key + value)
                raw_files_to_convert.append((key, raw_file_path))
        
        return raw_files_to_convert
    
    def _convert_raws(self, raw_files_to_convert, files_tbr):
        """并行转换RAW文件"""
        raw_start = time.time()
        import multiprocessing
        max_workers = min(4, multiprocessing.cpu_count())
        
        self._log(f"🔄 开始并行转换 {len(raw_files_to_convert)} 个RAW文件({max_workers}线程)...")
        
        def convert_single(args):
            key, raw_path = args
            try:
                raw_to_jpeg(raw_path)
                return (key, True, None)
            except Exception as e:
                return (key, False, str(e))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_raw = {
                executor.submit(convert_single, args): args 
                for args in raw_files_to_convert
            }
            converted_count = 0
            
            for future in as_completed(future_to_raw):
                key, success, error = future.result()
                if success:
                    files_tbr.append(key + ".jpg")
                    converted_count += 1
                    if converted_count % 5 == 0 or converted_count == len(raw_files_to_convert):
                        self._log(f"  ✅ 已转换 {converted_count}/{len(raw_files_to_convert)} 张")
                else:
                    self._log(f"  ❌ 转换失败: {key} ({error})", "error")
        
        raw_time = time.time() - raw_start
        avg_time = raw_time / len(raw_files_to_convert) if len(raw_files_to_convert) > 0 else 0
        self._log(f"⏱️  RAW转换耗时: {raw_time:.1f}秒 (平均 {avg_time:.1f}秒/张)\n")
    
    def _identify_heif_to_convert(self, files_tbr):
        """识别需要转换的 HEIF/HIF 文件"""
        heif_files = []
        heif_extensions = ['.heif', '.heic', '.hif']
        
        for filename in files_tbr:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in heif_extensions:
                filepath = os.path.join(self.dir_path, filename)
                heif_files.append((filename, filepath))
        
        return heif_files
    
    def _convert_heif_files(self, heif_files_to_convert):
        """并行转换 HEIF/HIF 文件为临时 JPG"""
        if not heif_files_to_convert:
            return
        
        heif_start = time.time()
        import multiprocessing
        max_workers = min(8, multiprocessing.cpu_count())  # HEIF转换可以更多线程
        
        self._log(f"🔄 开始并行转换 {len(heif_files_to_convert)} 个 HEIF/HIF 文件({max_workers}线程)...")
        
        # 创建临时目录
        temp_dir = os.path.join(self.dir_path, '.superpicky', 'temp_jpg')
        os.makedirs(temp_dir, exist_ok=True)
        
        def convert_single_heif(args):
            filename, heif_path = args
            try:
                # 生成临时 JPG 路径
                file_basename = os.path.splitext(filename)[0]
                temp_jpg_path = os.path.join(temp_dir, f"{file_basename}_temp.jpg")
                
                # 检查临时JPG是否已存在，如果存在则直接使用，跳过转换
                if os.path.exists(temp_jpg_path):
                    # 验证文件是否有效（大小大于0）
                    if os.path.getsize(temp_jpg_path) > 0:
                        return (filename, True, temp_jpg_path, None)
                    else:
                        # 文件存在但大小为0，删除后重新转换
                        try:
                            os.remove(temp_jpg_path)
                        except:
                            pass
                
                # 临时JPG不存在或无效，执行转换
                # 注册 pillow-heif
                try:
                    from pillow_heif import register_heif_opener
                    register_heif_opener()
                except ImportError:
                    pass
                
                from PIL import Image
                
                # 读取并转换
                pil_image = Image.open(heif_path).convert('RGB')
                
                # 保存为 JPG
                pil_image.save(temp_jpg_path, 'JPEG', quality=95)
                
                return (filename, True, temp_jpg_path, None)
            except Exception as e:
                return (filename, False, None, str(e))
        
        # 存储转换映射：原始文件名 -> 临时JPG路径
        self.heif_temp_map = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_heif = {
                executor.submit(convert_single_heif, args): args 
                for args in heif_files_to_convert
            }
            converted_count = 0
            reused_count = 0
            
            for future in as_completed(future_to_heif):
                filename, success, temp_jpg_path, error = future.result()
                if success:
                    self.heif_temp_map[filename] = temp_jpg_path
                    # 检查是否是复用的文件（通过检查文件修改时间是否早于处理开始时间）
                    if os.path.exists(temp_jpg_path):
                        file_mtime = os.path.getmtime(temp_jpg_path)
                        if file_mtime < heif_start:
                            reused_count += 1
                        else:
                            converted_count += 1
                    else:
                        converted_count += 1
                    
                    total_processed = converted_count + reused_count
                    if total_processed % 10 == 0 or total_processed == len(heif_files_to_convert):
                        status_msg = f"  ✅ 已处理 {total_processed}/{len(heif_files_to_convert)} 张"
                        if reused_count > 0:
                            status_msg += f" (转换: {converted_count}, 复用: {reused_count})"
                        self._log(status_msg)
                else:
                    self._log(f"  ❌ 转换失败: {filename} ({error})", "error")
        
        heif_time = time.time() - heif_start
        if converted_count > 0:
            avg_time = heif_time / converted_count
            if reused_count > 0:
                self._log(f"⏱️  HEIF转换耗时: {heif_time:.1f}秒 (转换 {converted_count} 张, 平均 {avg_time:.1f}秒/张, 复用 {reused_count} 张)\n")
            else:
                self._log(f"⏱️  HEIF转换耗时: {heif_time:.1f}秒 (转换 {converted_count} 张, 平均 {avg_time:.1f}秒/张)\n")
        else:
            self._log(f"⏱️  HEIF处理耗时: {heif_time:.1f}秒 (全部复用现有文件, {reused_count} 张)\n")
    
    def _process_images_with_pipeline(self, files_tbr, raw_dict):
        """
        使用新的流水线框架处理图片
        支持流水线式HEIF转换、多设备并行推理
        优化：HEIF转换完成后立即进入推理队列，CPU转换完成后可参与推理
        """
        self._log("🚀 使用流水线框架处理图片...")
        
        try:
            from core.pipeline_builder import PipelineBuilder
            from core.job_queue import JobQueue
            
            # 创建构建器
            builder = PipelineBuilder(
                dir_path=self.dir_path,
                settings=self.settings,
                raw_dict=raw_dict,
                log_callback=self._log,
                progress_callback=self._progress,
                stats_callback=self._handle_pipeline_stats
            )
            
            # 识别HEIF文件
            heif_files = self._identify_heif_to_convert(files_tbr)
            regular_files = [f for f in files_tbr if f not in [hf[0] for hf in heif_files]]
            
            # 创建统一的AI处理队列（HEIF转换输出和常规文件都进入此队列）
            device_configs = builder.device_mgr.get_all_configs()
            total_inference_workers = sum(cfg['max_workers'] for cfg in device_configs)
            queue_maxsize = max(8, total_inference_workers * 2)
            shared_ai_queue = JobQueue(maxsize=queue_maxsize)
            
            # 构建并启动流水线
            pipelines = []
            
            # 1. HEIF转换阶段（如果有HEIF文件）
            # 转换完成后立即将结果放入shared_ai_queue，实现流式处理
            if heif_files:
                self._log(f"📦 构建HEIF转换阶段（{len(heif_files)}个文件，转换完成后立即进入推理队列）...")
                heif_pipeline = builder.build_heif_conversion_stage(heif_files, shared_ai_queue)
                heif_pipeline.start()
                pipelines.append(heif_pipeline)
            
            # 2. 统一的AI处理流水线（处理HEIF转换输出和常规文件）
            # 所有设备共享同一个队列，CPU在转换完成后可以立即参与推理
            if heif_files or regular_files:
                total_files = len(heif_files) + len(regular_files)
                # 保存总文件数用于进度计算
                with self._pipeline_progress_lock:
                    self._pipeline_total_files = total_files
                    self._pipeline_processed_files = 0
                self._log(f"📦 构建统一AI处理流水线（{total_files}个文件，HEIF转换完成后CPU可参与推理）...")
                ai_pipeline = builder.build_unified_ai_processing_pipeline(regular_files, shared_ai_queue)
                ai_pipeline.start()
                pipelines.append(ai_pipeline)
            
            # 保存流水线实例供UI监控使用
            self._pipelines = pipelines
            self._shared_ai_queue = shared_ai_queue
            
            # 等待所有流水线完成（支持取消）
            self._log("⏳ 等待流水线处理完成...")
            import time
            last_progress_log = time.time()
            
            for pipeline in pipelines:
                # 轮询等待，允许中断检查
                while True:
                    # 检查是否已取消
                    if self.stop_event and self.stop_event.is_set():
                        self._log("⚠️  检测到取消信号，正在停止流水线...", "warning")
                        break
                    
                    # 检查所有阶段是否已完成
                    all_done = True
                    for stage in pipeline.stages:
                        if stage.input_queue:
                            # 检查队列统计：所有任务都已放入且都已完成
                            queue_stats = stage.input_queue.get_stats()
                            total_put = queue_stats.get('total_put', 0)
                            total_done = queue_stats.get('total_done', 0)
                            
                            # 如果队列不为空，或者还有任务在处理中（put > done），则未完成
                            if not stage.input_queue.empty() or total_put > total_done:
                                all_done = False
                                break
                    
                    # 定期输出进度日志（每5秒）
                    current_time = time.time()
                    if current_time - last_progress_log >= 5.0:
                        # 输出当前进度
                        for stage in pipeline.stages:
                            if stage.input_queue:
                                queue_stats = stage.input_queue.get_stats()
                                stage_stats = stage.get_stats()
                                processed = stage_stats.get('processed', 0)
                                failed = stage_stats.get('failed', 0)
                                self._log(f"  [{stage.name}] 已处理: {processed}, 失败: {failed}, "
                                        f"队列: {queue_stats.get('total_put', 0)}/{queue_stats.get('total_done', 0)}")
                        last_progress_log = current_time
                    
                    if all_done:
                        break
                    
                    time.sleep(0.1)  # 短暂等待，避免CPU占用过高
                
                # 如果已取消，跳出循环
                if self.stop_event and self.stop_event.is_set():
                    break
            
            # 等待所有队列完成（确保所有task_done都被调用）
            self._log("⏳ 等待所有任务完成...")
            for pipeline in pipelines:
                for stage in pipeline.stages:
                    if stage.input_queue:
                        try:
                            # 等待队列join完成（最多等待30秒）
                            import queue
                            start_wait = time.time()
                            while not stage.input_queue.empty() or stage.input_queue.qsize() > 0:
                                if time.time() - start_wait > 30:
                                    self._log(f"⚠️  等待 {stage.name} 队列超时", "warning")
                                    break
                                time.sleep(0.1)
                            # 尝试join，但设置超时
                            stage.input_queue.join()
                        except Exception as e:
                            self._log(f"⚠️  等待 {stage.name} 队列时出错: {e}", "warning")
            
            # 同步HEIF转换映射（用于保留临时JPG功能）
            # 从HEIF转换阶段获取heif_temp_map
            if heif_files:
                for stage in heif_pipeline.stages:
                    if hasattr(stage, 'heif_temp_map'):
                        # 同步映射到PhotoProcessor，供后续清理或保留使用
                        self.heif_temp_map.update(stage.heif_temp_map)
                        break
            
            # 停止所有流水线（无论是否完成）
            for pipeline in pipelines:
                pipeline.stop()
            
            # 清空流水线引用
            self._pipelines = []
            self._shared_ai_queue = None
            
            # 输出统计信息
            self._log("\n📊 流水线统计:")
            for pipeline in pipelines:
                stats = pipeline.get_stats()
                for stage_name, stage_stats in stats.items():
                    self._log(f"  {stage_name}: 处理 {stage_stats.get('processed', 0)} 个任务, "
                            f"失败 {stage_stats.get('failed', 0)} 个, "
                            f"平均耗时 {stage_stats.get('avg_time', 0):.2f}秒")
            
            self._log("✅ 流水线处理完成")
            
        except Exception as e:
            self._log(f"❌ 流水线处理失败: {e}", "error")
            import traceback
            self._log(traceback.format_exc(), "error")
            # 降级到原有方法
            self._log("⚠️  降级到原有处理方法", "warning")
            self._process_images(files_tbr, raw_dict)
    
    def _handle_pipeline_stats(self, result: Dict[str, Any]):
        """处理流水线统计回调"""
        # 更新已处理文件数并计算进度
        with self._pipeline_progress_lock:
            self._pipeline_processed_files += 1
            # 触发进度更新（传递 -1 表示自动计算）
            if self._pipeline_processed_files % 5 == 0 or self._pipeline_processed_files == self._pipeline_total_files:
                self._progress(-1)
        
        # 更新统计信息
        rating_value = result.get('rating', 0)
        is_flying = result.get('is_flying', False)
        has_exposure_issue = result.get('is_overexposed', False) or result.get('is_underexposed', False)
        
        self._update_stats(rating_value, is_flying, has_exposure_issue)
        
        # 记录处理时间
        processing_time = result.get('processing_time', 0) * 1000  # 转换为毫秒
        filename = result.get('filename', '')
        detected = result.get('detected', False)
        
        self.stats['photo_times'].append((filename, processing_time, detected))
        if detected:
            self.stats['with_bird_times'].append(processing_time)
        else:
            self.stats['no_bird_times'].append(processing_time)
        
        # 更新文件评分
        file_prefix = result.get('file_prefix')
        if file_prefix:
            self.file_ratings[file_prefix] = rating_value
            
            # 收集3星照片
            if rating_value == 3:
                topiq = result.get('topiq')
                head_sharpness = result.get('head_sharpness', 0)
                if topiq is not None:
                    filepath = result.get('filepath')
                    if filepath:
                        self.star_3_photos.append({
                            'file': filepath,
                            'nima': topiq,
                            'sharpness': head_sharpness
                        })
            
            # 记录2星原因
            if rating_value == 2:
                head_sharpness = result.get('head_sharpness', 0)
                topiq = result.get('topiq')
                sharpness_ok = head_sharpness >= self.settings.sharpness_threshold
                topiq_ok = topiq is not None and topiq >= self.settings.nima_threshold
                if sharpness_ok and not topiq_ok:
                    self.star2_reasons[file_prefix] = 'sharpness'
                elif topiq_ok and not sharpness_ok:
                    self.star2_reasons[file_prefix] = 'nima'
                else:
                    self.star2_reasons[file_prefix] = 'both'
        
        # 更新CSV（在EXIF写入阶段已经处理，这里可以跳过）
    
    def _process_images(self, files_tbr, raw_dict):
        """处理所有图片 - AI检测、关键点检测与评分"""
        # 检查是否已取消
        if self.stop_event and self.stop_event.is_set():
            self.stats['cancelled'] = True
            self._log("⚠️  处理已取消", "warning")
            return
        
        # 加载模型（使用指定设备）
        model_start = time.time()
        self._log("🤖 加载AI模型...")
        device = self.settings.device if hasattr(self.settings, 'device') else 'auto'
        self._log(f"🖥️  使用设备: {device}")
        model = load_yolo_model(device=device)
        model_time = (time.time() - model_start) * 1000
        self._log(f"⏱️  模型加载耗时: {model_time:.0f}ms")
        
        # 加载关键点检测模型
        self._log("👁️  加载关键点模型...")
        keypoint_detector = get_keypoint_detector()
        try:
            keypoint_detector.load_model()
            self._log("✅ 关键点模型加载成功")
            use_keypoints = True
        except FileNotFoundError:
            self._log("⚠️  关键点模型未找到，使用传统锐度计算", "warning")
            use_keypoints = False
        
        # V3.4: 加载飞版检测模型
        use_flight = False
        flight_detector = None
        if self.settings.detect_flight:
            self._log("🦅 加载飞版检测模型...")
            flight_detector = get_flight_detector()
            try:
                flight_detector.load_model()
                self._log("✅ 飞版检测模型加载成功")
                use_flight = True
            except FileNotFoundError:
                self._log("⚠️  飞版检测模型未找到，跳过飞版检测", "warning")
                use_flight = False
        
        total_files = len(files_tbr)
        self._log(f"📁 共 {total_files} 个文件待处理\n")
        
        exiftool_mgr = get_exiftool_manager()
        
        # UI设置转为列表格式
        ui_settings = [
            self.settings.ai_confidence,
            self.settings.sharpness_threshold,
            self.settings.nima_threshold,
            self.settings.save_crop,
            self.settings.normalization_mode
        ]
        
        ai_total_start = time.time()
        
        # 确定实际使用的设备
        actual_device = get_best_device(device) if hasattr(self, 'get_best_device') else device
        try:
            from utils import get_best_device
            actual_device = get_best_device(device)
        except:
            actual_device = device
        
        # 判断是否使用并行处理
        use_parallel = False
        is_cpu = actual_device == 'cpu'
        is_gpu = actual_device in ['cuda', 'mps']
        
        # CPU: 使用线程池并行
        if is_cpu:
            import multiprocessing
            cpu_threads = self.settings.cpu_threads if hasattr(self.settings, 'cpu_threads') else 0
            if cpu_threads == 0:
                cpu_threads = multiprocessing.cpu_count()
            use_parallel = cpu_threads > 1
            if use_parallel:
                self._log(f"🔄 使用 CPU 线程池并行处理（{cpu_threads} 线程）")
        
        # GPU: 使用队列控制并发（避免显存溢出）
        elif is_gpu:
            gpu_concurrent = self.settings.gpu_concurrent if hasattr(self.settings, 'gpu_concurrent') else 1
            use_parallel = gpu_concurrent > 1
            if use_parallel:
                self._log(f"🔄 使用 GPU 队列并发处理（并发数: {gpu_concurrent}）")
        
        if use_parallel:
            # 并行处理模式
            self._process_images_parallel(files_tbr, raw_dict, model, ui_settings, 
                                         use_keypoints, keypoint_detector, use_flight, 
                                         flight_detector, exiftool_mgr, actual_device, 
                                         is_cpu, is_gpu)
        else:
            # 串行处理模式（原有逻辑）
            self._process_images_sequential(files_tbr, raw_dict, model, ui_settings, 
                                          use_keypoints, keypoint_detector, use_flight, 
                                          flight_detector, exiftool_mgr)
        
        ai_total_time = time.time() - ai_total_start
        avg_ai_time = ai_total_time / len(files_tbr) if len(files_tbr) > 0 else 0
        self._log(f"\n⏱️  AI检测总耗时: {ai_total_time:.1f}秒 (平均 {avg_ai_time:.1f}秒/张)")
    
    def _process_images_sequential(self, files_tbr, raw_dict, model, ui_settings,
                                   use_keypoints, keypoint_detector, use_flight,
                                   flight_detector, exiftool_mgr):
        """串行处理图片（原有逻辑）"""
        total_files = len(files_tbr)
        
        for i, filename in enumerate(files_tbr, 1):
            # 检查是否已取消
            if self.stop_event and self.stop_event.is_set():
                self.stats['cancelled'] = True
                self._log(f"\n⚠️  处理已取消（已处理 {i-1}/{total_files} 张）", "warning")
                break
            
            # 记录每张照片的开始时间
            photo_start_time = time.time()
            
            filepath = os.path.join(self.dir_path, filename)
            file_prefix, _ = os.path.splitext(filename)
            
            # 更新进度
            should_update = (i % 5 == 0 or i == total_files or i == 1)
            if should_update:
                progress = int((i / total_files) * 100)
                self._progress(progress)
            
            # 优化流程：YOLO → 关键点检测(在crop上) → 条件NIMA
            # Phase 1: 先做YOLO检测（跳过NIMA），获取鸟的位置和bbox
            try:
                result = detect_and_draw_birds(
                    filepath, model, None, self.dir_path, ui_settings, None, skip_nima=True
                )
                if result is None:
                    self._log(f"  ⚠️  无法处理(AI推理失败)", "error")
                    continue
            except Exception as e:
                self._log(f"  ❌ 处理异常: {e}", "error")
                continue
            
            # 解构 AI 结果 (包含bbox, 图像尺寸, 分割掩码) - V3.2移除BRISQUE
            detected, _, confidence, sharpness, _, bird_bbox, img_dims, bird_mask = result
            
            # Phase 2: 关键点检测（在裁剪区域上执行，更准确）
            all_keypoints_hidden = False
            both_eyes_hidden = False  # 保留用于日志/调试
            best_eye_visibility = 0.0  # V3.8: 眼睛最高置信度，用于封顶逻辑
            head_sharpness = 0.0
            has_visible_eye = False
            has_visible_beak = False
            left_eye_vis = 0.0
            right_eye_vis = 0.0
            beak_vis = 0.0
            
            # V3.9: 头部区域信息（用于对焦验证）
            head_center_orig = None
            head_radius_val = None
            
            # V3.2优化: 只读取原图一次，在关键点检测和NIMA计算中复用
            orig_img = None  # 原图缓存
            bird_crop_bgr = None  # 裁剪区域缓存（BGR）
            bird_crop_mask = None # 裁剪区域掩码缓存
            bird_mask_orig = None  # V3.9: 原图尺寸的分割掩码（用于对焦验证）
            
            if use_keypoints and detected and bird_bbox is not None and img_dims is not None:
                try:
                    import cv2
                    from utils import read_image
                    orig_img = read_image(filepath)  # 只读取一次!（支持 HEIF/HEIC）
                    if orig_img is not None:
                        h_orig, w_orig = orig_img.shape[:2]
                        # 获取YOLO处理时的图像尺寸
                        w_resized, h_resized = img_dims
                        
                        # 计算缩放比例：原图 / 缩放图
                        scale_x = w_orig / w_resized
                        scale_y = h_orig / h_resized
                        
                        # 将bbox从缩放尺寸转换到原图尺寸
                        x, y, w, h = bird_bbox
                        x_orig = int(x * scale_x)
                        y_orig = int(y * scale_y)
                        w_orig_box = int(w * scale_x)
                        h_orig_box = int(h * scale_y)
                        
                        # 确保边界有效
                        x_orig = max(0, min(x_orig, w_orig - 1))
                        y_orig = max(0, min(y_orig, h_orig - 1))
                        w_orig_box = min(w_orig_box, w_orig - x_orig)
                        h_orig_box = min(h_orig_box, h_orig - y_orig)
                        
                        # 裁剪鸟的区域（保存BGR版本供NIMA使用）
                        bird_crop_bgr = orig_img[y_orig:y_orig+h_orig_box, x_orig:x_orig+w_orig_box]
                        
                        # 同样裁剪 mask (如果存在)
                        if bird_mask is not None:
                            # 缩放 mask 到原图尺寸 (Mask是整图的)
                            # bird_mask 是 (h_resized, w_resized)，需要放大到 (h_orig, w_orig)
                            if bird_mask.shape[:2] != (h_orig, w_orig):
                                # 使用最近邻插值保持二值特性
                                bird_mask_orig = cv2.resize(bird_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                            else:
                                bird_mask_orig = bird_mask
                                
                            bird_crop_mask = bird_mask_orig[y_orig:y_orig+h_orig_box, x_orig:x_orig+w_orig_box]
                        
                        if bird_crop_bgr.size > 0:
                            crop_rgb = cv2.cvtColor(bird_crop_bgr, cv2.COLOR_BGR2RGB)
                            # 在裁剪区域上进行关键点检测，传入分割掩码
                            kp_result = keypoint_detector.detect(
                                crop_rgb, 
                                box=(x_orig, y_orig, w_orig_box, h_orig_box),
                                seg_mask=bird_crop_mask  # 传入分割掩码
                            )
                            if kp_result is not None:
                                both_eyes_hidden = kp_result.both_eyes_hidden  # 保留兼容
                                all_keypoints_hidden = kp_result.all_keypoints_hidden  # 新属性
                                best_eye_visibility = kp_result.best_eye_visibility  # V3.8
                                has_visible_eye = kp_result.visible_eye is not None
                                has_visible_beak = kp_result.beak_vis >= 0.3  # V3.8: 降低到 0.3
                                left_eye_vis = kp_result.left_eye_vis
                                right_eye_vis = kp_result.right_eye_vis
                                beak_vis = kp_result.beak_vis
                                head_sharpness = kp_result.head_sharpness
                                
                                # V3.9: 计算头部区域中心和半径（用于对焦验证）
                                ch, cw = bird_crop_bgr.shape[:2]
                                # 选择更可见的眼睛作为头部中心
                                if left_eye_vis >= right_eye_vis and left_eye_vis >= 0.3:
                                    eye_px = (int(kp_result.left_eye[0] * cw), int(kp_result.left_eye[1] * ch))
                                elif right_eye_vis >= 0.3:
                                    eye_px = (int(kp_result.right_eye[0] * cw), int(kp_result.right_eye[1] * ch))
                                else:
                                    eye_px = None
                                
                                if eye_px is not None:
                                    # 转换到原图坐标
                                    head_center_orig = (eye_px[0] + x_orig, eye_px[1] + y_orig)
                                    # 计算半径
                                    beak_px = (int(kp_result.beak[0] * cw), int(kp_result.beak[1] * ch))
                                    if beak_vis >= 0.3:
                                        import math
                                        dist = math.sqrt((eye_px[0] - beak_px[0])**2 + (eye_px[1] - beak_px[1])**2)
                                        head_radius_val = int(dist * 1.2)
                                    else:
                                        head_radius_val = int(max(cw, ch) * 0.15)
                                    head_radius_val = max(20, min(head_radius_val, min(cw, ch) // 2))
                except Exception as e:
                    self._log(f"  ⚠️ 关键点检测异常: {e}", "warning")
                    # import traceback
                    # self._log(traceback.format_exc(), "error")
                    pass
            
            # Phase 3: 根据关键点可见性决定是否计算TOPIQ
            # V4.0: 眼睛可见度 < 30% 时也跳过 TOPIQ（节省时间）
            topiq = None
            if detected and not all_keypoints_hidden and best_eye_visibility >= 0.3:
                # 双眼可见，需要计算NIMA以进行星级判定
                try:
                    from iqa_scorer import get_iqa_scorer
                    from utils import get_best_device
                    import time as time_module
                    
                    step_start = time_module.time()
                    # 使用设置中指定的设备
                    device = get_best_device(self.settings.device if hasattr(self.settings, 'device') else 'auto')
                    scorer = get_iqa_scorer(device=device)
                    
                    # V3.7: 使用全图而非裁剪图进行TOPIQ美学评分
                    # 全图评分 + 头部锐度阈值 是更好的组合：
                    # - 全图评分评估整体画面构图和美感
                    # - 头部锐度阈值确保鸟本身足够清晰
                    topiq = scorer.calculate_nima(filepath)
                    
                    topiq_time = (time_module.time() - step_start) * 1000
                except Exception as e:
                    pass  # V3.3: 简化日志，静默 TOPIQ 计算失败
            # V3.8: 移除跳过日志，改用 all_keypoints_hidden 后跳过的情况会少很多
            
            # Phase 4: V3.4 飞版检测（在鸟的裁剪区域上执行）
            is_flying = False
            flight_confidence = 0.0
            if use_flight and detected and bird_crop_bgr is not None and bird_crop_bgr.size > 0:
                try:
                    flight_result = flight_detector.detect(bird_crop_bgr)
                    is_flying = flight_result.is_flying
                    flight_confidence = flight_result.confidence
                    # DEBUG: 输出飞版检测结果
                    # self._log(f"  🦅 飞版检测: is_flying={is_flying}, conf={flight_confidence:.2f}")
                except Exception as e:
                    self._log(f"  ⚠️ 飞版检测异常: {e}", "warning")
            
            # Phase 5: V3.8 曝光检测（在鸟的裁剪区域上执行）
            is_overexposed = False
            is_underexposed = False
            if self.settings.detect_exposure and detected and bird_crop_bgr is not None and bird_crop_bgr.size > 0:
                try:
                    exposure_detector = get_exposure_detector()
                    exposure_result = exposure_detector.detect(
                        bird_crop_bgr, 
                        threshold=self.settings.exposure_threshold
                    )
                    is_overexposed = exposure_result.is_overexposed
                    is_underexposed = exposure_result.is_underexposed
                except Exception as e:
                    pass  # 曝光检测失败不影响处理
            
            # V3.8: 飞版加成（仅当 confidence >= 0.5 且 is_flying 时）
            # 锐度+100，美学+0.5，加成后的值用于评分
            rating_sharpness = head_sharpness
            rating_topiq = topiq
            if is_flying and confidence >= 0.5:
                rating_sharpness = head_sharpness + 100
                if topiq is not None:
                    rating_topiq = topiq + 0.5
            
            # V4.0 优化: 先计算初步评分（不考虑对焦），只对 1 星以上做对焦检测
            # 这样 0 星和 -1 星照片不需要调用 exiftool，节省大量时间
            preliminary_result = self.rating_engine.calculate(
                detected=detected,
                confidence=confidence,
                sharpness=head_sharpness,   # V4.0: 原始锐度（飞鸟加成在引擎内）
                topiq=topiq,                # V4.0: 原始美学（飞鸟加成在引擎内）
                all_keypoints_hidden=all_keypoints_hidden,
                best_eye_visibility=best_eye_visibility,
                is_overexposed=is_overexposed,
                is_underexposed=is_underexposed,
                focus_sharpness_weight=1.0,  # 初步评分不考虑对焦
                focus_topiq_weight=1.0,
                is_flying=False,             # 初步评分不考虑飞鸟加成
            )
            
            # Phase 6: V4.0 对焦点验证（仅对 1 星以上照片）
            # 4 层检测返回两个权重: 锐度权重 + 美学权重
            focus_sharpness_weight = 1.0  # 默认无影响
            focus_topiq_weight = 1.0      # 默认无影响
            focus_x, focus_y = None, None
            
            # 只对 1 星以上照片做对焦检测（0 星和 -1 星跳过，节省时间）
            if preliminary_result.rating >= 1:
                if detected and bird_bbox is not None and img_dims is not None:
                    if file_prefix in raw_dict:
                        raw_ext = raw_dict[file_prefix]
                        raw_path = os.path.join(self.dir_path, file_prefix + raw_ext)
                        # Nikon, Sony, Canon, Olympus, Fujifilm, Panasonic 全支持
                        if raw_ext.lower() in ['.nef', '.nrw', '.arw', '.cr3', '.cr2', '.orf', '.raf', '.rw2']:
                            try:
                                focus_detector = get_focus_detector()
                                focus_result = focus_detector.detect(raw_path)
                                if focus_result is not None:
                                    # V3.9 修复：使用原图尺寸而非 resize 后的 img_dims
                                    # head_center_orig 和 bird_mask_orig 都是原图坐标系
                                    orig_dims = (w_orig, h_orig) if 'w_orig' in dir() and 'h_orig' in dir() else img_dims
                                    # V4.0: 返回元组 (锐度权重, 美学权重)
                                    focus_sharpness_weight, focus_topiq_weight = verify_focus_in_bbox(
                                        focus_result, 
                                        bird_bbox, 
                                        orig_dims,  # 使用原图尺寸！
                                        seg_mask=bird_mask_orig,
                                        head_center=head_center_orig,
                                        head_radius=head_radius_val,
                                    )
                                    focus_x, focus_y = focus_result.x, focus_result.y
                            except Exception as e:
                                pass  # 对焦检测失败不影响处理
            
            # V4.0: 最终评分计算（传入对焦权重和飞鸟状态）
            # 注意: 现在总是重新计算，因为需要传入 is_flying 参数
            rating_result = self.rating_engine.calculate(
                detected=detected,
                confidence=confidence,
                sharpness=head_sharpness,  # V4.0: 使用原始锐度，权重在引擎内应用
                topiq=topiq,              # V4.0: 使用原始美学，权重在引擎内应用
                all_keypoints_hidden=all_keypoints_hidden,
                best_eye_visibility=best_eye_visibility,
                is_overexposed=is_overexposed,
                is_underexposed=is_underexposed,
                focus_sharpness_weight=focus_sharpness_weight,  # V4.0: 锐度权重
                focus_topiq_weight=focus_topiq_weight,          # V4.0: 美学权重
                is_flying=is_flying,                            # V4.0: 飞鸟乘法加成
            )
            
            rating_value = rating_result.rating
            pick = rating_result.pick
            reason = rating_result.reason
            
            # V4.0: 根据 focus_sharpness_weight 计算对焦状态文本
            # 只有检测到鸟才设置对焦状态，避免无鸟照片也写入
            focus_status = None
            focus_status_en = None  # 英文版本用于调试图（避免中文字体问题）
            if detected:  # 只有检测到鸟才计算对焦状态
                if focus_sharpness_weight > 1.0:
                    focus_status = "精准"
                    focus_status_en = "BEST"
                elif focus_sharpness_weight >= 1.0:
                    focus_status = "鸟身"
                    focus_status_en = "GOOD"
                elif focus_sharpness_weight >= 0.7:
                    focus_status = "偏移"
                    focus_status_en = "BAD"
                elif focus_sharpness_weight < 0.7:
                    focus_status = "脱焦"
                    focus_status_en = "WORST"
            
            # V3.9: 生成调试可视化图（仅对有鸟的照片）
            if detected and bird_crop_bgr is not None:
                # 计算裁剪区域内的坐标
                head_center_crop = None
                if head_center_orig is not None:
                    # 转换到裁剪区域坐标
                    head_center_crop = (head_center_orig[0] - x_orig, head_center_orig[1] - y_orig)
                
                focus_point_crop = None
                if focus_x is not None and focus_y is not None:
                    # 对焦点从归一化坐标转换为裁剪区域坐标
                    # 使用原图尺寸 (w_orig, h_orig) 而不是 resize 后的 img_dims
                    if 'w_orig' in dir() and 'h_orig' in dir():
                        fx_px = int(focus_x * w_orig) - x_orig
                        fy_px = int(focus_y * h_orig) - y_orig
                        focus_point_crop = (fx_px, fy_px)
                
                try:
                    self._save_debug_crop(
                        filename,
                        bird_crop_bgr,
                        bird_crop_mask if 'bird_crop_mask' in dir() else None,
                        head_center_crop,
                        head_radius_val,
                        focus_point_crop,
                        focus_status_en  # 使用英文标签
                    )
                except Exception as e:
                    pass  # 调试图生成失败不影响主流程
            
            # 计算真正总耗时并输出简化日志
            photo_time_ms = (time.time() - photo_start_time) * 1000
            has_exposure_issue = is_overexposed or is_underexposed
            self._log_photo_result_simple(i, total_files, filename, rating_value, reason, photo_time_ms, is_flying, has_exposure_issue, focus_status)
            
            # 记录统计
            self._update_stats(rating_value, is_flying, has_exposure_issue)
            
            # 记录处理时间统计
            self.stats['photo_times'].append((filename, photo_time_ms, detected))
            if detected:
                self.stats['with_bird_times'].append(photo_time_ms)
            else:
                self.stats['no_bird_times'].append(photo_time_ms)
            
            # V3.4: 确定要处理的目标文件（RAW 优先，没有则用 JPEG/HEIF）
            # 注意：对于 HEIF/HEIC/HIF 文件，虽然 AI 推理时使用了临时 JPG，
            # 但 EXIF 元数据会写入原始文件（filepath 始终指向原始文件）
            target_file_path = None
            target_extension = None
            
            if file_prefix in raw_dict:
                # 有对应的 RAW 文件
                raw_extension = raw_dict[file_prefix]
                target_file_path = os.path.join(self.dir_path, file_prefix + raw_extension)
                target_extension = raw_extension
                
                # 写入 EXIF（仅限 RAW 文件）
                if os.path.exists(target_file_path):
                    # V4.0: 标签逻辑 - 飞鸟绿色优先，头部对焦红色
                    label = None
                    if is_flying:
                        label = 'Green'
                    elif focus_sharpness_weight > 1.0:  # 头部对焦 (1.1)
                        label = 'Red'
                    
                    # V4.0: 构建详细评分说明
                    caption_parts = []
                    caption_parts.append(f"[SuperPicky V4.0 评分报告]")
                    caption_parts.append(f"最终评分: {rating_value}星 | {reason}")
                    caption_parts.append("")
                    
                    # 原始数据
                    caption_parts.append("[原始检测数据]")
                    caption_parts.append(f"AI置信度: {confidence:.0%}")
                    caption_parts.append(f"头部锐度: {head_sharpness:.2f}" if head_sharpness else "头部锐度: 无法计算")
                    caption_parts.append(f"TOPIQ美学: {topiq:.2f}" if topiq else "TOPIQ美学: 未计算")
                    caption_parts.append(f"眼睛可见度: {best_eye_visibility:.0%}")
                    caption_parts.append("")
                    
                    # 修正因子
                    caption_parts.append("[修正因子]")
                    caption_parts.append(f"对焦锐度权重: {focus_sharpness_weight:.2f}")
                    caption_parts.append(f"对焦美学权重: {focus_topiq_weight:.2f}")
                    caption_parts.append(f"是否飞鸟: {'是 (锐度×1.2, 美学×1.1)' if is_flying else '否'}")
                    caption_parts.append("")
                    
                    # 调整后数值
                    caption_parts.append("[调整后数值]")
                    adj_sharpness = head_sharpness * focus_sharpness_weight if head_sharpness else 0
                    if is_flying and head_sharpness:
                        adj_sharpness = adj_sharpness * 1.2
                    caption_parts.append(f"调整后锐度: {adj_sharpness:.2f} (阈值400)")
                    
                    if topiq:
                        adj_topiq = topiq * focus_topiq_weight
                        if is_flying:
                            adj_topiq = adj_topiq * 1.1
                        caption_parts.append(f"调整后美学: {adj_topiq:.2f} (阈值5.0)")
                    caption_parts.append("")
                    
                    # 渐进可见度
                    visibility_weight = max(0.5, min(1.0, best_eye_visibility * 2))
                    caption_parts.append(f"[可见度降权]")
                    caption_parts.append(f"可见度权重: {visibility_weight:.2f}")
                    caption_parts.append(f"公式: max(0.5, min(1.0, {best_eye_visibility:.2f}×2))")
                    
                    caption = " | ".join(caption_parts)
                    
                    single_batch = [{
                        'file': target_file_path,
                        'rating': rating_value if rating_value >= 0 else 0,
                        'pick': pick,
                        'sharpness': head_sharpness,
                        'nima_score': topiq,  # V3.8: 实际是 TOPIQ 分数
                        'label': label,
                        'focus_status': focus_status,  # V3.9: 对焦状态写入 Country 字段
                        'caption': caption,  # V4.0: 详细评分说明
                    }]
                    exiftool_mgr.batch_set_metadata(single_batch)
            else:
                # V3.4: 纯 JPEG/HEIF 文件（没有对应 RAW）
                # 注意：filepath 始终是原始文件路径（如 .hif），即使 AI 推理时使用了临时 JPG
                target_file_path = filepath  # 使用原始文件路径（HIF/HEIF/HEIC/JPG）
                target_extension = os.path.splitext(filename)[1]
            
            # V3.4: 以下操作对 RAW 和纯 JPEG 都执行
            if target_file_path and os.path.exists(target_file_path):
                # 更新 CSV 中的关键点数据（V3.9: 添加对焦状态和坐标）
                self._update_csv_keypoint_data(
                    file_prefix, 
                    rating_sharpness,  # 使用加成后的锐度
                    has_visible_eye, 
                    has_visible_beak,
                    left_eye_vis,
                    right_eye_vis,
                    beak_vis,
                    rating_topiq,  # V3.8: 改为 rating_topiq
                    rating_value,
                    is_flying,
                    flight_confidence,
                    focus_status,  # V3.9: 对焦状态
                    focus_x,  # V3.9: 对焦点X坐标
                    focus_y   # V3.9: 对焦点Y坐标
                )
                
                # 收集3星照片（V3.8: 使用加成后的值）
                if rating_value == 3 and rating_topiq is not None:
                    self.star_3_photos.append({
                        'file': target_file_path,
                        'nima': rating_topiq,  # V3.8: 实际是 TOPIQ，保留字段名兼容
                        'sharpness': rating_sharpness  # 加成后的锐度
                    })
                
                # 记录评分（用于文件移动）
                self.file_ratings[file_prefix] = rating_value
                
                # 记录2星原因（用于分目录）（V3.8: 使用加成后的值）
                if rating_value == 2:
                    sharpness_ok = rating_sharpness >= self.settings.sharpness_threshold
                    topiq_ok = rating_topiq is not None and rating_topiq >= self.settings.nima_threshold
                    if sharpness_ok and not topiq_ok:
                        self.star2_reasons[file_prefix] = 'sharpness'
                    elif topiq_ok and not sharpness_ok:
                        self.star2_reasons[file_prefix] = 'nima'  # 保留原字段名兼容
                    else:
                        self.star2_reasons[file_prefix] = 'both'
    
    def _process_images_parallel(self, files_tbr, raw_dict, model, ui_settings,
                                 use_keypoints, keypoint_detector, use_flight,
                                 flight_detector, exiftool_mgr, actual_device,
                                 is_cpu, is_gpu):
        """并行处理图片（CPU线程池或GPU队列）"""
        total_files = len(files_tbr)
        
        if is_cpu:
            # CPU: 使用线程池并行
            import multiprocessing
            cpu_threads = self.settings.cpu_threads if hasattr(self.settings, 'cpu_threads') else 0
            if cpu_threads == 0:
                cpu_threads = multiprocessing.cpu_count()
            
            # 准备任务列表
            tasks = [(i, filename) for i, filename in enumerate(files_tbr, 1)]
            
            # 使用线程池处理
            with ThreadPoolExecutor(max_workers=cpu_threads) as executor:
                futures = {
                    executor.submit(
                        self._process_single_image,
                        i, filename, total_files, raw_dict, model, ui_settings,
                        use_keypoints, keypoint_detector, use_flight,
                        flight_detector, exiftool_mgr
                    ): (i, filename)
                    for i, filename in tasks
                }
                
                completed = 0
                for future in as_completed(futures):
                    if self.stop_event and self.stop_event.is_set():
                        break
                    try:
                        future.result()  # 获取结果（可能抛出异常）
                        completed += 1
                        if completed % 5 == 0 or completed == total_files:
                            progress = int((completed / total_files) * 100)
                            self._progress(progress)
                    except Exception as e:
                        i, filename = futures[future]
                        self._log(f"  ❌ 处理失败 {filename}: {e}", "error")
        
        elif is_gpu:
            # GPU: 使用队列控制并发（避免显存溢出）
            gpu_concurrent = self.settings.gpu_concurrent if hasattr(self.settings, 'gpu_concurrent') else 1
            import queue
            import threading
            
            task_queue = queue.Queue()
            for i, filename in enumerate(files_tbr, 1):
                task_queue.put((i, filename))
            
            # 使用信号量控制并发数
            semaphore = threading.Semaphore(gpu_concurrent)
            results_lock = threading.Lock()
            completed_count = [0]  # 使用列表以便在线程间共享
            
            def worker():
                while True:
                    if self.stop_event and self.stop_event.is_set():
                        break
                    try:
                        i, filename = task_queue.get_nowait()
                    except queue.Empty:
                        break
                    
                    with semaphore:  # 控制并发数
                        try:
                            self._process_single_image(
                                i, filename, total_files, raw_dict, model, ui_settings,
                                use_keypoints, keypoint_detector, use_flight,
                                flight_detector, exiftool_mgr
                            )
                            with results_lock:
                                completed_count[0] += 1
                                if completed_count[0] % 5 == 0 or completed_count[0] == total_files:
                                    progress = int((completed_count[0] / total_files) * 100)
                                    self._progress(progress)
                        except Exception as e:
                            self._log(f"  ❌ 处理失败 {filename}: {e}", "error")
                    task_queue.task_done()
            
            # 启动工作线程（每个并发任务一个线程）
            threads = []
            for _ in range(gpu_concurrent):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)
            
            # 等待所有任务完成
            task_queue.join()
            for t in threads:
                t.join()
    
    def _process_single_image(self, i, filename, total_files, raw_dict, model, ui_settings,
                             use_keypoints, keypoint_detector, use_flight,
                             flight_detector, exiftool_mgr):
        """处理单张图片（用于并行处理）"""
        # 检查是否已取消
        if self.stop_event and self.stop_event.is_set():
            return
        
        # 记录每张照片的开始时间
        photo_start_time = time.time()
        
        filepath = os.path.join(self.dir_path, filename)
        file_prefix, _ = os.path.splitext(filename)
        
        # 优化流程：YOLO → 关键点检测(在crop上) → 条件NIMA
        # Phase 1: 先做YOLO检测（跳过NIMA），获取鸟的位置和bbox
        try:
            result = detect_and_draw_birds(
                filepath, model, None, self.dir_path, ui_settings, None, skip_nima=True
            )
            if result is None:
                return
        except Exception as e:
            return
        
        # 解构 AI 结果
        detected, _, confidence, sharpness, _, bird_bbox, img_dims, bird_mask = result
        
        # Phase 2: 关键点检测（在裁剪区域上执行，更准确）
        all_keypoints_hidden = False
        best_eye_visibility = 0.0
        head_sharpness = 0.0
        has_visible_eye = False
        has_visible_beak = False
        left_eye_vis = 0.0
        right_eye_vis = 0.0
        beak_vis = 0.0
        head_center_orig = None
        head_radius_val = None
        orig_img = None
        bird_crop_bgr = None
        bird_crop_mask = None
        bird_mask_orig = None
        
        if use_keypoints and detected and bird_bbox is not None and img_dims is not None:
            try:
                import cv2
                from utils import read_image
                orig_img = read_image(filepath)
                if orig_img is not None:
                    h_orig, w_orig = orig_img.shape[:2]
                    w_resized, h_resized = img_dims
                    scale_x = w_orig / w_resized
                    scale_y = h_orig / h_resized
                    x, y, w, h = bird_bbox
                    x_orig = max(0, min(int(x * scale_x), w_orig - 1))
                    y_orig = max(0, min(int(y * scale_y), h_orig - 1))
                    w_orig_box = min(int(w * scale_x), w_orig - x_orig)
                    h_orig_box = min(int(h * scale_y), h_orig - y_orig)
                    bird_crop_bgr = orig_img[y_orig:y_orig+h_orig_box, x_orig:x_orig+w_orig_box]
                    
                    if bird_mask is not None:
                        if bird_mask.shape[:2] != (h_orig, w_orig):
                            bird_mask_orig = cv2.resize(bird_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                        else:
                            bird_mask_orig = bird_mask
                        bird_crop_mask = bird_mask_orig[y_orig:y_orig+h_orig_box, x_orig:x_orig+w_orig_box]
                    
                    if bird_crop_bgr.size > 0:
                        crop_rgb = cv2.cvtColor(bird_crop_bgr, cv2.COLOR_BGR2RGB)
                        kp_result = keypoint_detector.detect(
                            crop_rgb, 
                            box=(x_orig, y_orig, w_orig_box, h_orig_box),
                            seg_mask=bird_crop_mask
                        )
                        if kp_result is not None:
                            all_keypoints_hidden = kp_result.all_keypoints_hidden
                            best_eye_visibility = kp_result.best_eye_visibility
                            has_visible_eye = kp_result.visible_eye is not None
                            has_visible_beak = kp_result.beak_vis >= 0.3
                            left_eye_vis = kp_result.left_eye_vis
                            right_eye_vis = kp_result.right_eye_vis
                            beak_vis = kp_result.beak_vis
                            head_sharpness = kp_result.head_sharpness
                            
                            ch, cw = bird_crop_bgr.shape[:2]
                            if left_eye_vis >= right_eye_vis and left_eye_vis >= 0.3:
                                eye_px = (int(kp_result.left_eye[0] * cw), int(kp_result.left_eye[1] * ch))
                            elif right_eye_vis >= 0.3:
                                eye_px = (int(kp_result.right_eye[0] * cw), int(kp_result.right_eye[1] * ch))
                            else:
                                eye_px = None
                            
                            if eye_px is not None:
                                head_center_orig = (eye_px[0] + x_orig, eye_px[1] + y_orig)
                                beak_px = (int(kp_result.beak[0] * cw), int(kp_result.beak[1] * ch))
                                if beak_vis >= 0.3:
                                    import math
                                    dist = math.sqrt((eye_px[0] - beak_px[0])**2 + (eye_px[1] - beak_px[1])**2)
                                    head_radius_val = int(dist * 1.2)
                                else:
                                    head_radius_val = int(max(cw, ch) * 0.15)
                                head_radius_val = max(20, min(head_radius_val, min(cw, ch) // 2))
            except Exception:
                pass
        
        # Phase 3: TOPIQ计算
        topiq = None
        if detected and not all_keypoints_hidden and best_eye_visibility >= 0.3:
            try:
                from iqa_scorer import get_iqa_scorer
                from utils import get_best_device
                device = get_best_device(self.settings.device if hasattr(self.settings, 'device') else 'auto')
                scorer = get_iqa_scorer(device=device)
                topiq = scorer.calculate_nima(filepath)
            except Exception:
                pass
        
        # Phase 4: 飞版检测
        is_flying = False
        flight_confidence = 0.0
        if use_flight and detected and bird_crop_bgr is not None and bird_crop_bgr.size > 0:
            try:
                flight_result = flight_detector.detect(bird_crop_bgr)
                is_flying = flight_result.is_flying
                flight_confidence = flight_result.confidence
            except Exception:
                pass
        
        # Phase 5: 曝光检测
        is_overexposed = False
        is_underexposed = False
        if self.settings.detect_exposure and detected and bird_crop_bgr is not None and bird_crop_bgr.size > 0:
            try:
                exposure_detector = get_exposure_detector()
                exposure_result = exposure_detector.detect(
                    bird_crop_bgr, 
                    threshold=self.settings.exposure_threshold
                )
                is_overexposed = exposure_result.is_overexposed
                is_underexposed = exposure_result.is_underexposed
            except Exception:
                pass
        
        # 飞版加成
        rating_sharpness = head_sharpness
        rating_topiq = topiq
        if is_flying and confidence >= 0.5:
            rating_sharpness = head_sharpness + 100
            if topiq is not None:
                rating_topiq = topiq + 0.5
        
        # 初步评分
        preliminary_result = self.rating_engine.calculate(
            detected=detected,
            confidence=confidence,
            sharpness=head_sharpness,
            topiq=topiq,
            all_keypoints_hidden=all_keypoints_hidden,
            best_eye_visibility=best_eye_visibility,
            is_overexposed=is_overexposed,
            is_underexposed=is_underexposed,
            focus_sharpness_weight=1.0,
            focus_topiq_weight=1.0,
            is_flying=False,
        )
        
        # 对焦点验证（仅对1星以上）
        focus_sharpness_weight = 1.0
        focus_topiq_weight = 1.0
        focus_x, focus_y = None, None
        
        if preliminary_result.rating >= 1:
            if detected and bird_bbox is not None and img_dims is not None:
                if file_prefix in raw_dict:
                    raw_ext = raw_dict[file_prefix]
                    raw_path = os.path.join(self.dir_path, file_prefix + raw_ext)
                    if raw_ext.lower() in ['.nef', '.nrw', '.arw', '.cr3', '.cr2', '.orf', '.raf', '.rw2']:
                        try:
                            focus_detector = get_focus_detector()
                            focus_result = focus_detector.detect(raw_path)
                            if focus_result is not None:
                                orig_dims = (w_orig, h_orig) if 'w_orig' in locals() and 'h_orig' in locals() else img_dims
                                focus_sharpness_weight, focus_topiq_weight = verify_focus_in_bbox(
                                    focus_result, 
                                    bird_bbox, 
                                    orig_dims,
                                    seg_mask=bird_mask_orig,
                                    head_center=head_center_orig,
                                    head_radius=head_radius_val,
                                )
                                focus_x, focus_y = focus_result.x, focus_result.y
                        except Exception:
                            pass
        
        # 最终评分
        rating_result = self.rating_engine.calculate(
            detected=detected,
            confidence=confidence,
            sharpness=head_sharpness,
            topiq=topiq,
            all_keypoints_hidden=all_keypoints_hidden,
            best_eye_visibility=best_eye_visibility,
            is_overexposed=is_overexposed,
            is_underexposed=is_underexposed,
            focus_sharpness_weight=focus_sharpness_weight,
            focus_topiq_weight=focus_topiq_weight,
            is_flying=is_flying,
        )
        
        rating_value = rating_result.rating
        pick = rating_result.pick
        reason = rating_result.reason
        
        # 对焦状态
        focus_status = None
        focus_status_en = None
        if detected:
            if focus_sharpness_weight > 1.0:
                focus_status = "精准"
                focus_status_en = "BEST"
            elif focus_sharpness_weight >= 1.0:
                focus_status = "鸟身"
                focus_status_en = "GOOD"
            elif focus_sharpness_weight >= 0.7:
                focus_status = "偏移"
                focus_status_en = "BAD"
            elif focus_sharpness_weight < 0.7:
                focus_status = "脱焦"
                focus_status_en = "WORST"
        
        # 计算耗时
        photo_time_ms = (time.time() - photo_start_time) * 1000
        has_exposure_issue = is_overexposed or is_underexposed
        
        # 线程安全地更新统计（需要初始化锁）
        if not hasattr(self, '_stats_lock'):
            import threading
            self._stats_lock = threading.Lock()
        
        with self._stats_lock:
            self._log_photo_result_simple(i, total_files, filename, rating_value, reason, photo_time_ms, is_flying, has_exposure_issue, focus_status)
            self._update_stats(rating_value, is_flying, has_exposure_issue)
            self.stats['photo_times'].append((filename, photo_time_ms, detected))
            if detected:
                self.stats['with_bird_times'].append(photo_time_ms)
            else:
                self.stats['no_bird_times'].append(photo_time_ms)
        
        # 确定目标文件
        target_file_path = None
        if file_prefix in raw_dict:
            raw_ext = raw_dict[file_prefix]
            target_file_path = os.path.join(self.dir_path, file_prefix + raw_ext)
            if os.path.exists(target_file_path):
                label = None
                if is_flying:
                    label = 'Green'
                elif focus_sharpness_weight > 1.0:
                    label = 'Red'
                
                caption_parts = [
                    f"[SuperPicky V4.0 评分报告]",
                    f"最终评分: {rating_value}星 | {reason}",
                    "",
                    "[原始检测数据]",
                    f"AI置信度: {confidence:.0%}",
                    f"头部锐度: {head_sharpness:.2f}" if head_sharpness else "头部锐度: 无法计算",
                    f"TOPIQ美学: {topiq:.2f}" if topiq else "TOPIQ美学: 未计算",
                    f"眼睛可见度: {best_eye_visibility:.0%}",
                    "",
                    "[修正因子]",
                    f"对焦锐度权重: {focus_sharpness_weight:.2f}",
                    f"对焦美学权重: {focus_topiq_weight:.2f}",
                    f"是否飞鸟: {'是 (锐度×1.2, 美学×1.1)' if is_flying else '否'}",
                ]
                caption = " | ".join(caption_parts)
                
                single_batch = [{
                    'file': target_file_path,
                    'rating': rating_value if rating_value >= 0 else 0,
                    'pick': pick,
                    'sharpness': head_sharpness,
                    'nima_score': topiq,
                    'label': label,
                    'focus_status': focus_status,
                    'caption': caption,
                }]
                exiftool_mgr.batch_set_metadata(single_batch)
        else:
            target_file_path = filepath
        
        # 更新CSV和记录评分（线程安全）
        if target_file_path and os.path.exists(target_file_path):
            with self._stats_lock:
                self._update_csv_keypoint_data(
                    file_prefix, 
                    rating_sharpness,
                    has_visible_eye, 
                    has_visible_beak,
                    left_eye_vis,
                    right_eye_vis,
                    beak_vis,
                    rating_topiq,
                    rating_value,
                    is_flying,
                    flight_confidence,
                    focus_status,
                    focus_x,
                    focus_y
                )
                
                if rating_value == 3 and rating_topiq is not None:
                    self.star_3_photos.append({
                        'file': target_file_path,
                        'nima': rating_topiq,
                        'sharpness': rating_sharpness
                    })
                
                self.file_ratings[file_prefix] = rating_value
                
                if rating_value == 2:
                    sharpness_ok = rating_sharpness >= self.settings.sharpness_threshold
                    topiq_ok = rating_topiq is not None and rating_topiq >= self.settings.nima_threshold
                    if sharpness_ok and not topiq_ok:
                        self.star2_reasons[file_prefix] = 'sharpness'
                    elif topiq_ok and not sharpness_ok:
                        self.star2_reasons[file_prefix] = 'nima'
                    else:
                        self.star2_reasons[file_prefix] = 'both'
    
    # 注意: _calculate_rating 方法已移至 core/rating_engine.py
    # 现在使用 self.rating_engine.calculate() 替代
    
    def _log_photo_result(
        self, 
        rating: int, 
        reason: str, 
        conf: float, 
        sharp: float, 
        nima: Optional[float]
    ):
        """记录照片处理结果（详细版，保留用于调试）"""
        iqa_text = ""
        if nima is not None:
            iqa_text += f", 美学:{nima:.2f}"
        
        if rating == 3:
            self._log(f"  ⭐⭐⭐ 优选照片 (AI:{conf:.2f}, 锐度:{sharp:.1f}{iqa_text})", "success")
        elif rating == 2:
            self._log(f"  ⭐⭐ 良好照片 (AI:{conf:.2f}, 锐度:{sharp:.1f}{iqa_text})", "info")
        elif rating == 1:
            self._log(f"  ⭐ 普通照片 (AI:{conf:.2f}, 锐度:{sharp:.1f}{iqa_text})", "warning")
        elif rating == 0:
            self._log(f"  普通照片 - {reason}", "warning")
        else:  # -1
            self._log(f"  ❌ 无鸟 - {reason}", "error")
    
    def _log_photo_result_simple(
        self,
        index: int,
        total: int,
        filename: str,
        rating: int,
        reason: str,
        time_ms: float,
        is_flying: bool = False,  # V3.4: 飞鸟标识
        has_exposure_issue: bool = False,  # V3.8: 曝光问题标识
        focus_status: str = None  # V3.9: 对焦状态
    ):
        """记录照片处理结果（简化版，单行输出）"""
        # 星级标识
        star_map = {3: "3星", 2: "2星", 1: "1星", 0: "0星", -1: "-1星"}
        star_text = star_map.get(rating, "?星")
        
        # V3.4: 飞鸟标识
        flight_tag = "【飞鸟】" if is_flying else ""
        
        # V3.8: 曝光问题标识
        exposure_tag = "【曝光】" if has_exposure_issue else ""
        
        # V3.9: 对焦状态标识
        focus_tag = ""
        if focus_status and focus_status != "鸟身":
            focus_tag = f"【{focus_status}】"
        
        # 简化原因显示（V3.9: 增加到35字符避免截断）
        reason_short = reason if len(reason) < 35 else reason[:32] + "..."
        
        # 时间格式化
        if time_ms >= 1000:
            time_text = f"{time_ms/1000:.1f}s"
        else:
            time_text = f"{time_ms:.0f}ms"
        
        # 输出简化格式
        self._log(f"[{index:03d}/{total}] {filename} | {star_text} ({reason_short}) {flight_tag}{exposure_tag}{focus_tag}| {time_text}")
    
    def _save_debug_crop(
        self,
        filename: str,
        bird_crop_bgr: np.ndarray,
        bird_crop_mask: np.ndarray = None,
        head_center_crop: tuple = None,
        head_radius: int = None,
        focus_point_crop: tuple = None,
        focus_status: str = None
    ):
        """
        V3.9: 保存调试可视化图片到 .superpicky/debug_crops/ 目录
        
        标注内容：
        - 🟢 绿色半透明: SEG mask 鸟身区域
        - 🔵 蓝色圆圈: 头部检测区域
        - 🔴 红色十字: 对焦点位置
        """
        import cv2
        
        # 创建调试目录
        debug_dir = os.path.join(self.dir_path, ".superpicky", "debug_crops")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 复制原图
        debug_img = bird_crop_bgr.copy()
        h, w = debug_img.shape[:2]
        
        # 1. 绘制 SEG mask（绿色半透明覆盖）
        if bird_crop_mask is not None and bird_crop_mask.shape[:2] == (h, w):
            green_overlay = np.zeros_like(debug_img)
            green_overlay[:] = (0, 255, 0)  # BGR 绿色
            mask_bool = bird_crop_mask > 0
            # 半透明叠加
            debug_img[mask_bool] = cv2.addWeighted(
                debug_img[mask_bool], 0.7,
                green_overlay[mask_bool], 0.3, 0
            )
        
        # 2. 绘制头部圆圈（蓝色）
        if head_center_crop is not None and head_radius is not None:
            cx, cy = head_center_crop
            cv2.circle(debug_img, (cx, cy), head_radius, (255, 0, 0), 2)  # 蓝色圆圈
            cv2.circle(debug_img, (cx, cy), 3, (255, 0, 0), -1)  # 圆心
        
        # 3. 绘制对焦点（红色十字）
        if focus_point_crop is not None:
            fx, fy = focus_point_crop
            cross_size = 15
            cv2.line(debug_img, (fx - cross_size, fy), (fx + cross_size, fy), (0, 0, 255), 2)
            cv2.line(debug_img, (fx, fy - cross_size), (fx, fy + cross_size), (0, 0, 255), 2)
        
        # 4. 添加状态文字
        if focus_status:
            cv2.putText(debug_img, focus_status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 保存调试图
        file_prefix = os.path.splitext(filename)[0]
        debug_path = os.path.join(debug_dir, f"{file_prefix}_debug.jpg")
        cv2.imwrite(debug_path, debug_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    def _update_stats(self, rating: int, is_flying: bool = False, has_exposure_issue: bool = False):
        """更新统计数据"""
        self.stats['total'] += 1
        if rating == 3:
            self.stats['star_3'] += 1
        elif rating == 2:
            self.stats['star_2'] += 1
        elif rating == 1:
            self.stats['star_1'] += 1  # 普通照片（合格）
        elif rating == 0:
            self.stats['star_0'] += 1  # 普通照片（问题）
        else:  # -1
            self.stats['no_bird'] += 1
        
        # V3.6: 统计飞鸟照片
        if is_flying:
            self.stats['flying'] += 1
        
        # V3.8: 统计曝光问题照片
        if has_exposure_issue:
            self.stats['exposure_issue'] += 1
    
    def _update_csv_keypoint_data(
        self, 
        filename: str, 
        head_sharpness: float,
        has_visible_eye: bool,
        has_visible_beak: bool,
        left_eye_vis: float,
        right_eye_vis: float,
        beak_vis: float,
        nima: float,
        rating: int,
        is_flying: bool = False,
        flight_confidence: float = 0.0,
        focus_status: str = None,  # V3.9: 对焦状态
        focus_x: float = None,  # V3.9: 对焦点X坐标
        focus_y: float = None   # V3.9: 对焦点Y坐标
    ):
        """更新CSV中的关键点数据和评分（V3.9: 添加对焦状态和坐标）"""
        import csv
        
        csv_path = os.path.join(self.dir_path, ".superpicky", "report.csv")
        if not os.path.exists(csv_path):
            return
        
        try:
            # 读取现有CSV
            rows = []
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames) if reader.fieldnames else []
                
                # V3.9: 如果没有对焦相关字段则添加
                if 'focus_status' not in fieldnames:
                    rating_idx = fieldnames.index('rating') if 'rating' in fieldnames else len(fieldnames)
                    fieldnames.insert(rating_idx + 1, 'focus_status')
                if 'focus_x' not in fieldnames:
                    focus_status_idx = fieldnames.index('focus_status') if 'focus_status' in fieldnames else len(fieldnames)
                    fieldnames.insert(focus_status_idx + 1, 'focus_x')
                if 'focus_y' not in fieldnames:
                    focus_x_idx = fieldnames.index('focus_x') if 'focus_x' in fieldnames else len(fieldnames)
                    fieldnames.insert(focus_x_idx + 1, 'focus_y')
                
                for row in reader:
                    if row.get('filename') == filename:
                        # V3.4: 使用英文字段名更新数据
                        row['head_sharp'] = f"{head_sharpness:.0f}" if head_sharpness > 0 else "-"
                        row['left_eye'] = f"{left_eye_vis:.2f}"
                        row['right_eye'] = f"{right_eye_vis:.2f}"
                        row['beak'] = f"{beak_vis:.2f}"
                        row['nima_score'] = f"{nima:.2f}" if nima is not None else "-"
                        # V3.4: 飞版检测字段
                        row['is_flying'] = "yes" if is_flying else "no"
                        row['flight_conf'] = f"{flight_confidence:.2f}"
                        row['rating'] = str(rating)
                        # V3.9: 对焦状态和坐标字段
                        row['focus_status'] = focus_status if focus_status else "-"
                        row['focus_x'] = f"{focus_x:.3f}" if focus_x is not None else "-"
                        row['focus_y'] = f"{focus_y:.3f}" if focus_y is not None else "-"
                    rows.append(row)
            
            # 写回CSV
            if fieldnames and rows:
                with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
        except Exception as e:
            self._log(f"  ⚠️  更新CSV失败: {e}", "warning")
    
    def _calculate_picked_flags(self):
        """计算精选旗标 - 3星照片中美学+锐度双排名交集"""
        if len(self.star_3_photos) == 0:
            self._log("\nℹ️  无3星照片，跳过精选旗标计算")
            return
        
        self._log(f"\n🎯 计算精选旗标 (共{len(self.star_3_photos)}张3星照片)...")
        top_percent = self.config.picked_top_percentage / 100.0
        top_count = max(1, int(len(self.star_3_photos) * top_percent))
        
        # 美学排序
        sorted_by_nima = sorted(self.star_3_photos, key=lambda x: x['nima'], reverse=True)
        nima_top_files = set([photo['file'] for photo in sorted_by_nima[:top_count]])
        
        # 锐度排序
        sorted_by_sharpness = sorted(self.star_3_photos, key=lambda x: x['sharpness'], reverse=True)
        sharpness_top_files = set([photo['file'] for photo in sorted_by_sharpness[:top_count]])
        
        # 交集
        picked_files = nima_top_files & sharpness_top_files
        
        if len(picked_files) > 0:
            self._log(f"  📌 美学Top{self.config.picked_top_percentage}%: {len(nima_top_files)}张")
            self._log(f"  📌 锐度Top{self.config.picked_top_percentage}%: {len(sharpness_top_files)}张")
            self._log(f"  ⭐ 双排名交集: {len(picked_files)}张 → 设为精选")
            
            # 调试：显示精选文件路径
            for file_path in picked_files:
                exists = os.path.exists(file_path)
                self._log(f"    🔍 精选: {os.path.basename(file_path)} (存在: {exists})")
            
            # 批量写入
            picked_batch = [{
                'file': file_path,
                'rating': 3,
                'pick': 1
            } for file_path in picked_files]
            
            exiftool_mgr = get_exiftool_manager()
            picked_stats = exiftool_mgr.batch_set_metadata(picked_batch)
            
            if picked_stats['failed'] == 0:
                self._log(f"  ✅ 精选旗标写入成功")
            else:
                self._log(f"  ⚠️  {picked_stats['failed']} 张精选旗标写入失败", "warning")
            
            self.stats['picked'] = len(picked_files) - picked_stats.get('failed', 0)
            # 保存精选文件集合，供后续使用
            self.picked_files = picked_files
        else:
            self._log(f"  ℹ️  双排名交集为空，未设置精选旗标")
            self.stats['picked'] = 0
            self.picked_files = set()
    
    def _move_files_to_rating_folders(self, raw_dict):
        """移动文件到分类文件夹（V3.4: 支持纯 JPEG）"""
        # 筛选需要移动的文件（包括所有星级，确保原目录为空）
        files_to_move = []
        for prefix, rating in self.file_ratings.items():
            if rating in [-1, 0, 1, 2, 3]:
                # V3.4: 优先使用 RAW，没有则使用 JPEG
                if prefix in raw_dict:
                    # 有对应的 RAW 文件
                    raw_ext = raw_dict[prefix]
                    file_path = os.path.join(self.dir_path, prefix + raw_ext)
                    if os.path.exists(file_path):
                        folder = RATING_FOLDER_NAMES.get(rating, "0星_放弃")
                        files_to_move.append({
                            'filename': prefix + raw_ext,
                            'rating': rating,
                            'folder': folder
                        })
                else:
                    # V3.4: 纯 JPEG/HEIF 文件（包括 HEIF/HEIC）
                    for jpg_ext in ['.jpg', '.jpeg', '.heif', '.heic', '.hif', '.JPG', '.JPEG', '.HEIF', '.heic', '.hif']:
                        jpg_path = os.path.join(self.dir_path, prefix + jpg_ext)
                        if os.path.exists(jpg_path):
                            folder = RATING_FOLDER_NAMES.get(rating, "0星_放弃")
                            files_to_move.append({
                                'filename': prefix + jpg_ext,
                                'rating': rating,
                                'folder': folder
                            })
                            break  # 找到就跳出
        
        if not files_to_move:
            self._log("\n📂 无需移动文件")
            return
        
        self._log(f"\n📂 移动 {len(files_to_move)} 张照片到分类文件夹...")
        
        # 创建文件夹（使用实际的目录名）
        folders_in_use = set(f['folder'] for f in files_to_move)
        for folder_name in folders_in_use:
            folder_path = os.path.join(self.dir_path, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                self._log(f"  📁 创建文件夹: {folder_name}/")
        
        # 移动文件
        moved_count = 0
        for file_info in files_to_move:
            src_path = os.path.join(self.dir_path, file_info['filename'])
            dst_folder = os.path.join(self.dir_path, file_info['folder'])
            dst_path = os.path.join(dst_folder, file_info['filename'])
            
            try:
                if os.path.exists(dst_path):
                    continue
                shutil.move(src_path, dst_path)
                moved_count += 1
            except Exception as e:
                self._log(f"  ⚠️  移动失败: {file_info['filename']} - {e}", "warning")
        
        # 生成manifest
        manifest = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "app_version": "Refactored-Core",
            "original_dir": self.dir_path,
            "folder_structure": RATING_FOLDER_NAMES,
            "files": files_to_move,
            "stats": {"total_moved": moved_count}
        }
        
        manifest_path = os.path.join(self.dir_path, ".superpicky_manifest.json")
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            self._log(f"  ✅ 已移动 {moved_count} 张照片")
            self._log(f"  📋 Manifest: .superpicky_manifest.json")
        except Exception as e:
            self._log(f"  ⚠️  保存manifest失败: {e}", "warning")
    
    def _cleanup_temp_files(self, files_tbr, raw_dict):
        """清理临时JPG文件或保留并写入EXIF"""
        if self.settings.keep_temp_jpg:
            self._log("\n💾 保留临时转换的JPG文件...")
            self._process_keep_temp_jpg(files_tbr)
        else:
            self._log("\n🧹 清理临时文件...")
            deleted_count = 0
            
            # 删除 RAW 转换的临时 JPG
            for filename in files_tbr:
                file_prefix, file_ext = os.path.splitext(filename)
                if file_prefix in raw_dict and file_ext.lower() in ['.jpg', '.jpeg']:
                    jpg_path = os.path.join(self.dir_path, filename)
                    try:
                        if os.path.exists(jpg_path):
                            os.remove(jpg_path)
                            deleted_count += 1
                    except Exception as e:
                        self._log(f"  ⚠️  删除失败 {filename}: {e}", "warning")
            
            # 删除 HEIF 转换的临时 JPG
            temp_dir = os.path.join(self.dir_path, '.superpicky', 'temp_jpg')
            if os.path.exists(temp_dir):
                # 如果heif_temp_map为空（流水线框架可能未同步），扫描临时目录
                if not self.heif_temp_map:
                    for temp_file in os.listdir(temp_dir):
                        if temp_file.endswith('_temp.jpg'):
                            temp_jpg_path = os.path.join(temp_dir, temp_file)
                            try:
                                if os.path.exists(temp_jpg_path):
                                    os.remove(temp_jpg_path)
                                    deleted_count += 1
                            except Exception as e:
                                self._log(f"  ⚠️  删除失败 {temp_file}: {e}", "warning")
                else:
                    # 使用映射删除
                    for temp_jpg_path in self.heif_temp_map.values():
                        try:
                            if os.path.exists(temp_jpg_path):
                                os.remove(temp_jpg_path)
                                deleted_count += 1
                        except Exception as e:
                            self._log(f"  ⚠️  删除失败 {os.path.basename(temp_jpg_path)}: {e}", "warning")
            
            if deleted_count > 0:
                self._log(f"  ✅ 已删除 {deleted_count} 个临时JPG文件")
            else:
                self._log(f"  ℹ️  无临时文件需清理")
    
    def _process_keep_temp_jpg(self, files_tbr):
        """处理保留的临时JPG文件：写入EXIF并移动到对应星级目录"""
        from exiftool_manager import get_exiftool_manager
        exiftool_mgr = get_exiftool_manager()
        
        processed_count = 0
        
        # 处理 HEIF 转换的临时 JPG
        # 如果heif_temp_map为空（流水线框架可能未同步），尝试从临时目录扫描
        if not self.heif_temp_map:
            temp_dir = os.path.join(self.dir_path, '.superpicky', 'temp_jpg')
            if os.path.exists(temp_dir):
                # 扫描临时目录，重建映射
                for temp_file in os.listdir(temp_dir):
                    if temp_file.endswith('_temp.jpg'):
                        # 从文件名提取原始文件名（去掉_temp.jpg后缀）
                        file_basename = temp_file[:-10]  # 去掉'_temp.jpg'
                        # 尝试匹配原始HEIF文件名
                        for filename in files_tbr:
                            file_prefix, ext = os.path.splitext(filename)
                            if file_prefix == file_basename and ext.lower() in ['.heif', '.heic', '.hif']:
                                temp_jpg_path = os.path.join(temp_dir, temp_file)
                                self.heif_temp_map[filename] = temp_jpg_path
                                break
        
        for original_filename, temp_jpg_path in self.heif_temp_map.items():
            if not os.path.exists(temp_jpg_path):
                continue
            
            # 获取原始文件的评分（从file_ratings中获取，不是从EXIF读取）
            file_prefix = os.path.splitext(original_filename)[0]
            rating = self.file_ratings.get(file_prefix, -1)
            
            if rating < 0:
                # 无评分，删除临时文件
                try:
                    os.remove(temp_jpg_path)
                except:
                    pass
                continue
            
            try:
                # 构建 JPG 文件名（去掉 _temp 后缀）
                jpg_filename = file_prefix + ".jpg"
                final_jpg_path = os.path.join(self.dir_path, jpg_filename)
                
                # 如果已存在同名文件，使用带时间戳的名称
                if os.path.exists(final_jpg_path):
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    jpg_filename = f"{file_prefix}_{timestamp}.jpg"
                    final_jpg_path = os.path.join(self.dir_path, jpg_filename)
                
                # 移动临时文件到最终位置
                import shutil
                shutil.move(temp_jpg_path, final_jpg_path)
                
                # 检查是否是精选照片
                # 通过检查原始文件路径是否在 picked_files 中
                original_file_path = os.path.join(self.dir_path, original_filename)
                is_picked = original_file_path in self.picked_files
                
                # 尝试从原始HEIF文件读取EXIF数据（如果已写入）
                # 如果读取失败，使用默认值
                sharpness = None
                nima_score = None
                focus_status = None
                caption = f"[SuperPicky] 从 {os.path.splitext(original_filename)[1]} 转换"
                
                try:
                    # 尝试从原始文件读取EXIF（如果已写入）
                    from exiftool_manager import ExifToolManager
                    if os.path.exists(original_file_path):
                        # 注意：这里只是尝试读取，如果失败就使用默认值
                        # 实际EXIF数据应该在原始HEIF文件中
                        pass
                except:
                    pass
                
                # 写入 EXIF 元数据到JPG文件
                batch_data = [{
                    'file': final_jpg_path,
                    'rating': rating if rating >= 0 else 0,
                    'pick': 1 if is_picked else 0,
                    'sharpness': sharpness,  # 可能为None
                    'nima_score': nima_score,  # 可能为None
                    'label': None,
                    'focus_status': focus_status,  # 可能为None
                    'caption': caption
                }]
                
                exiftool_mgr.batch_set_metadata(batch_data)
                
                # 移动到对应星级目录（按星级归档）
                folder = RATING_FOLDER_NAMES.get(rating, "0星_放弃")
                folder_path = os.path.join(self.dir_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                
                dst_path = os.path.join(folder_path, jpg_filename)
                if not os.path.exists(dst_path):
                    shutil.move(final_jpg_path, dst_path)
                    processed_count += 1
                    # 记录归档信息（可选，避免日志过多）
                    # self._log(f"  📁 已归档到 {folder}/: {jpg_filename}")
                else:
                    # 目标已存在，删除源文件
                    os.remove(final_jpg_path)
                    self._log(f"  ⚠️  目标文件已存在，跳过: {folder}/{jpg_filename}", "warning")
                
            except Exception as e:
                self._log(f"  ⚠️  处理临时JPG失败 {original_filename}: {e}", "warning")
        
        if processed_count > 0:
            self._log(f"  ✅ 已保留并归档 {processed_count} 个临时JPG文件到对应星级目录")
        else:
            self._log(f"  ℹ️  无临时JPG文件需保留")
    
    def get_pipeline_status(self):
        """
        获取流水线状态（供UI监控使用）
        
        Returns:
            dict: 包含转换、队列、推理三个管线的状态
        """
        if not hasattr(self, '_pipelines') or not self._pipelines:
            return {
                'conversion': {'workers': 0, 'active_jobs': []},
                'queue': {'size': 0, 'max_size': 100},
                'inference_gpu': {'workers': 0, 'active_jobs': []},
                'inference_cpu': {'workers': 0, 'active_jobs': []}
            }
        
        status = {
            'conversion': {'workers': 0, 'active_jobs': []},
            'queue': {'size': 0, 'max_size': 100},
            'inference_gpu': {'workers': 0, 'active_jobs': []},
            'inference_cpu': {'workers': 0, 'active_jobs': []}
        }
        
        # 遍历所有流水线，收集状态
        for pipeline in self._pipelines:
            for stage in pipeline.stages:
                stage_name = stage.name.lower()
                
                # 检查是否是转换阶段
                if 'heif' in stage_name or '转换' in stage_name:
                    workers = stage.max_workers
                    # 估算活跃任务数（基于队列统计）
                    if stage.input_queue:
                        queue_stats = stage.input_queue.get_stats()
                        active_count = min(workers, max(0, queue_stats.get('total_put', 0) - queue_stats.get('total_done', 0)))
                        active_jobs = [i < active_count for i in range(workers)]
                    else:
                        active_jobs = [False] * workers
                    status['conversion']['workers'] = max(status['conversion']['workers'], workers)
                    status['conversion']['active_jobs'] = active_jobs
                
                # 检查是否是推理阶段，区分GPU和CPU
                elif 'ai处理' in stage_name or '推理' in stage_name or 'inference' in stage_name:
                    workers = stage.max_workers
                    device = stage.device.lower()
                    # 估算活跃任务数
                    if stage.input_queue:
                        queue_stats = stage.input_queue.get_stats()
                        active_count = min(workers, max(0, queue_stats.get('total_put', 0) - queue_stats.get('total_done', 0)))
                        active_jobs = [i < active_count for i in range(workers)]
                    else:
                        active_jobs = [False] * workers
                    
                    # 根据设备类型分类
                    if 'cuda' in device or 'gpu' in device or 'mps' in device:
                        status['inference_gpu']['workers'] = max(status['inference_gpu']['workers'], workers)
                        status['inference_gpu']['active_jobs'] = active_jobs
                    else:  # CPU
                        status['inference_cpu']['workers'] = max(status['inference_cpu']['workers'], workers)
                        status['inference_cpu']['active_jobs'] = active_jobs
        
        # 获取共享队列大小
        if hasattr(self, '_shared_ai_queue') and self._shared_ai_queue:
            status['queue']['size'] = self._shared_ai_queue.qsize()
            # 估算最大队列大小
            queue_stats = self._shared_ai_queue.get_stats()
            status['queue']['max_size'] = max(100, queue_stats.get('total_put', 0))
        
        return status
