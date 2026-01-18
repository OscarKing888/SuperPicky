#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Processor - 命令行处理器
简化版 - 调用核心 PhotoProcessor
"""

from typing import Dict, List, Any
from core.photo_processor import (
    PhotoProcessor,
    ProcessingSettings,
    ProcessingCallbacks,
    ProcessingResult
)
from core.config_manager import UISettings
from utils import log_message


class CLIProcessor:
    """CLI 处理器 - 只负责命令行交互"""
    
    def __init__(self, dir_path: str, ui_settings = None, verbose: bool = True, detect_flight: bool = True):
        """
        初始化处理器
        
        Args:
            dir_path: 处理目录
            ui_settings: UISettings 实例或列表（向后兼容）
            verbose: 详细输出
            detect_flight: 是否启用飞鸟检测
        """
        self.verbose = verbose
        self.dir_path = dir_path  # 保存目录路径用于日志
        
        # V3.9.4: 修正默认值，与 GUI 保持完全一致
        # GUI 默认: sharpness=400, nima=5.0, exposure=True, burst=True
        if ui_settings is None:
            ui_settings = UISettings(
                ai_confidence=50,
                sharpness_threshold=400,
                nima_threshold=5.0,
                save_crop=False,
                normalization_mode='log_compression',
                detect_flight=detect_flight,
                detect_exposure=True,   # V3.9.4: 默认开启曝光检测，与 GUI 一致
                detect_burst=True       # V3.9.4: 默认开启连拍检测，与 GUI 一致
            )
        elif isinstance(ui_settings, (list, tuple)):
            # 向后兼容列表格式
            ui_settings = UISettings(
                ai_confidence=ui_settings[0] if len(ui_settings) > 0 else 50,
                sharpness_threshold=ui_settings[1] if len(ui_settings) > 1 else 400,
                nima_threshold=ui_settings[2] if len(ui_settings) > 2 else 5.0,
                save_crop=ui_settings[3] if len(ui_settings) > 3 else False,
                normalization_mode=ui_settings[4] if len(ui_settings) > 4 else 'log_compression',
                detect_flight=detect_flight,
                detect_exposure=True,   # V3.9.4: 默认开启曝光检测，与 GUI 一致
                detect_burst=True       # V3.9.4: 默认开启连拍检测，与 GUI 一致
            )
        
        # 转换为 ProcessingSettings
        settings = ProcessingSettings(
            ai_confidence=ui_settings.ai_confidence,
            sharpness_threshold=ui_settings.sharpness_threshold,
            nima_threshold=ui_settings.nima_threshold,
            save_crop=ui_settings.save_crop,
            normalization_mode=ui_settings.normalization_mode,
            detect_flight=ui_settings.detect_flight,
            detect_exposure=ui_settings.detect_exposure,
            detect_burst=ui_settings.detect_burst
        )
        
        # 创建核心处理器
        self.processor = PhotoProcessor(
            dir_path=dir_path,
            settings=settings,
            callbacks=ProcessingCallbacks(
                log=self._log,
                progress=self._progress
            )
        )
    
    def _log(self, msg: str, level: str = "info"):
        """日志回调 - 带颜色输出并写入文件"""
        if not self.verbose:
            return
        
        # ANSI颜色代码
        colors = {
            "success": "\033[92m",  # 绿色
            "error": "\033[91m",    # 红色
            "warning": "\033[93m",  # 黄色
            "info": "\033[94m",     # 蓝色
            "reset": "\033[0m"
        }
        
        color = colors.get(level, "")
        reset = colors["reset"] if color else ""
        
        # 输出到终端（带颜色）
        print(f"{color}{msg}{reset}")
        
        # 同时写入日志文件（不带颜色，不重复打印）
        log_message(msg, self.dir_path, file_only=True)
    
    def _progress(self, percent: int):
        """进度回调 - CLI可选"""
        # CLI 模式下可以选择是否显示进度
        # 目前不显示，避免输出过多
        pass
    
    def process(self, organize_files: bool = True, cleanup_temp: bool = True) -> Dict:
        """
        主处理流程
        
        Args:
            organize_files: 是否移动文件到分类文件夹
            cleanup_temp: 是否清理临时JPG
            
        Returns:
            处理统计字典
        """
        # 打印横幅
        self._print_banner()
        
        # 调用核心处理器
        result = self.processor.process(
            organize_files=organize_files,
            cleanup_temp=cleanup_temp
        )
        
        # 打印摘要
        self._print_summary(result)
        
        return result.stats
    
    def _print_banner(self):
        """打印CLI横幅"""
        self._log("\n" + "="*60)
        self._log("🐦 SuperPicky CLI - 慧眼选鸟 (命令行版)")
        self._log("="*60 + "\n")
        
        self._log("📁 阶段1: 文件扫描", "info")
    
    def _print_summary(self, result: ProcessingResult):
        """打印完成摘要（使用共享格式化模块）"""
        from core.stats_formatter import format_processing_summary, print_summary
        
        lines = format_processing_summary(result.stats, include_time=True)
        print_summary(lines, self._log)
        
        # 在统计报告之后输出流水线耗时统计
        if hasattr(result, 'pipeline_stats') and result.pipeline_stats:
            self._log_pipeline_stats(result.pipeline_stats, result.total_files_processed)
    
    def _log_pipeline_stats(self, pipeline_stats: Dict[str, Any], total_files: int) -> None:
        """输出流水线各阶段的耗时统计（在统计报告之后）"""
        # 分类统计，按设备分开
        heif_time = 0.0
        heif_processed = 0
        
        # 按设备分开统计 AI 推理
        cpu_ai_time = 0.0
        cpu_processed = 0
        cuda_ai_time = 0.0
        cuda_processed = 0
        mps_ai_time = 0.0
        mps_processed = 0
        
        exif_time = 0.0
        exif_processed = 0
        
        for stage_name, stage_stats in pipeline_stats.items():
            total_time = stage_stats.get('total_time', 0.0)
            processed = stage_stats.get('processed', 0)
            
            if 'HEIF' in stage_name or 'heif' in stage_name.lower():
                heif_time += total_time
                heif_processed += processed
            elif 'EXIF' in stage_name or 'exif' in stage_name.lower():
                exif_time += total_time
                exif_processed += processed
            elif 'CPU-Hybrid' in stage_name:
                # CPUHybridStage 的名称是 "CPU-Hybrid"，需要单独处理
                # 使用 inference_time 而不是 total_time（total_time 包含转换时间）
                inference_time = stage_stats.get('inference_time', 0.0)
                inferred = stage_stats.get('inferred', 0)
                if inferred > 0:
                    cpu_ai_time += inference_time
                    cpu_processed += inferred
            elif 'AI处理' in stage_name:
                # 阶段名称格式: "AI处理-{device.upper()}"
                device = stage_name.split('-')[-1] if '-' in stage_name else ''
                device_upper = device.upper()
                if device_upper == 'CPU':
                    cpu_ai_time += total_time
                    cpu_processed += processed
                elif device_upper == 'CUDA':
                    cuda_ai_time += total_time
                    cuda_processed += processed
                elif device_upper == 'MPS':
                    mps_ai_time += total_time
                    mps_processed += processed
        
        # 计算 AI 检测总耗时（所有设备）
        ai_total_time = cpu_ai_time + cuda_ai_time + mps_ai_time
        
        # 输出统计信息（在"平均每张"之后，即使为0也显示）
        self._log("")
        self._log("⏱️  流水线耗时统计:")
        
        # HEIF转换（即使为0也显示）
        heif_avg = heif_time / heif_processed if heif_processed > 0 else 0
        self._log(f"  HEIF转换: {heif_time:.1f}秒 (平均 {heif_avg:.2f}秒/张, {heif_processed}张)")
        
        # AI推理按设备分开显示
        if cpu_processed > 0:
            cpu_avg = cpu_ai_time / cpu_processed if cpu_processed > 0 else 0
            self._log(f"  AI推理(CPU): {cpu_ai_time:.1f}秒 (平均 {cpu_avg:.2f}秒/张, {cpu_processed}张)")
        else:
            self._log(f"  AI推理(CPU): 0.0秒 (平均 0.00秒/张, 0张)")
        
        if cuda_processed > 0:
            cuda_avg = cuda_ai_time / cuda_processed if cuda_processed > 0 else 0
            self._log(f"  AI推理(CUDA): {cuda_ai_time:.1f}秒 (平均 {cuda_avg:.2f}秒/张, {cuda_processed}张)")
        else:
            self._log(f"  AI推理(CUDA): 0.0秒 (平均 0.00秒/张, 0张)")
        
        if mps_processed > 0:
            mps_avg = mps_ai_time / mps_processed if mps_processed > 0 else 0
            self._log(f"  AI推理(MPS): {mps_ai_time:.1f}秒 (平均 {mps_avg:.2f}秒/张, {mps_processed}张)")
        else:
            self._log(f"  AI推理(MPS): 0.0秒 (平均 0.00秒/张, 0张)")
        
        # EXIF写入（即使为0也显示）
        exif_avg = exif_time / exif_processed if exif_processed > 0 else 0
        self._log(f"  EXIF写入: {exif_time:.1f}秒 (平均 {exif_avg:.2f}秒/张, {exif_processed}张)")
        
        # 输出 AI 检测总耗时
        ai_avg = ai_total_time / total_files if total_files > 0 else 0
        self._log(f"⏱️  AI检测总耗时: {ai_total_time:.1f}秒 (平均 {ai_avg:.2f}秒/张)")
        self._log("")
