#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Job queue based photo processor."""

import os
import time
import threading
import shutil
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

from constants import RAW_EXTENSIONS, JPG_EXTENSIONS, RATING_FOLDER_NAMES
from core.job_queue import JobQueue
from core.pipeline_builder import ParallelPipelineBuilder
from core.photo_processor import PhotoProcessor, ProcessingResult


class JobQueuePhotoProcessor(PhotoProcessor):
    """PhotoProcessor variant that uses the job queue pipeline."""

    HEIF_EXTENSIONS = ['.heic', '.heif', '.hif']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._heif_map: Dict[str, str] = {}
        self._pipeline_stats: Dict[str, Any] = {}
        self._total_files_processed: int = 0

    def process(
        self,
        organize_files: bool = True,
        cleanup_temp: bool = False,
    ) -> ProcessingResult:
        start_time = time.time()
        self.stats['start_time'] = start_time

        raw_dict, jpg_dict, heif_files, files_tbr = self._scan_files_with_heif()

        raw_files_to_convert = self._identify_raws_to_convert(raw_dict, jpg_dict, files_tbr)
        if raw_files_to_convert:
            self._convert_raws(raw_files_to_convert, files_tbr)

        self._process_images_with_pipeline(files_tbr, heif_files, raw_dict)

        self._calculate_picked_flags()

        if organize_files:
            self._move_files_to_rating_folders(raw_dict)

        if cleanup_temp:
            self._cleanup_temp_files(files_tbr, raw_dict)
            self._cleanup_heif_temp_files()

        end_time = time.time()
        self.stats['end_time'] = end_time
        self.stats['total_time'] = end_time - start_time
        self.stats['avg_time'] = (
            self.stats['total_time'] / self.stats['total']
            if self.stats['total'] > 0 else 0
        )

        return ProcessingResult(
            stats=self.stats.copy(),
            file_ratings=self.file_ratings.copy(),
            star_3_photos=self.star_3_photos.copy(),
            total_time=self.stats['total_time'],
            avg_time=self.stats['avg_time'],
            pipeline_stats=self._pipeline_stats.copy(),
            total_files_processed=self._total_files_processed,
        )

    def _scan_files_with_heif(self) -> Tuple[dict, dict, list, list]:
        scan_start = time.time()

        raw_dict = {}
        jpg_dict = {}
        files_tbr: List[str] = []
        heif_files: List[Tuple[str, str]] = []
        heif_map: Dict[str, str] = {}

        for filename in os.listdir(self.dir_path):
            if filename.startswith('.'):
                continue
            file_prefix, file_ext = os.path.splitext(filename)
            ext_lower = file_ext.lower()
            if ext_lower in RAW_EXTENSIONS:
                raw_dict[file_prefix] = file_ext
                continue
            if ext_lower in JPG_EXTENSIONS:
                jpg_dict[file_prefix] = file_ext
                files_tbr.append(filename)

        for filename in os.listdir(self.dir_path):
            if filename.startswith('.'):
                continue
            file_prefix, file_ext = os.path.splitext(filename)
            ext_lower = file_ext.lower()
            if ext_lower not in self.HEIF_EXTENSIONS:
                continue
            if file_prefix in raw_dict or file_prefix in jpg_dict:
                continue
            filepath = os.path.join(self.dir_path, filename)
            heif_files.append((filename, filepath))
            heif_map[file_prefix] = filename

        scan_time = (time.time() - scan_start) * 1000
        self._log(
            f"Scan time: {scan_time:.1f}ms (jpg={len(files_tbr)}, heif={len(heif_files)}, raw={len(raw_dict)})"
        )

        self._heif_map = heif_map
        return raw_dict, jpg_dict, heif_files, files_tbr

    def _process_images_with_pipeline(
        self,
        files_tbr: List[str],
        heif_files: List[Tuple[str, str]],
        raw_dict: Dict[str, str],
    ) -> None:
        total_files = len(files_tbr) + len(heif_files)
        if total_files == 0:
            self._log("No files to process.")
            return

        if not hasattr(self.settings, 'device') or self.settings.device in (None, '', 'auto'):
            self.settings.device = 'all'
        if not hasattr(self.settings, 'gpu_concurrent') or not self.settings.gpu_concurrent:
            # 根据可用显存动态计算 gpu_concurrent: 可用显存(GB) / 1.2 - 1
            gpu_concurrent = 6  # 默认值
            try:
                import torch
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    # 在获取显存之前，先清理释放的显存缓存
                    # 这有助于释放之前线程分配但未正确释放的显存
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # 确保所有 CUDA 操作完成
                    
                    # 获取第一个 GPU 的可用显存（单位：字节）
                    free_memory_bytes, _ = torch.cuda.mem_get_info(0)
                    # 转换为 GB
                    free_memory_gb = free_memory_bytes / (1024 ** 3)
                    # 计算: 可用显存(GB) / 1.2 - 1
                    gpu_concurrent = int(free_memory_gb / 2 - 2)
                    # 确保至少为 1
                    gpu_concurrent = max(1, gpu_concurrent)
                    self._log(f"根据可用显存({free_memory_gb:.2f}GB)计算 gpu_concurrent={gpu_concurrent}")
            except Exception:
                # 如果获取显存失败，使用默认值
                pass
            self.settings.gpu_concurrent = gpu_concurrent

        results: List[Dict[str, Any]] = []
        results_lock = threading.Lock()
        processed_count = 0

        def stats_callback(result: Dict[str, Any]):
            nonlocal processed_count
            filename = result.get('filename', 'unknown')
            device_tag = result.get('device')
            if device_tag:
                filename = f"[{str(device_tag).upper()}] {filename}"
            rating = result.get('rating', 0)
            reason = result.get('reason', '')
            processing_time = result.get('processing_time', 0.0)
            is_flying = result.get('is_flying', False)
            focus_status = result.get('focus_status')
            has_exposure_issue = bool(result.get('is_overexposed')) or bool(result.get('is_underexposed'))
            with results_lock:
                results.append(result)
                processed_count += 1
                progress = int((processed_count / total_files) * 100)
                index = processed_count
            self._progress(progress)
            self._log_photo_result_simple(
                index=index,
                total=total_files,
                filename=filename,
                rating=rating,
                reason=reason,
                time_ms=processing_time * 1000,
                is_flying=is_flying,
                has_exposure_issue=has_exposure_issue,
                focus_status=focus_status,
            )

        def progress_callback(value: int):
            if value is None or value < 0:
                return
            self._progress(int(value))

        builder = ParallelPipelineBuilder(
            dir_path=self.dir_path,
            settings=self.settings,
            raw_dict=raw_dict,
            log_callback=self.callbacks.log,
            progress_callback=progress_callback,
            stats_callback=stats_callback,
        )

        shared_ai_queue = JobQueue()
        heif_pipeline = None
        if heif_files:
            heif_pipeline = builder.build_heif_conversion_stage(heif_files, shared_ai_queue)

        ai_pipeline = builder.build_unified_ai_processing_pipeline(files_tbr, shared_ai_queue)

        if heif_pipeline:
            heif_pipeline.start()
        ai_pipeline.start()

        if heif_pipeline:
            heif_pipeline.wait_complete()
        ai_pipeline.wait_complete()

        # wait_complete 已经输出了"所有任务已完成"和"正在停止流水线..."
        # 所以 stop 时跳过这些日志，避免重复
        if heif_pipeline:
            heif_pipeline.stop(skip_log=True)
        ai_pipeline.stop(skip_log=True)

        # 收集各阶段的统计信息（保存到实例变量，稍后输出）
        self._pipeline_stats = {}
        if heif_pipeline:
            heif_stats = heif_pipeline.get_stats()
            for stage_name, stage_stats in heif_stats.items():
                self._pipeline_stats[stage_name] = stage_stats
        
        ai_stats = ai_pipeline.get_stats()
        for stage_name, stage_stats in ai_stats.items():
            self._pipeline_stats[stage_name] = stage_stats
        
        self._total_files_processed = total_files

        with results_lock:
            results_snapshot = list(results)

        for result in results_snapshot:
            self._apply_pipeline_result(result, raw_dict)

    def _apply_pipeline_result(self, result: Dict[str, Any], raw_dict: Dict[str, str]) -> None:
        filename = result.get('filename')
        file_prefix = result.get('file_prefix')
        if not file_prefix and filename:
            file_prefix, _ = os.path.splitext(filename)

        if not file_prefix:
            return

        rating_value = result.get('rating', 0)
        is_flying = result.get('is_flying', False)
        has_exposure_issue = bool(result.get('is_overexposed')) or bool(result.get('is_underexposed'))

        self._update_stats(rating_value, is_flying, has_exposure_issue)
        self.file_ratings[file_prefix] = rating_value

        rating_sharpness = result.get('rating_sharpness', result.get('head_sharpness', 0.0))
        rating_topiq = result.get('rating_topiq', result.get('topiq'))

        if rating_value == 2:
            sharpness_ok = rating_sharpness >= self.settings.sharpness_threshold
            topiq_ok = rating_topiq is not None and rating_topiq >= self.settings.nima_threshold
            if sharpness_ok and not topiq_ok:
                self.star2_reasons[file_prefix] = 'sharpness'
            elif topiq_ok and not sharpness_ok:
                self.star2_reasons[file_prefix] = 'nima'
            else:
                self.star2_reasons[file_prefix] = 'both'

        target_file_path = None
        if file_prefix in raw_dict:
            raw_ext = raw_dict[file_prefix]
            target_file_path = os.path.join(self.dir_path, file_prefix + raw_ext)
        elif filename:
            target_file_path = os.path.join(self.dir_path, filename)

        if not target_file_path or not os.path.exists(target_file_path):
            return

        head_sharpness = result.get('head_sharpness', 0.0)
        topiq = result.get('topiq')
        focus_sharpness_weight = result.get('focus_sharpness_weight', 1.0)
        focus_topiq_weight = result.get('focus_topiq_weight', 1.0)

        adj_sharpness = head_sharpness * focus_sharpness_weight if head_sharpness else 0
        if is_flying and head_sharpness:
            adj_sharpness *= 1.2
        adj_topiq = topiq * focus_topiq_weight if topiq is not None else None
        if is_flying and adj_topiq is not None:
            adj_topiq *= 1.1

        self._update_csv_keypoint_data(
            file_prefix,
            head_sharpness,
            result.get('has_visible_eye', False),
            result.get('has_visible_beak', False),
            result.get('left_eye_vis', 0.0),
            result.get('right_eye_vis', 0.0),
            result.get('beak_vis', 0.0),
            topiq,
            rating_value,
            is_flying,
            result.get('flight_confidence', 0.0),
            result.get('focus_status'),
            result.get('focus_x'),
            result.get('focus_y'),
            adj_sharpness,
            adj_topiq,
        )

        if rating_value == 3 and adj_topiq is not None:
            self.star_3_photos.append({
                'file': target_file_path,
                'nima': adj_topiq,
                'sharpness': adj_sharpness,
            })

    def _move_files_to_rating_folders(self, raw_dict: Dict[str, str]) -> None:
        files_to_move = []
        temp_dir = os.path.join(self.dir_path, '.superpicky', 'temp_jpg')

        for prefix, rating in self.file_ratings.items():
            if rating not in [-1, 0, 1, 2, 3]:
                continue

            if prefix in raw_dict:
                raw_ext = raw_dict[prefix]
                file_path = os.path.join(self.dir_path, prefix + raw_ext)
                if os.path.exists(file_path):
                    folder = RATING_FOLDER_NAMES.get(rating, RATING_FOLDER_NAMES.get(0, "0_star_discard"))
                    files_to_move.append({
                        'filename': prefix + raw_ext,
                        'rating': rating,
                        'folder': folder,
                    })
                continue

            moved = False
            for jpg_ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                jpg_path = os.path.join(self.dir_path, prefix + jpg_ext)
                if os.path.exists(jpg_path):
                    folder = RATING_FOLDER_NAMES.get(rating, RATING_FOLDER_NAMES.get(0, "0_star_discard"))
                    files_to_move.append({
                        'filename': prefix + jpg_ext,
                        'rating': rating,
                        'folder': folder,
                    })
                    moved = True
                    break

            if moved:
                continue

            heif_name = self._heif_map.get(prefix)
            if heif_name:
                heif_path = os.path.join(self.dir_path, heif_name)
                if os.path.exists(heif_path):
                    folder = RATING_FOLDER_NAMES.get(rating, RATING_FOLDER_NAMES.get(0, "0_star_discard"))
                    files_to_move.append({
                        'filename': heif_name,
                        'rating': rating,
                        'folder': folder,
                    })

                temp_name = f"{prefix}.jpg"
                temp_path = os.path.join(temp_dir, temp_name)
                if os.path.exists(temp_path):
                    folder = RATING_FOLDER_NAMES.get(rating, RATING_FOLDER_NAMES.get(0, "0_star_discard"))
                    files_to_move.append({
                        'filename': temp_name,
                        'rating': rating,
                        'folder': folder,
                        'source_path': temp_path,
                    })

        if not files_to_move:
            self._log("No files to move.")
            return

        self._log(f"Moving {len(files_to_move)} files into rating folders...")

        folders_in_use = set(item['folder'] for item in files_to_move)
        for folder_name in folders_in_use:
            folder_path = os.path.join(self.dir_path, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                self._log(f"Created folder {folder_name}/")

        moved_count = 0
        for file_info in files_to_move:
            src_path = file_info.get('source_path') or os.path.join(self.dir_path, file_info['filename'])
            dst_folder = os.path.join(self.dir_path, file_info['folder'])
            dst_path = os.path.join(dst_folder, file_info['filename'])
            try:
                if os.path.exists(dst_path):
                    continue
                shutil.move(src_path, dst_path)
                moved_count += 1
            except Exception as exc:
                self._log(f"Move failed: {file_info['filename']} - {exc}", "warning")

        manifest = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "app_version": "JobQueue",
            "original_dir": self.dir_path,
            "folder_structure": RATING_FOLDER_NAMES,
            "files": files_to_move,
            "stats": {"total_moved": moved_count},
        }

        manifest_path = os.path.join(self.dir_path, ".superpicky_manifest.json")
        try:
            with open(manifest_path, 'w', encoding='utf-8') as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
            self._log(f"Moved {moved_count} files.")
            self._log("Manifest: .superpicky_manifest.json")
        except Exception as exc:
            self._log(f"Manifest save failed: {exc}", "warning")

    def _cleanup_heif_temp_files(self) -> None:
        if not self._heif_map:
            return

        temp_dir = os.path.join(self.dir_path, '.superpicky', 'temp_jpg')
        if not os.path.isdir(temp_dir):
            return

        deleted = 0
        for prefix in self._heif_map:
            temp_path = os.path.join(temp_dir, f"{prefix}.jpg")
            if not os.path.exists(temp_path):
                continue
            try:
                os.remove(temp_path)
                deleted += 1
            except Exception as exc:
                self._log(f"Temp cleanup failed: {os.path.basename(temp_path)} - {exc}", "warning")

        if deleted > 0:
            self._log(f"Removed {deleted} HEIF temp files.")
    
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
