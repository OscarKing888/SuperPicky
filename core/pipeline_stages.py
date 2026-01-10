#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流水线阶段具体实现
包括：HEIF转换、RAW转换、AI推理、关键点检测、TOPIQ评分等
"""

import os
import time
import uuid
import threading
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image

from core.job_queue import PipelineStage, Job, JobQueue, JobStatus


class HEIFConversionStage(PipelineStage):
    """HEIF转换阶段 - 流水线式，转换一张立即传递给下一阶段"""
    
    def __init__(
        self,
        input_queue: JobQueue,
        output_queue: JobQueue,
        dir_path: str,
        max_workers: int = 2,  # 默认2个线程（推理慢10倍，转换线程应该更少）
        log_callback: Optional[callable] = None
    ):
        """
        初始化HEIF转换阶段
        
        Args:
            input_queue: 输入队列（包含待转换的HEIF文件信息）
            output_queue: 输出队列（转换后的JPG文件信息）
            dir_path: 工作目录
            max_workers: 最大并发转换线程数（根据推理速度动态调整）
            log_callback: 日志回调
        """
        super().__init__(
            name="HEIF转换",
            input_queue=input_queue,
            output_queue=output_queue,
            max_workers=max_workers,
            device='cpu',
            log_callback=log_callback
        )
        self.dir_path = dir_path
        self.temp_dir = os.path.join(dir_path, '.superpicky', 'temp_jpg')
        os.makedirs(self.temp_dir, exist_ok=True)
        self.heif_temp_map = {}  # 存储转换映射
    
    def process_job(self, job: Job) -> Dict[str, Any]:
        """处理HEIF转换任务"""
        data = job.data
        filename = data.get('filename')
        filepath = data.get('filepath')
        
        if not filename or not filepath:
            raise ValueError("缺少filename或filepath")
        
        # 生成临时 JPG 路径
        file_basename = os.path.splitext(filename)[0]
        temp_jpg_path = os.path.join(self.temp_dir, f"{file_basename}_temp.jpg")
        
        # 检查临时JPG是否已存在，如果存在则直接使用，跳过转换
        # 这样可以立即将任务送入推理队列，无需等待转换
        if os.path.exists(temp_jpg_path):
            # 验证文件是否有效（大小大于0）
            if os.path.getsize(temp_jpg_path) > 0:
                # 存储映射
                self.heif_temp_map[filename] = temp_jpg_path
                
                # 返回结果，立即传递给下一阶段（AI推理）
                file_prefix, _ = os.path.splitext(filename)
                return {
                    'filename': filename,
                    'filepath': filepath,  # 保留原始路径（EXIF写入用）
                    'file_prefix': file_prefix,  # 文件前缀（用于EXIF写入等）
                    'temp_jpg_path': temp_jpg_path,  # 临时JPG路径（AI推理用）
                    'is_heif': True
                }
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
        
        # 读取并转换
        pil_image = Image.open(filepath).convert('RGB')
        
        # 保存为 JPG
        pil_image.save(temp_jpg_path, 'JPEG', quality=95)
        
        # 存储映射
        self.heif_temp_map[filename] = temp_jpg_path
        
        # 返回结果，传递给下一阶段
        file_prefix, _ = os.path.splitext(filename)
        return {
            'filename': filename,
            'filepath': filepath,  # 保留原始路径（EXIF写入用）
            'file_prefix': file_prefix,  # 文件前缀（用于EXIF写入等）
            'temp_jpg_path': temp_jpg_path,  # 临时JPG路径（AI推理用）
            'is_heif': True
        }


class RAWConversionStage(PipelineStage):
    """RAW转换阶段"""
    
    def __init__(
        self,
        input_queue: JobQueue,
        output_queue: JobQueue,
        dir_path: str,
        max_workers: int = 2,  # 默认2个线程
        log_callback: Optional[callable] = None
    ):
        super().__init__(
            name="RAW转换",
            input_queue=input_queue,
            output_queue=output_queue,
            max_workers=max_workers,
            device='cpu',
            log_callback=log_callback
        )
        self.dir_path = dir_path
    
    def process_job(self, job: Job) -> Dict[str, Any]:
        """处理RAW转换任务"""
        data = job.data
        file_prefix = data.get('file_prefix')
        raw_ext = data.get('raw_ext')
        
        if not file_prefix or not raw_ext:
            raise ValueError("缺少file_prefix或raw_ext")
        
        raw_path = os.path.join(self.dir_path, file_prefix + raw_ext)
        
        # 使用现有的raw_to_jpeg函数
        from find_bird_util import raw_to_jpeg
        raw_to_jpeg(raw_path)
        
        # 生成JPG文件名
        jpg_filename = file_prefix + ".jpg"
        jpg_path = os.path.join(self.dir_path, jpg_filename)
        
        return {
            'filename': jpg_filename,
            'filepath': jpg_path,
            'file_prefix': file_prefix,
            'is_raw': True
        }


class ImageProcessingStage(PipelineStage):
    """图像处理阶段 - 整合YOLO、关键点、TOPIQ、飞版、曝光、对焦、评分等"""
    
    def __init__(
        self,
        input_queue: JobQueue,
        output_queue: Optional[JobQueue],
        dir_path: str,
        raw_dict: Dict[str, str],
        settings: Any,  # ProcessingSettings
        device: str = 'cpu',
        max_workers: int = 1,
        log_callback: Optional[callable] = None,
        stats_callback: Optional[callable] = None,
        progress_callback: Optional[callable] = None
    ):
        """
        初始化图像处理阶段
        
        Args:
            input_queue: 输入队列（包含文件信息）
            output_queue: 输出队列（None表示最终阶段）
            dir_path: 工作目录
            raw_dict: RAW文件字典
            settings: 处理设置
            device: 计算设备
            max_workers: 最大并发数
            log_callback: 日志回调
            stats_callback: 统计回调
            progress_callback: 进度回调
        """
        super().__init__(
            name=f"AI处理-{device.upper()}",
            input_queue=input_queue,
            output_queue=output_queue,
            max_workers=max_workers,
            device=device,
            log_callback=log_callback
        )
        self.dir_path = dir_path
        self.raw_dict = raw_dict
        self.settings = settings
        
        # 回调函数
        self.stats_callback = stats_callback
        self.progress_callback = progress_callback
        
        # 延迟加载的模型
        self._yolo_model = None
        self._keypoint_detector = None
        self._flight_detector = None
        self._exiftool_mgr = None
        self._rating_engine = None
        
        # 线程本地存储（每个工作线程有自己的模型实例）
        self._thread_local = None
    
    def _get_yolo_model(self):
        """获取YOLO模型（延迟加载）"""
        if self._yolo_model is None:
            from ai_model import load_yolo_model
            self._yolo_model = load_yolo_model(device=self.device)
        return self._yolo_model
    
    def _get_keypoint_detector(self):
        """获取关键点检测器（延迟加载）"""
        if self._keypoint_detector is None:
            from core.keypoint_detector import get_keypoint_detector
            detector = get_keypoint_detector()
            try:
                detector.load_model()
                self._keypoint_detector = detector
            except FileNotFoundError:
                self._keypoint_detector = None
        return self._keypoint_detector
    
    def _get_flight_detector(self):
        """获取飞版检测器（延迟加载）"""
        if self._flight_detector is None and self.settings.detect_flight:
            from core.flight_detector import get_flight_detector
            detector = get_flight_detector()
            try:
                detector.load_model()
                self._flight_detector = detector
            except FileNotFoundError:
                self._flight_detector = None
        return self._flight_detector
    
    def _get_exiftool_mgr(self):
        """获取EXIF工具管理器（延迟加载）"""
        if self._exiftool_mgr is None:
            from exiftool_manager import get_exiftool_manager
            self._exiftool_mgr = get_exiftool_manager()
        return self._exiftool_mgr
    
    def _get_rating_engine(self):
        """获取评分引擎（延迟加载）"""
        if self._rating_engine is None:
            from core.rating_engine import create_rating_engine_from_config
            from advanced_config import get_advanced_config
            config = get_advanced_config()
            self._rating_engine = create_rating_engine_from_config(config)
            self._rating_engine.update_thresholds(
                sharpness_threshold=self.settings.sharpness_threshold,
                nima_threshold=self.settings.nima_threshold
            )
        return self._rating_engine
    
    def process_job(self, job: Job) -> Dict[str, Any]:
        """
        处理单张图片的完整流程
        这是核心处理逻辑，整合了所有检测和评分步骤
        """
        data = job.data
        filename = data.get('filename')
        filepath = data.get('filepath')
        file_prefix = data.get('file_prefix')
        
        # 如果是HEIF，使用临时JPG路径进行AI推理
        if data.get('is_heif') and data.get('temp_jpg_path'):
            ai_inference_path = data['temp_jpg_path']
        else:
            ai_inference_path = filepath
        
        if not filename or not filepath:
            raise ValueError("缺少filename或filepath")
        
        if not file_prefix:
            file_prefix, _ = os.path.splitext(filename)
        
        # 记录开始时间
        start_time = time.time()
        
        # 准备UI设置
        ui_settings = [
            self.settings.ai_confidence,
            self.settings.sharpness_threshold,
            self.settings.nima_threshold,
            self.settings.save_crop,
            self.settings.normalization_mode
        ]
        
        # Phase 1: YOLO检测
        from ai_model import detect_and_draw_birds
        yolo_model = self._get_yolo_model()
        result = detect_and_draw_birds(
            ai_inference_path, yolo_model, None, self.dir_path, 
            ui_settings, None, skip_nima=True
        )
        
        if result is None:
            raise RuntimeError("YOLO检测失败")
        
        detected, _, confidence, sharpness, _, bird_bbox, img_dims, bird_mask = result
        
        # Phase 2: 关键点检测
        head_sharpness = 0.0
        best_eye_visibility = 0.0
        all_keypoints_hidden = False
        has_visible_eye = False
        has_visible_beak = False
        left_eye_vis = 0.0
        right_eye_vis = 0.0
        beak_vis = 0.0
        head_center_orig = None
        head_radius_val = None
        bird_crop_bgr = None
        bird_crop_mask = None
        bird_mask_orig = None
        
        keypoint_detector = self._get_keypoint_detector()
        if keypoint_detector and detected and bird_bbox is not None and img_dims is not None:
            try:
                import cv2
                from utils import read_image
                orig_img = read_image(filepath)  # 读取原图（支持HEIF）
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
                            
                            # 计算头部区域
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
            except Exception as e:
                pass  # 关键点检测失败不影响主流程
        
        # Phase 3: TOPIQ评分（条件执行）
        topiq = None
        if detected and not all_keypoints_hidden and best_eye_visibility >= 0.3:
            try:
                from iqa_scorer import get_iqa_scorer
                scorer = get_iqa_scorer(device=self.device)
                topiq = scorer.calculate_nima(filepath)
            except Exception:
                pass
        
        # Phase 4: 飞版检测
        is_flying = False
        flight_confidence = 0.0
        flight_detector = self._get_flight_detector()
        if flight_detector and detected and bird_crop_bgr is not None and bird_crop_bgr.size > 0:
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
                from core.exposure_detector import get_exposure_detector
                exposure_detector = get_exposure_detector()
                exposure_result = exposure_detector.detect(
                    bird_crop_bgr,
                    threshold=self.settings.exposure_threshold
                )
                is_overexposed = exposure_result.is_overexposed
                is_underexposed = exposure_result.is_underexposed
            except Exception:
                pass
        
        # Phase 6: 初步评分（不考虑对焦）
        rating_engine = self._get_rating_engine()
        preliminary_result = rating_engine.calculate(
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
        
        # Phase 7: 对焦点验证（仅1星以上）
        focus_sharpness_weight = 1.0
        focus_topiq_weight = 1.0
        focus_x, focus_y = None, None
        
        if preliminary_result.rating >= 1:
            if detected and bird_bbox is not None and img_dims is not None:
                if file_prefix in self.raw_dict:
                    raw_ext = self.raw_dict[file_prefix]
                    raw_path = os.path.join(self.dir_path, file_prefix + raw_ext)
                    if raw_ext.lower() in ['.nef', '.nrw', '.arw', '.cr3', '.cr2', '.orf', '.raf', '.rw2']:
                        try:
                            from core.focus_point_detector import get_focus_detector, verify_focus_in_bbox
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
        
        # Phase 8: 最终评分
        rating_result = rating_engine.calculate(
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
        
        # 计算对焦状态
        focus_status = None
        if detected:
            if focus_sharpness_weight > 1.0:
                focus_status = "精准"
            elif focus_sharpness_weight >= 1.0:
                focus_status = "鸟身"
            elif focus_sharpness_weight >= 0.7:
                focus_status = "偏移"
            elif focus_sharpness_weight < 0.7:
                focus_status = "脱焦"
        
        # 计算耗时
        processing_time = time.time() - start_time
        
        # 计算飞鸟加成后的值（用于统计）
        rating_sharpness = head_sharpness
        rating_topiq = topiq
        if is_flying and confidence >= 0.5:
            rating_sharpness = head_sharpness + 100
            if topiq is not None:
                rating_topiq = topiq + 0.5
        
        # 构建结果
        result = {
            'filename': filename,
            'filepath': filepath,
            'file_prefix': file_prefix,
            'rating': rating_value,
            'pick': pick,
            'reason': reason,
            'confidence': confidence,
            'head_sharpness': head_sharpness,
            'rating_sharpness': rating_sharpness,  # 加成后的锐度
            'topiq': topiq,
            'rating_topiq': rating_topiq,  # 加成后的美学
            'is_flying': is_flying,
            'flight_confidence': flight_confidence,
            'is_overexposed': is_overexposed,
            'is_underexposed': is_underexposed,
            'focus_status': focus_status,
            'focus_sharpness_weight': focus_sharpness_weight,  # 对焦权重
            'focus_topiq_weight': focus_topiq_weight,
            'focus_x': focus_x,
            'focus_y': focus_y,
            'has_visible_eye': has_visible_eye,
            'has_visible_beak': has_visible_beak,
            'left_eye_vis': left_eye_vis,
            'right_eye_vis': right_eye_vis,
            'beak_vis': beak_vis,
            'best_eye_visibility': best_eye_visibility,
            'detected': detected,
            'processing_time': processing_time,
        }
        
        # 调用统计回调
        if self.stats_callback:
            self.stats_callback(result)
        
        # 调用进度回调
        # 注意：流水线模式下无法准确计算总进度，传递 -1 表示"有更新但百分比未知"
        if self.progress_callback:
            try:
                # 检查回调函数是否需要参数
                import inspect
                sig = inspect.signature(self.progress_callback)
                param_count = len(sig.parameters)
                if param_count > 0:
                    # 回调需要参数，传递 -1 表示进度更新但百分比未知
                    self.progress_callback(-1)
                else:
                    # 回调不需要参数（向后兼容）
                    self.progress_callback()
            except (TypeError, ValueError, AttributeError):
                # 如果检查签名失败，尝试无参数调用（向后兼容）
                try:
                    self.progress_callback()
                except Exception:
                    pass  # 静默失败，不影响主流程
            except Exception:
                pass  # 静默失败，不影响主流程
        
        return result


class EXIFWriteStage(PipelineStage):
    """EXIF写入阶段 - 批量写入元数据"""
    
    def __init__(
        self,
        input_queue: JobQueue,
        dir_path: str,
        raw_dict: Dict[str, str],
        settings: Any,
        max_workers: int = 2,  # EXIF写入可以并行
        log_callback: Optional[callable] = None,
        stats_callback: Optional[callable] = None
    ):
        super().__init__(
            name="EXIF写入",
            input_queue=input_queue,
            output_queue=None,  # 最终阶段
            max_workers=max_workers,
            device='cpu',
            log_callback=log_callback
        )
        self.dir_path = dir_path
        self.raw_dict = raw_dict
        self.settings = settings
        self.stats_callback = stats_callback
        self._exiftool_mgr = None
        self._csv_lock = threading.Lock()
    
    def _get_exiftool_mgr(self):
        """获取EXIF工具管理器（延迟加载）"""
        if self._exiftool_mgr is None:
            from exiftool_manager import get_exiftool_manager
            self._exiftool_mgr = get_exiftool_manager()
        return self._exiftool_mgr
    
    def process_job(self, job: Job) -> Dict[str, Any]:
        """处理EXIF写入任务"""
        data = job.data
        file_prefix = data.get('file_prefix')
        rating_value = data.get('rating', 0)
        pick = data.get('pick', 0)
        head_sharpness = data.get('head_sharpness', 0.0)
        topiq = data.get('topiq')
        is_flying = data.get('is_flying', False)
        focus_status = data.get('focus_status')
        focus_sharpness_weight = data.get('focus_sharpness_weight', 1.0)
        focus_topiq_weight = data.get('focus_topiq_weight', 1.0)
        confidence = data.get('confidence', 0.0)
        best_eye_visibility = data.get('best_eye_visibility', 0.0)
        reason = data.get('reason', '')
        
        # 确定目标文件（RAW优先）
        target_file_path = None
        if file_prefix in self.raw_dict:
            raw_ext = self.raw_dict[file_prefix]
            target_file_path = os.path.join(self.dir_path, file_prefix + raw_ext)
        else:
            # 纯JPG/HEIF文件
            filename = data.get('filename')
            if filename:
                target_file_path = os.path.join(self.dir_path, filename)
        
        if not target_file_path or not os.path.exists(target_file_path):
            return {'status': 'skipped', 'reason': 'file_not_found'}
        
        # 构建标签
        label = None
        if is_flying:
            label = 'Green'
        elif focus_sharpness_weight > 1.0:
            label = 'Red'
        
        # 构建详细评分说明
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
            f"是否飞鸟: {'是 (锐度×1.2, 美学×1.1)' if is_flying else '否'}",
        ]
        caption = " | ".join(caption_parts)
        
        # 批量写入EXIF
        exiftool_mgr = self._get_exiftool_mgr()
        batch_data = [{
            'file': target_file_path,
            'rating': rating_value if rating_value >= 0 else 0,
            'pick': pick,
            'sharpness': head_sharpness,
            'nima_score': topiq,
            'label': label,
            'focus_status': focus_status,
            'caption': caption,
        }]
        
        exiftool_mgr.batch_set_metadata(batch_data)
        
        # 更新CSV（线程安全）
        with self._csv_lock:
            self._update_csv_data(data)
        
        return {'status': 'completed', 'file': target_file_path}
    
    def _update_csv_data(self, data: Dict[str, Any]):
        """更新CSV数据（线程安全）"""
        try:
            from core.photo_processor import PhotoProcessor
            # 这里需要访问PhotoProcessor的_update_csv_keypoint_data方法
            # 为了解耦，我们直接在这里实现CSV更新逻辑
            import csv
            csv_path = os.path.join(self.dir_path, ".superpicky", "report.csv")
            if not os.path.exists(csv_path):
                return
            
            # 读取现有CSV
            rows = []
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames) if reader.fieldnames else []
                
                # 确保字段存在
                required_fields = ['focus_status', 'focus_x', 'focus_y']
                for field in required_fields:
                    if field not in fieldnames:
                        rating_idx = fieldnames.index('rating') if 'rating' in fieldnames else len(fieldnames)
                        fieldnames.insert(rating_idx + 1, field)
                
                for row in reader:
                    if row.get('filename') == data.get('file_prefix'):
                        # 更新数据
                        row['head_sharp'] = f"{data.get('head_sharpness', 0):.0f}" if data.get('head_sharpness', 0) > 0 else "-"
                        row['left_eye'] = f"{data.get('left_eye_vis', 0):.2f}"
                        row['right_eye'] = f"{data.get('right_eye_vis', 0):.2f}"
                        row['beak'] = f"{data.get('beak_vis', 0):.2f}"
                        row['nima_score'] = f"{data.get('topiq', 0):.2f}" if data.get('topiq') is not None else "-"
                        row['is_flying'] = "yes" if data.get('is_flying', False) else "no"
                        row['flight_conf'] = f"{data.get('flight_confidence', 0):.2f}"
                        row['rating'] = str(data.get('rating', 0))
                        row['focus_status'] = data.get('focus_status', '') or "-"
                        row['focus_x'] = f"{data.get('focus_x', 0):.3f}" if data.get('focus_x') is not None else "-"
                        row['focus_y'] = f"{data.get('focus_y', 0):.3f}" if data.get('focus_y') is not None else "-"
                    rows.append(row)
            
            # 写回CSV
            if fieldnames and rows:
                with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
        except Exception as e:
            if self.log_callback:
                self.log_callback(f"更新CSV失败: {e}", "warning")

