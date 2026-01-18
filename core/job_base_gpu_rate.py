# -*- coding: utf-8 -*-

import os
from typing import Optional, Dict, Any

from core.job_base_gpu import JobBaseGPU
from core.job_base import JobFileInfo
from core.photo_processor import PhotoProcessor


class JobBaseGPU_Rate(JobBaseGPU):
    """GPU评分任务（模型/设备由GPUJobWorker注入）"""

    def __init__(
        self,
        job_file_info: JobFileInfo,
        photo_processor: PhotoProcessor,
        raw_dict: Dict[str, str],
    ):
        super().__init__(job_file_info)
        self.photo_processor = photo_processor
        self.raw_dict = raw_dict
        self.result: Optional[Dict[str, Any]] = None

        # 由 worker 注入
        self.model = None
        self.keypoint_detector = None
        self.flight_detector = None
        self.exiftool_mgr = None
        self.device = 'cuda'

    def do_job(self):
        """执行GPU评分任务（依赖worker已注入模型/检测器/EXIF管理器）"""
        try:
            # 确定推理使用的路径（HEIF使用临时JPG，其他使用原路径）
            if self.job_file_info.needs_tmp_file() and self.job_file_info.tmp_file_path:
                ai_inference_path = self.job_file_info.tmp_file_path
                filepath = self.job_file_info.src_file_path  # 原HEIF路径
            else:
                ai_inference_path = None
                filepath = self.job_file_info.src_file_path
            
            filename = os.path.basename(filepath)
            self.result = self.photo_processor.process_single_image(
                filename=filename,
                filepath=filepath,
                raw_dict=self.raw_dict,
                model=self.model,
                keypoint_detector=self.keypoint_detector,
                flight_detector=self.flight_detector,
                exiftool_mgr=self.exiftool_mgr,
                use_keypoints=self.keypoint_detector is not None,
                use_flight=self.flight_detector is not None and self.photo_processor.settings.detect_flight,
                ai_inference_path=ai_inference_path,
            )
            self.result['device'] = self.device
            self.result['job_file_info'] = self.job_file_info
        except Exception as e:
            self.error = f"GPU评分失败: {str(e)}"
            self.result = {
                'filename': os.path.basename(self.job_file_info.src_file_path) if self.job_file_info.src_file_path else 'unknown',
                'filepath': self.job_file_info.src_file_path,
                'rating': -1,
                'reason': f'处理异常: {str(e)}',
                'processing_time': 0.0,
                'job_file_info': self.job_file_info,
            }

    def get_result(self) -> Dict[str, Any]:
        return self.result or {}
