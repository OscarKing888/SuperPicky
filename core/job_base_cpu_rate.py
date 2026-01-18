# -*- coding: utf-8 -*-

import os
from typing import Optional, Dict, Any

from core.job_base import JobBase
from core.photo_processor import PhotoProcessor


class JobBaseCPU_Rate(JobBase):
    """CPU评分任务（模型/设备由CPUJobWorker注入）"""

    def __init__(
        self,
        job_id: str,
        file_path: str,
        photo_processor: PhotoProcessor,
        raw_dict: Dict[str, str],
        ai_inference_path: Optional[str] = None,
    ):
        super().__init__(job_id)
        self.file_path = file_path
        self.photo_processor = photo_processor
        self.raw_dict = raw_dict
        self.ai_inference_path = ai_inference_path
        self.result: Optional[Dict[str, Any]] = None

        # 由 worker 注入（避免在 job 内创建/缓存设备资源）
        self.model = None
        self.keypoint_detector = None
        self.flight_detector = None
        self.exiftool_mgr = None
        self.device = 'cpu'

    def do_job(self):
        """执行CPU评分任务（依赖worker已注入模型/检测器/EXIF管理器）"""
        try:
            filename = os.path.basename(self.file_path)
            self.result = self.photo_processor.process_single_image(
                filename=filename,
                filepath=self.file_path,
                raw_dict=self.raw_dict,
                model=self.model,
                keypoint_detector=self.keypoint_detector,
                flight_detector=self.flight_detector,
                exiftool_mgr=self.exiftool_mgr,
                use_keypoints=self.keypoint_detector is not None,
                use_flight=self.flight_detector is not None and self.photo_processor.settings.detect_flight,
                ai_inference_path=self.ai_inference_path,
            )
            self.result['device'] = self.device
        except Exception as e:
            self.error = f"CPU评分失败: {str(e)}"
            self.result = {
                'filename': os.path.basename(self.file_path) if self.file_path else 'unknown',
                'filepath': self.file_path,
                'rating': -1,
                'reason': f'处理异常: {str(e)}',
                'processing_time': 0.0,
            }

    def get_result(self) -> Dict[str, Any]:
        return self.result or {}
