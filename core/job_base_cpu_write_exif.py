# -*- coding: utf-8 -*-

import os
import time
from typing import Optional, Dict, Any

from core.job_base import JobBase


class JobBaseCPU_WriteEXIF(JobBase):
    """CPU EXIF写入任务"""
    
    def __init__(
        self,
        job_id: str,
        file_path: str,
        exif_data: Dict[str, Any],
        temp_jpg_path: Optional[str] = None
    ):
        """
        初始化EXIF写入任务
        
        Args:
            job_id: 任务ID
            file_path: 目标文件路径（RAW优先）
            exif_data: EXIF数据字典，包含rating、pick、sharpness等
            temp_jpg_path: 临时JPG路径（用于HEIF，可选）
        """
        super().__init__(job_id)
        self.file_path = file_path
        self.exif_data = exif_data
        self.temp_jpg_path = temp_jpg_path

    def do_job(self):
        """执行EXIF写入任务"""
        try:
            from exiftool_manager import get_exiftool_manager
            exiftool_mgr = get_exiftool_manager()
            
            # 构建批量写入数据
            batch_data = []
            
            # 主文件（RAW或JPG）
            if self.file_path and os.path.exists(self.file_path):
                batch_data.append({
                    'file': self.file_path,
                    'rating': self.exif_data.get('rating', 0) if self.exif_data.get('rating', 0) >= 0 else 0,
                    'pick': self.exif_data.get('pick', 0),
                    'sharpness': self.exif_data.get('adj_sharpness') or self.exif_data.get('head_sharpness'),
                    'nima_score': self.exif_data.get('adj_topiq') or self.exif_data.get('topiq'),
                    'label': self._get_label(),
                    'focus_status': self.exif_data.get('focus_status'),
                    'caption': self._build_caption(),
                })
            
            # 临时JPG文件（HEIF转换后的）
            if self.temp_jpg_path and os.path.exists(self.temp_jpg_path) and self.temp_jpg_path != self.file_path:
                batch_data.append({
                    'file': self.temp_jpg_path,
                    'rating': self.exif_data.get('rating', 0) if self.exif_data.get('rating', 0) >= 0 else 0,
                    'pick': self.exif_data.get('pick', 0),
                    'sharpness': self.exif_data.get('adj_sharpness') or self.exif_data.get('head_sharpness'),
                    'nima_score': self.exif_data.get('adj_topiq') or self.exif_data.get('topiq'),
                    'label': self._get_label(),
                    'focus_status': self.exif_data.get('focus_status'),
                    'caption': self._build_caption(),
                })
            
            if not batch_data:
                self.error = "没有可写入的文件"
                return
            
            # 批量写入EXIF
            exif_stats = exiftool_mgr.batch_set_metadata(batch_data)
            
            if exif_stats.get('failed', 0) > 0:
                self.error = f"部分EXIF写入失败: {exif_stats.get('failed')}个文件"
            
        except Exception as e:
            self.error = f"EXIF写入失败: {str(e)}"
    
    def _get_label(self) -> Optional[str]:
        """获取标签（飞鸟绿色，头部对焦红色）"""
        is_flying = self.exif_data.get('is_flying', False)
        focus_sharpness_weight = self.exif_data.get('focus_sharpness_weight', 1.0)
        
        if is_flying:
            return 'Green'
        elif focus_sharpness_weight > 1.0:
            return 'Red'
        return None
    
    def _build_caption(self) -> str:
        """构建详细评分说明"""
        rating_value = self.exif_data.get('rating', 0)
        reason = self.exif_data.get('reason', '')
        confidence = self.exif_data.get('confidence', 0.0)
        head_sharpness = self.exif_data.get('head_sharpness', 0.0)
        topiq = self.exif_data.get('topiq')
        best_eye_visibility = self.exif_data.get('best_eye_visibility', 0.0)
        focus_sharpness_weight = self.exif_data.get('focus_sharpness_weight', 1.0)
        focus_topiq_weight = self.exif_data.get('focus_topiq_weight', 1.0)
        is_flying = self.exif_data.get('is_flying', False)
        
        caption_lines = [
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
        
        return " | ".join(caption_lines)
    
    def get_result(self) -> Dict[str, Any]:
        """获取写入结果"""
        return {
            'file_path': self.file_path,
            'temp_jpg_path': self.temp_jpg_path,
            'success': self.error is None
        }
