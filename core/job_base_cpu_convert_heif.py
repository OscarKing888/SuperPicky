# -*- coding: utf-8 -*-

import os
import time
from typing import Optional, Dict, Any
from PIL import Image

from core.job_base import JobBase


class JobBaseCPU_ConvertHEIF(JobBase):
    """CPU HEIF转换任务"""
    
    def __init__(self, job_id: str, heif_file_path: str, output_dir: str):
        """
        初始化HEIF转换任务
        
        Args:
            job_id: 任务ID
            heif_file_path: HEIF文件路径
            output_dir: 输出目录（临时JPG保存位置）
        """
        super().__init__(job_id)
        self.heif_file_path = heif_file_path
        self.output_dir = output_dir
        self.temp_jpg_path = None
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def do_job(self):
        """执行HEIF转换"""
        try:
            # 生成临时JPG路径
            file_basename = os.path.splitext(os.path.basename(self.heif_file_path))[0]
            self.temp_jpg_path = os.path.join(self.output_dir, f"{file_basename}.jpg")
            
            # 检查是否已存在
            if os.path.exists(self.temp_jpg_path) and os.path.getsize(self.temp_jpg_path) > 0:
                return  # 已存在，跳过转换
            
            # 注册 pillow-heif
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            except ImportError:
                self.error = "pillow-heif未安装"
                return
            
            # 读取并转换
            pil_image = Image.open(self.heif_file_path).convert('RGB')
            
            # 保存为JPG
            pil_image.save(self.temp_jpg_path, 'JPEG', quality=100, subsampling=0, optimize=True)
            
        except Exception as e:
            self.error = f"HEIF转换失败: {str(e)}"
    
    def get_result(self) -> Dict[str, Any]:
        """获取转换结果"""
        return {
            'temp_jpg_path': self.temp_jpg_path,
            'heif_file_path': self.heif_file_path,
            'success': self.error is None
        }
