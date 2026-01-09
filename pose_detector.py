#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PoseDetector - 鸟类姿态检测器
封装 YOLOv11-pose 模型的加载和推理

支持:
- 单张图片检测
- 批量检测
- 关键点可视化
- 结果导出 (JSON/CSV)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict
import json

# CUB-200-2011 数据集的15个关键点定义
BIRD_KEYPOINT_NAMES = [
    "back",           # 0: 背部
    "beak",           # 1: 喙
    "belly",          # 2: 腹部
    "breast",         # 3: 胸部
    "crown",          # 4: 头冠
    "forehead",       # 5: 前额
    "left_eye",       # 6: 左眼
    "left_leg",       # 7: 左腿
    "left_wing",      # 8: 左翼
    "nape",           # 9: 颈背
    "right_eye",      # 10: 右眼
    "right_leg",      # 11: 右腿
    "right_wing",     # 12: 右翼
    "tail",           # 13: 尾巴
    "throat",         # 14: 喉咙
]

# 关键点连接骨架 (用于可视化)
BIRD_SKELETON = [
    (4, 5),   # crown - forehead
    (5, 1),   # forehead - beak
    (4, 9),   # crown - nape
    (9, 0),   # nape - back
    (0, 13),  # back - tail
    (3, 2),   # breast - belly
    (2, 7),   # belly - left_leg
    (2, 11),  # belly - right_leg
    (0, 8),   # back - left_wing
    (0, 12),  # back - right_wing
    (6, 4),   # left_eye - crown
    (10, 4),  # right_eye - crown
    (14, 3),  # throat - breast
]

# 关键点颜色 (BGR格式)
KEYPOINT_COLORS = {
    "head": (255, 100, 100),    # 头部相关 - 蓝色
    "body": (100, 255, 100),    # 身体相关 - 绿色
    "limb": (100, 100, 255),    # 四肢相关 - 红色
    "default": (255, 255, 0),   # 默认 - 青色
}

# 关键点分组
KEYPOINT_GROUPS = {
    "head": [1, 4, 5, 6, 10, 14],        # beak, crown, forehead, eyes, throat
    "body": [0, 2, 3, 9],                # back, belly, breast, nape
    "limb": [7, 8, 11, 12, 13],          # legs, wings, tail
}


@dataclass
class Keypoint:
    """单个关键点数据"""
    name: str
    x: float
    y: float
    confidence: float
    visible: bool
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BirdDetection:
    """单只鸟的检测结果"""
    bbox: tuple          # (x1, y1, x2, y2)
    confidence: float
    keypoints: List[Keypoint]
    
    @property
    def visible_keypoints(self) -> int:
        """可见关键点数量"""
        return sum(1 for kp in self.keypoints if kp.visible)
    
    @property
    def keypoint_quality(self) -> float:
        """关键点检测质量 (0-1)"""
        if not self.keypoints:
            return 0.0
        avg_conf = sum(kp.confidence for kp in self.keypoints) / len(self.keypoints)
        visibility_ratio = self.visible_keypoints / len(self.keypoints)
        return (avg_conf + visibility_ratio) / 2
    
    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "visible_keypoints": self.visible_keypoints,
            "keypoint_quality": self.keypoint_quality,
        }


class PoseDetector:
    """
    鸟类姿态检测器
    
    使用 YOLOv11-pose 模型检测鸟类并预测关键点
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        verbose: bool = False
    ):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径，None则自动查找
            device: 推理设备 ("auto", "cpu", "mps", "cuda")
            verbose: 是否显示详细日志
        """
        self.verbose = verbose
        self.model_path = model_path or self._find_model()
        self.device = self._select_device(device)
        self.model = None
        
        self._load_model()
    
    def _find_model(self) -> str:
        """查找可用模型"""
        search_paths = [
            "./models/yolo11s-pose-bird.pt",
            "./models/best.pt",
            "./yolo11s-pose.pt",
            "yolo11s-pose.pt",  # 使用预训练模型
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        # 如果找不到自定义模型，返回预训练模型名称
        # Ultralytics 会自动下载
        return "yolo11s-pose.pt"
    
    def _select_device(self, device: str) -> str:
        """选择推理设备"""
        if device != "auto":
            return device
        
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        
        return "cpu"
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            
            if self.verbose:
                print(f"📦 加载模型: {self.model_path}")
                print(f"   设备: {self.device}")
            
            self.model = YOLO(self.model_path)
            
            if self.verbose:
                print(f"✅ 模型加载成功")
                
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def detect(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> Dict[str, Any]:
        """
        检测图片中的鸟类姿态
        
        Args:
            image_path: 图片路径
            conf: 置信度阈值
            iou: NMS IOU阈值
            
        Returns:
            检测结果字典
        """
        if not os.path.exists(image_path):
            return {"error": f"图片不存在: {image_path}", "detections": []}
        
        try:
            # 运行推理
            results = self.model(
                image_path,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False
            )
            
            # 解析结果
            detections = self._parse_results(results[0])
            
            return {
                "image": image_path,
                "detections": [det.to_dict() for det in detections],
                "count": len(detections),
            }
            
        except Exception as e:
            return {"error": str(e), "detections": []}
    
    def _parse_results(self, result) -> List[BirdDetection]:
        """解析 YOLO 推理结果"""
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        
        # 检查是否有 keypoints
        if result.keypoints is None:
            # 没有关键点数据，只返回边界框
            for i, (box, conf) in enumerate(zip(boxes, confs)):
                det = BirdDetection(
                    bbox=tuple(box.tolist()),
                    confidence=float(conf),
                    keypoints=[]
                )
                detections.append(det)
            return detections
        
        # 解析关键点
        keypoints_data = result.keypoints.data.cpu().numpy()
        
        for i, (box, conf, kps) in enumerate(zip(boxes, confs, keypoints_data)):
            keypoints = []
            
            for j, kp in enumerate(kps):
                x, y = kp[0], kp[1]
                kp_conf = kp[2] if len(kp) > 2 else 0.0
                
                # 判断关键点名称
                kp_name = BIRD_KEYPOINT_NAMES[j] if j < len(BIRD_KEYPOINT_NAMES) else f"kp_{j}"
                
                keypoints.append(Keypoint(
                    name=kp_name,
                    x=float(x),
                    y=float(y),
                    confidence=float(kp_conf),
                    visible=kp_conf > 0.5
                ))
            
            det = BirdDetection(
                bbox=tuple(box.tolist()),
                confidence=float(conf),
                keypoints=keypoints
            )
            detections.append(det)
        
        return detections
    
    def detect_batch(
        self,
        image_paths: List[str],
        conf: float = 0.25,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量检测多张图片
        
        Args:
            image_paths: 图片路径列表
            conf: 置信度阈值
            show_progress: 是否显示进度
            
        Returns:
            检测结果列表
        """
        results = []
        total = len(image_paths)
        
        for i, path in enumerate(image_paths, 1):
            if show_progress:
                print(f"[{i}/{total}] {Path(path).name}", end=" ... ")
            
            result = self.detect(path, conf=conf)
            results.append(result)
            
            if show_progress:
                count = result.get("count", 0)
                print(f"检测到 {count} 只鸟")
        
        return results
    
    def save_visualization(
        self,
        image_path: str,
        results: Dict[str, Any],
        output_dir: str,
        show_labels: bool = True,
        show_skeleton: bool = True,
    ) -> str:
        """
        保存可视化结果图片
        
        Args:
            image_path: 原图路径
            results: 检测结果
            output_dir: 输出目录
            show_labels: 是否显示关键点标签
            show_skeleton: 是否显示骨架连接
            
        Returns:
            输出图片路径
        """
        from utils import read_image
        img = read_image(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        for det in results.get("detections", []):
            # 绘制边界框
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制置信度
            conf_text = f"{det['confidence']:.2f}"
            cv2.putText(img, conf_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            keypoints = det.get("keypoints", [])
            if not keypoints:
                continue
            
            # 为绘制骨架准备坐标
            kp_coords = {}
            
            # 绘制关键点
            for kp in keypoints:
                if not kp.get("visible", False):
                    continue
                
                x, y = int(kp["x"]), int(kp["y"])
                name = kp["name"]
                
                # 存储坐标用于骨架绘制
                idx = BIRD_KEYPOINT_NAMES.index(name) if name in BIRD_KEYPOINT_NAMES else -1
                if idx >= 0:
                    kp_coords[idx] = (x, y)
                
                # 选择颜色
                color = KEYPOINT_COLORS["default"]
                for group, indices in KEYPOINT_GROUPS.items():
                    if idx in indices:
                        color = KEYPOINT_COLORS[group]
                        break
                
                # 绘制点
                cv2.circle(img, (x, y), 5, color, -1)
                cv2.circle(img, (x, y), 7, (255, 255, 255), 1)
                
                # 绘制标签
                if show_labels:
                    cv2.putText(img, name, (x + 8, y + 3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 绘制骨架
            if show_skeleton:
                for start_idx, end_idx in BIRD_SKELETON:
                    if start_idx in kp_coords and end_idx in kp_coords:
                        pt1 = kp_coords[start_idx]
                        pt2 = kp_coords[end_idx]
                        cv2.line(img, pt1, pt2, (200, 200, 200), 2)
        
        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        output_name = f"viz_{Path(image_path).stem}.jpg"
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, img)
        
        return output_path
    
    def export_results(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        format: str = "json"
    ):
        """
        导出检测结果
        
        Args:
            results: 检测结果列表
            output_path: 输出文件路径
            format: 输出格式 ("json" 或 "csv")
        """
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 表头
                headers = ["image", "detection_id", "confidence", 
                          "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                          "visible_keypoints", "keypoint_quality"]
                writer.writerow(headers)
                
                # 数据行
                for result in results:
                    image = result.get("image", "")
                    for i, det in enumerate(result.get("detections", [])):
                        bbox = det.get("bbox", (0, 0, 0, 0))
                        row = [
                            image, i, det.get("confidence", 0),
                            bbox[0], bbox[1], bbox[2], bbox[3],
                            det.get("visible_keypoints", 0),
                            det.get("keypoint_quality", 0)
                        ]
                        writer.writerow(row)
        
        if self.verbose:
            print(f"💾 结果已保存: {output_path}")


# 便捷函数
def quick_detect(image_path: str, conf: float = 0.25) -> Dict[str, Any]:
    """快速检测单张图片"""
    detector = PoseDetector(verbose=False)
    return detector.detect(image_path, conf=conf)


if __name__ == "__main__":
    # 简单测试
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        detector = PoseDetector(verbose=True)
        results = detector.detect(image_path)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("Usage: python pose_detector.py <image_path>")
