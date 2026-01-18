#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IQA (Image Quality Assessment) 评分器
使用 TOPIQ 美学评分模型

V3.7: 切换到 TOPIQ 模型，更好的鸟类摄影美学评估
- TOPIQ 使用 Top-down 语义理解，对主体识别更准确
- 比 NIMA 快约 40%
- 基于 ResNet50 + CFANet 架构
"""

import os
import sys
import torch
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torchvision.transforms as T

# 使用 TOPIQ 模型
from topiq_model import CFANet, load_topiq_weights, get_topiq_weight_path


class IQAScorer:
    """IQA 评分器 - 使用 TOPIQ 美学评分"""

    def __init__(self, device='mps'):
        """
        初始化 IQA 评分器

        Args:
            device: 计算设备 ('mps', 'cuda', 'cpu')
        """
        self.device = self._get_device(device)
        print(f"🎨 IQA 评分器初始化中... (设备: {self.device})")

        # 延迟加载模型（第一次使用时才加载）
        self._topiq_model = None

        print("✅ IQA 评分器已就绪 (TOPIQ模型将在首次使用时加载)")

    def _get_device(self, preferred_device='mps'):
        """
        获取最佳计算设备

        Args:
            preferred_device: 首选设备

        Returns:
            可用的设备
        """
        # 检查 MPS (Apple GPU)
        if preferred_device == 'mps':
            try:
                if torch.backends.mps.is_available():
                    return torch.device('mps')
            except:
                pass

        # 检查 CUDA (NVIDIA GPU)
        if preferred_device == 'cuda' or torch.cuda.is_available():
            return torch.device('cuda')

        # 默认使用 CPU
        return torch.device('cpu')

    def _load_topiq(self):
        """延迟加载 TOPIQ 模型"""
        if self._topiq_model is None:
            print("📥 加载 TOPIQ 美学评分模型...")
            try:
                # 获取权重路径
                weight_path = get_topiq_weight_path()
                
                # 初始化 TOPIQ 模型
                self._topiq_model = CFANet()
                # 先加载到 CPU，再移动到目标设备（避免设备不匹配）
                load_topiq_weights(self._topiq_model, weight_path, torch.device('cpu'))
                # 确保模型所有参数都在目标设备上
                self._topiq_model = self._topiq_model.to(self.device)
                self._topiq_model.eval()
                print("✅ TOPIQ 模型加载完成")
            except Exception as e:
                print(f"⚠️  TOPIQ 模型加载失败: {e}")
                print("   尝试使用 CPU 模式...")
                try:
                    weight_path = get_topiq_weight_path()
                    self._topiq_model = CFANet()
                    load_topiq_weights(self._topiq_model, weight_path, torch.device('cpu'))
                    self._topiq_model.to(torch.device('cpu'))
                    self._topiq_model.eval()
                    self.device = torch.device('cpu')
                    print("✅ TOPIQ 模型加载完成 (CPU模式)")
                except Exception as e2:
                    raise RuntimeError(f"TOPIQ 模型加载失败: {e2}")
        else:
            # 确保模型在正确的设备上
            if next(self._topiq_model.parameters()).device != self.device:
                self._topiq_model = self._topiq_model.to(self.device)
        return self._topiq_model

    def calculate_nima(self, image_path: str) -> Optional[float]:
        """
        计算美学评分 (使用 TOPIQ，保持接口名称兼容)

        Args:
            image_path: 图片路径

        Returns:
            美学分数 (1-10, 越高越好) 或 None (失败时)
        """
        return self.calculate_aesthetic(image_path)

    def calculate_aesthetic(self, image_path: str) -> Optional[float]:
        """
        计算 TOPIQ 美学评分

        Args:
            image_path: 图片路径

        Returns:
            美学分数 (1-10, 越高越好) 或 None (失败时)
        """
        if not os.path.exists(image_path):
            print(f"❌ 图片不存在: {image_path}")
            return None

        try:
            # 加载模型
            topiq_model = self._load_topiq()

            # 加载图片
            img = Image.open(image_path).convert('RGB')
            
            # 调整尺寸到 384x384 (TOPIQ 推荐尺寸，避免 MPS 兼容性问题)
            img = img.resize((384, 384), Image.LANCZOS)
            
            # 转为张量
            transform = T.ToTensor()
            img_tensor = transform(img).unsqueeze(0)
            
            # 确保输入张量在正确的设备上
            img_tensor = img_tensor.to(self.device)
            
            # 确保模型在正确的设备上
            topiq_model = topiq_model.to(self.device)

            # 计算评分
            with torch.no_grad():
                score = topiq_model(img_tensor, return_mos=True)

            # 转换为 Python float
            if isinstance(score, torch.Tensor):
                score = score.item()

            # 分数范围 [1, 10]
            score = float(score)
            score = max(1.0, min(10.0, score))

            return score

        except Exception as e:
            print(f"❌ TOPIQ 计算失败: {e}")
            return None

    def calculate_brisque(self, image_input) -> Optional[float]:
        """
        计算 BRISQUE 技术质量评分 (已弃用，返回 None)
        
        保留此方法以保持接口兼容性
        """
        # BRISQUE 已弃用
        return None

    def calculate_both(self,
                       full_image_path: str,
                       crop_image) -> Tuple[Optional[float], Optional[float]]:
        """
        计算美学评分 (BRISQUE 已弃用)

        Args:
            full_image_path: 全图路径 (用于美学评分)
            crop_image: 不再使用

        Returns:
            (aesthetic_score, None) 元组
        """
        aesthetic_score = self.calculate_aesthetic(full_image_path)
        return aesthetic_score, None


# 按设备存储的实例字典（支持多设备）
_iqa_scorer_instances = {}


def get_iqa_scorer(device='mps') -> IQAScorer:
    """
    获取 IQA 评分器实例（按设备创建，支持多设备）

    Args:
        device: 计算设备 ('mps', 'cuda', 'cpu')

    Returns:
        IQAScorer 实例
    """
    global _iqa_scorer_instances
    
    # 标准化设备字符串
    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = str(device)
    
    # 为每个设备创建独立的实例
    if device_str not in _iqa_scorer_instances:
        _iqa_scorer_instances[device_str] = IQAScorer(device=device)
    
    return _iqa_scorer_instances[device_str]


# 便捷函数 (保持向后兼容)
def calculate_nima(image_path: str) -> Optional[float]:
    """计算美学评分的便捷函数 (使用 TOPIQ)"""
    scorer = get_iqa_scorer()
    return scorer.calculate_aesthetic(image_path)


def calculate_brisque(image_input) -> Optional[float]:
    """计算 BRISQUE 评分 (已弃用)"""
    return None


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("IQA 评分器测试 (TOPIQ)")
    print("=" * 70)

    # 初始化评分器
    scorer = IQAScorer(device='mps')

    # 测试图片路径
    test_image = "img/_Z9W0960.jpg"

    if os.path.exists(test_image):
        print(f"\n📷 测试图片: {test_image}")

        import time
        start = time.time()
        score = scorer.calculate_aesthetic(test_image)
        elapsed = time.time() - start

        if score is not None:
            print(f"   ✅ TOPIQ 分数: {score:.2f} / 10")
            print(f"   ⏱️  耗时: {elapsed*1000:.0f}ms")
        else:
            print(f"   ❌ 评分计算失败")

    else:
        print(f"\n⚠️  测试图片不存在: {test_image}")
        print("   请提供有效的测试图片路径")

    print("\n" + "=" * 70)
