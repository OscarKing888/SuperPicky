#!/usr/bin/env python3
"""
OSEA ResNet34 鸟类分类器

基于 OSEA 开源模型 (https://github.com/bird-feeder/OSEA)
支持 10,964 种鸟类识别

优化策略 (基于 test_preprocessing.py 实验):
- 中心裁剪预处理 (Resize 256 + CenterCrop 224): 置信度提升 ~15%
- 可选 TTA 模式 (原图 + 水平翻转): 额外提升 ~0.5%，但推理时间翻倍
"""

__version__ = "1.0.0"

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from PIL import Image
from torchvision import models, transforms


# ==================== 路径配置 ====================

def _get_birdid_dir() -> Path:
    """获取 birdid 模块目录"""
    return Path(__file__).parent


def _get_project_root() -> Path:
    """获取项目根目录"""
    return _get_birdid_dir().parent


def _get_resource_path(relative_path: str) -> Path:
    """获取资源路径 (支持 PyInstaller 打包)"""
    if getattr(sys, 'frozen', False):
        base = Path(sys._MEIPASS)
    else:
        base = _get_project_root()
    return base / relative_path


# ==================== 设备配置 ====================

def _get_device() -> torch.device:
    """获取最佳计算设备"""
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


DEVICE = _get_device()


# ==================== 预处理 transforms ====================

CENTER_CROP_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BASELINE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==================== OSEA 分类器 ====================

class OSEAClassifier:
    """
    OSEA ResNet34 鸟类分类器

    Attributes:
        model: ResNet34 模型
        bird_info: 物种信息列表 [[cn_name, en_name, scientific_name], ...]
        transform: 图像预处理 transform
        num_classes: 物种数量 (10964)
    """

    DEFAULT_MODEL_PATH = "models/model20240824.pth"
    DEFAULT_BIRD_INFO_PATH = "birdid/data/osea_bird_info.json"

    def __init__(
        self,
        model_path: Optional[str] = None,
        bird_info_path: Optional[str] = None,
        use_center_crop: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        初始化 OSEA 分类器

        Args:
            model_path: 模型文件路径 (默认: models/model20240824.pth)
            bird_info_path: 物种信息文件路径 (默认: birdid/data/osea_bird_info.json)
            use_center_crop: 是否使用中心裁剪预处理 (推荐: True)
            device: 计算设备 (默认: 自动检测)
        """
        self.device = device or DEVICE
        self.use_center_crop = use_center_crop
        self.transform = CENTER_CROP_TRANSFORM if use_center_crop else BASELINE_TRANSFORM

        self.model_path = model_path or str(_get_resource_path(self.DEFAULT_MODEL_PATH))
        self.model = self._load_model()

        self.bird_info_path = bird_info_path or str(_get_resource_path(self.DEFAULT_BIRD_INFO_PATH))
        self.bird_info = self._load_bird_info()
        self.num_classes = len(self.bird_info)

        print(f"[OSEA] 模型已加载: {self.num_classes} 物种, 设备: {self.device}")

    def _load_model(self) -> torch.nn.Module:
        """加载 ResNet34 模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"OSEA 模型未找到: {self.model_path}")

        model = models.resnet34(num_classes=11000)
        state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)

        return model

    def _load_bird_info(self) -> List[List[str]]:
        """加载物种信息 JSON"""
        if not os.path.exists(self.bird_info_path):
            raise FileNotFoundError(f"物种信息文件未找到: {self.bird_info_path}")

        with open(self.bird_info_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
        temperature: float = 1.0,
        ebird_species_set: Optional[Set[str]] = None,
    ) -> List[Dict]:
        """
        预测鸟类物种

        Args:
            image: PIL Image 对象 (RGB)
            top_k: 返回前 K 个结果
            temperature: softmax 温度参数 (1.0 为标准, <1 更尖锐, >1 更平滑)
            ebird_species_set: eBird 物种代码集合 (用于过滤)

        Returns:
            识别结果列表 [{cn_name, en_name, scientific_name, confidence, class_id}, ...]
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)[0]

        output = output[:self.num_classes]
        probs = torch.nn.functional.softmax(output / temperature, dim=0)

        k = min(100 if ebird_species_set else top_k, self.num_classes)
        top_probs, top_indices = torch.topk(probs, k)

        results = []
        for i in range(len(top_indices)):
            class_id = top_indices[i].item()
            confidence = top_probs[i].item() * 100

            min_confidence = 0.3 if ebird_species_set else 1.0
            if confidence < min_confidence:
                continue

            info = self.bird_info[class_id]
            cn_name = info[0]
            en_name = info[1]
            scientific_name = info[2] if len(info) > 2 else None

            ebird_match = False

            results.append({
                'class_id': class_id,
                'cn_name': cn_name,
                'en_name': en_name,
                'scientific_name': scientific_name,
                'confidence': confidence,
                'ebird_match': ebird_match,
            })

            if len(results) >= top_k:
                break

        return results

    def predict_with_tta(
        self,
        image: Image.Image,
        top_k: int = 5,
        temperature: float = 1.0,
        ebird_species_set: Optional[Set[str]] = None,
    ) -> List[Dict]:
        """
        使用 TTA (Test-Time Augmentation) 预测

        TTA 策略: 原图 + 水平翻转取平均
        推理时间翻倍，但可能提高准确率
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        input1 = self.transform(image).unsqueeze(0).to(self.device)

        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        input2 = self.transform(flipped).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output1 = self.model(input1)[0][:self.num_classes]
            output2 = self.model(input2)[0][:self.num_classes]

        avg_output = (output1 + output2) / 2
        probs = torch.nn.functional.softmax(avg_output / temperature, dim=0)

        k = min(100 if ebird_species_set else top_k, self.num_classes)
        top_probs, top_indices = torch.topk(probs, k)

        results = []
        for i in range(len(top_indices)):
            class_id = top_indices[i].item()
            confidence = top_probs[i].item() * 100

            min_confidence = 0.3 if ebird_species_set else 1.0
            if confidence < min_confidence:
                continue

            info = self.bird_info[class_id]
            cn_name = info[0]
            en_name = info[1]
            scientific_name = info[2] if len(info) > 2 else None

            results.append({
                'class_id': class_id,
                'cn_name': cn_name,
                'en_name': en_name,
                'scientific_name': scientific_name,
                'confidence': confidence,
                'ebird_match': False,
            })

            if len(results) >= top_k:
                break

        return results


# ==================== 全局单例 ====================

_osea_classifier: Optional[OSEAClassifier] = None


def get_osea_classifier() -> OSEAClassifier:
    """获取 OSEA 分类器单例"""
    global _osea_classifier
    if _osea_classifier is None:
        _osea_classifier = OSEAClassifier()
    return _osea_classifier


# ==================== 便捷函数 ====================

def osea_predict(image: Image.Image, top_k: int = 5) -> List[Dict]:
    """快速 OSEA 预测"""
    classifier = get_osea_classifier()
    return classifier.predict(image, top_k=top_k)


def osea_predict_file(image_path: str, top_k: int = 5) -> List[Dict]:
    """OSEA 预测 (从文件路径)"""
    from birdid.bird_identifier import load_image
    image = load_image(image_path)
    return osea_predict(image, top_k=top_k)


# ==================== 测试 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OSEA 鸟类分类器测试")
    parser.add_argument("image", help="测试图片路径")
    parser.add_argument("--top-k", type=int, default=5, help="返回前 K 个结果")
    parser.add_argument("--tta", action="store_true", help="使用 TTA 模式")
    args = parser.parse_args()

    from birdid.bird_identifier import load_image
    image = load_image(args.image)

    classifier = OSEAClassifier()

    if args.tta:
        results = classifier.predict_with_tta(image, top_k=args.top_k)
        print(f"\n[OSEA TTA 预测结果] 前 {args.top_k} 名:")
    else:
        results = classifier.predict(image, top_k=args.top_k)
        print(f"\n[OSEA 预测结果] 前 {args.top_k} 名:")

    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['cn_name']} ({r['en_name']})")
        print(f"     学名: {r['scientific_name']}")
        print(f"     置信度: {r['confidence']:.1f}%")
