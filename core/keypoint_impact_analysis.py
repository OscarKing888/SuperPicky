#!/usr/bin/env python3
"""
关键点模型影响分析脚本
对比当前评分与引入关键点检测后的预测评分

使用方法:
    python core/keypoint_impact_analysis.py /path/to/photos
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

# 配置
MODEL_PATH = "models/cub200_keypoint_resnet50.pth"
YOLO_MODEL_PATH = "models/yolo11m-seg.pt"
IMG_SIZE = 416
VISIBILITY_THRESHOLD = 0.5  # 可见性阈值


class PartLocalizer(nn.Module):
    """关键点定位模型"""
    def __init__(self, backbone='resnet50', num_parts=3, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.num_parts = num_parts
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.coord_head = nn.Linear(hidden_dim // 2, num_parts * 2)
        self.vis_head = nn.Linear(hidden_dim // 2, num_parts)

    def forward(self, x):
        features = self.head(self.backbone(x))
        coords = torch.sigmoid(self.coord_head(features)).view(-1, self.num_parts, 2)
        vis = torch.sigmoid(self.vis_head(features))
        return coords, vis


def load_keypoint_model(device):
    """加载关键点模型"""
    print(f"📦 加载关键点模型...")
    model = PartLocalizer()
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def calculate_eye_sharpness(image, keypoint, radius=25):
    """计算眼睛区域锐度"""
    h, w = image.shape[:2]
    x, y = int(keypoint[0] * w), int(keypoint[1] * h)
    
    # 边界检查
    x1, y1 = max(0, x - radius), max(0, y - radius)
    x2, y2 = min(w, x + radius), min(h, y + radius)
    
    if x2 - x1 < 10 or y2 - y1 < 10:
        return 0.0
    
    eye_region = image[y1:y2, x1:x2]
    if eye_region.size == 0:
        return 0.0
    
    # 转灰度计算 Laplacian 方差
    if len(eye_region.shape) == 3:
        gray = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = eye_region
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    return sharpness


def analyze_image(jpg_path, yolo_model, keypoint_model, device, transform):
    """分析单张图片"""
    result = {
        'file': os.path.basename(jpg_path),
        'has_bird': False,
        'left_eye_vis': 0.0,
        'right_eye_vis': 0.0,
        'beak_vis': 0.0,
        'both_eyes_hidden': False,
        'eye_sharpness': 0.0,
        'visible_eye': None,
        'impact': 'none',
        'reason': ''
    }
    
    # YOLO 检测
    yolo_results = yolo_model(jpg_path, verbose=False)
    
    bird_detected = False
    bird_crop = None
    
    for r in yolo_results:
        if r.boxes is not None and len(r.boxes) > 0:
            for i, cls in enumerate(r.boxes.cls):
                if int(cls) == 14:  # bird class
                    bird_detected = True
                    # 获取边界框
                    box = r.boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    
                    # 读取图片并裁剪（支持 HEIF/HEIC）
                    from utils import read_image
                    img = read_image(jpg_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    bird_crop = img[y1:y2, x1:x2]
                    break
        if bird_detected:
            break
    
    if not bird_detected or bird_crop is None:
        return result
    
    result['has_bird'] = True
    
    # 关键点检测
    pil_crop = Image.fromarray(bird_crop)
    tensor = transform(pil_crop).unsqueeze(0).to(device)
    
    with torch.no_grad():
        coords, vis = keypoint_model(tensor)
    
    coords = coords[0].cpu().numpy()
    vis = vis[0].cpu().numpy()
    
    result['left_eye_vis'] = float(vis[0])
    result['right_eye_vis'] = float(vis[1])
    result['beak_vis'] = float(vis[2])
    
    # 判断双眼可见性
    left_visible = vis[0] >= VISIBILITY_THRESHOLD
    right_visible = vis[1] >= VISIBILITY_THRESHOLD
    
    result['both_eyes_hidden'] = not left_visible and not right_visible
    
    # 计算眼睛锐度
    if left_visible and right_visible:
        # 两眼都可见，取平均
        left_sharp = calculate_eye_sharpness(bird_crop, coords[0])
        right_sharp = calculate_eye_sharpness(bird_crop, coords[1])
        result['eye_sharpness'] = (left_sharp + right_sharp) / 2
        result['visible_eye'] = 'both'
    elif left_visible:
        result['eye_sharpness'] = calculate_eye_sharpness(bird_crop, coords[0])
        result['visible_eye'] = 'left'
    elif right_visible:
        result['eye_sharpness'] = calculate_eye_sharpness(bird_crop, coords[1])
        result['visible_eye'] = 'right'
    else:
        result['visible_eye'] = 'none'
    
    # 判断影响
    if result['both_eyes_hidden']:
        result['impact'] = 'downgrade'
        result['reason'] = '双眼不可见(角度不佳)'
    elif result['eye_sharpness'] > 500:  # 高锐度阈值（需要根据实际调整）
        result['impact'] = 'potential_upgrade'
        result['reason'] = f"眼睛锐度高({result['eye_sharpness']:.0f})"
    else:
        result['impact'] = 'none'
        result['reason'] = '正常'
    
    return result


def find_jpg_files(directory):
    """查找目录中的 JPG 文件（包括子目录）"""
    jpg_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.heif', '.heic', '.hif')):
                jpg_files.append(os.path.join(root, f))
    return jpg_files


def main():
    if len(sys.argv) < 2:
        print("用法: python core/keypoint_impact_analysis.py /path/to/photos")
        sys.exit(1)
    
    photo_dir = sys.argv[1]
    
    print("\n" + "="*70)
    print("🔍 关键点模型影响分析")
    print("="*70)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  设备: {device}")
    
    # 加载模型
    keypoint_model = load_keypoint_model(device)
    print(f"📦 加载 YOLO 模型...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 查找 JPG 文件
    # 先检查是否需要转换 RAW
    print(f"\n📂 扫描目录: {photo_dir}")
    
    # 优先使用已转换的 JPG（如果有的话）
    jpg_files = find_jpg_files(photo_dir)
    
    if not jpg_files:
        print("⚠️  未找到 JPG 文件")
        print("   请先运行 SuperPicky 处理目录以生成临时 JPG")
        sys.exit(1)
    
    print(f"📁 找到 {len(jpg_files)} 个 JPG 文件")
    
    # 分析每张图片
    print("\n" + "-"*70)
    print("开始分析...")
    print("-"*70)
    
    results = []
    stats = {
        'total': 0,
        'has_bird': 0,
        'both_eyes_hidden': 0,
        'potential_upgrade': 0,
        'no_change': 0
    }
    
    for i, jpg_path in enumerate(jpg_files, 1):
        print(f"\r[{i}/{len(jpg_files)}] 分析中...", end='', flush=True)
        
        try:
            result = analyze_image(jpg_path, yolo_model, keypoint_model, device, transform)
            results.append(result)
            
            stats['total'] += 1
            if result['has_bird']:
                stats['has_bird'] += 1
                if result['both_eyes_hidden']:
                    stats['both_eyes_hidden'] += 1
                elif result['impact'] == 'potential_upgrade':
                    stats['potential_upgrade'] += 1
                else:
                    stats['no_change'] += 1
        except Exception as e:
            print(f"\n⚠️  分析失败: {os.path.basename(jpg_path)} - {e}")
    
    print("\n")
    
    # 输出统计
    print("="*70)
    print("📊 影响分析统计")
    print("="*70)
    print(f"总文件数: {stats['total']}")
    print(f"检测到鸟: {stats['has_bird']}")
    print(f"")
    print(f"🔻 会降级的照片 (双眼不可见): {stats['both_eyes_hidden']} ({stats['both_eyes_hidden']/max(1,stats['has_bird'])*100:.1f}%)")
    print(f"🔺 可能升级的照片 (眼睛锐度高): {stats['potential_upgrade']} ({stats['potential_upgrade']/max(1,stats['has_bird'])*100:.1f}%)")
    print(f"➖ 不受影响的照片: {stats['no_change']} ({stats['no_change']/max(1,stats['has_bird'])*100:.1f}%)")
    
    # 输出受影响的照片详情
    print("\n" + "-"*70)
    print("📋 受影响的照片详情")
    print("-"*70)
    
    affected = [r for r in results if r['impact'] != 'none']
    
    if not affected:
        print("✅ 没有照片会受到显著影响")
    else:
        print(f"\n{'文件名':<40} {'影响':<12} {'原因':<30}")
        print("-"*82)
        for r in affected:
            impact_str = "🔻降级" if r['impact'] == 'downgrade' else "🔺可能升级"
            print(f"{r['file']:<40} {impact_str:<12} {r['reason']:<30}")
    
    # 保存详细报告
    report_path = os.path.join(photo_dir, "_keypoint_impact_report.json")
    report = {
        'timestamp': datetime.now().isoformat(),
        'directory': photo_dir,
        'stats': stats,
        'affected_files': affected,
        'all_results': results
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 详细报告已保存: {report_path}")
    print("="*70)


if __name__ == "__main__":
    main()
