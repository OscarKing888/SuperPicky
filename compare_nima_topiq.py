#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIMA vs TOPIQ 分数对比测试

对比同样的照片在两个模型下的分数差异
"""

import os
import sys
import glob
import subprocess
import time

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from topiq_model import TOPIQScorer
from nima_model import NIMA, load_nima_weights


class NIMAScorer:
    """独立的 NIMA 评分器"""
    
    def __init__(self, device='mps'):
        self.device = self._get_device(device)
        self._model = None
        
    def _get_device(self, preferred='mps'):
        if preferred == 'mps':
            try:
                if torch.backends.mps.is_available():
                    return torch.device('mps')
            except:
                pass
        if preferred == 'cuda' or torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def _load_model(self):
        if self._model is None:
            print(f"🎨 初始化 NIMA 评分器 (设备: {self.device})...")
            
            # 查找权重
            weight_paths = [
                'models/NIMA_InceptionV2_ava-b0c77c00.pth',
                os.path.join(os.path.dirname(__file__), 'models/NIMA_InceptionV2_ava-b0c77c00.pth')
            ]
            
            weight_path = None
            for p in weight_paths:
                if os.path.exists(p):
                    weight_path = p
                    break
            
            if not weight_path:
                raise FileNotFoundError("NIMA 权重文件未找到")
            
            self._model = NIMA()
            load_nima_weights(self._model, weight_path, self.device)
            self._model.to(self.device)
            self._model.eval()
            
        return self._model
    
    def calculate_score(self, image_path: str) -> float:
        """计算 NIMA 评分"""
        if not os.path.exists(image_path):
            return None
            
        try:
            model = self._load_model()
            
            img = Image.open(image_path).convert('RGB')
            # NIMA 使用 224x224 或 299x299
            img = img.resize((299, 299), Image.LANCZOS)
            
            transform = T.ToTensor()
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                score = model.predict_score(img_tensor)
            
            if isinstance(score, torch.Tensor):
                score = score.item()
            
            return float(max(1.0, min(10.0, score)))
            
        except Exception as e:
            print(f"❌ NIMA 计算失败: {e}")
            return None


def extract_preview(nef_path: str, output_dir: str) -> str:
    """从 NEF 提取预览 JPG"""
    basename = os.path.basename(nef_path).replace('.NEF', '.jpg')
    output_path = os.path.join(output_dir, basename)
    
    if os.path.exists(output_path):
        return output_path
    
    try:
        exiftool = os.path.join(os.path.dirname(__file__), 'exiftool')
        subprocess.run([exiftool, '-b', '-PreviewImage', '-w', output_dir + '/%f.jpg', nef_path],
                       capture_output=True, timeout=30)
        if os.path.exists(output_path):
            return output_path
    except:
        pass
    
    return None


def main():
    print("=" * 70)
    print("🔬 NIMA vs TOPIQ 分数对比测试")
    print("=" * 70)
    
    directory = "/Users/jameszhenyu/Desktop/2025-08-14"
    tmp_dir = "/tmp/nima_topiq_test"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 找 NEF 文件
    nef_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.upper().endswith('.NEF'):
                nef_files.append(os.path.join(root, f))
    
    # 测试全部
    nef_files = sorted(nef_files)
    
    print(f"\n📁 测试目录: {directory}")
    print(f"📊 测试文件: {len(nef_files)} 张\n")
    
    # 提取预览
    print("📥 提取预览图...")
    jpg_files = []
    for nef in nef_files:
        jpg = extract_preview(nef, tmp_dir)
        if jpg:
            jpg_files.append((nef, jpg))
    print(f"   ✅ 提取完成: {len(jpg_files)} 张\n")
    
    # 加载模型
    print("📥 加载模型...")
    yolo = YOLO('yolo11m-seg.pt')
    nima_scorer = NIMAScorer(device='mps')
    topiq_scorer = TOPIQScorer(device='mps')
    print("✅ 模型就绪\n")
    
    # 处理
    results = []
    print("-" * 80)
    print(f"{'文件名':<22} {'TOPIQ-Crop':>11} {'TOPIQ-Full':>11} {'差值':>8}")
    print("-" * 80)
    
    for nef_path, jpg_path in jpg_files:
        filename = os.path.basename(nef_path)
        
        # YOLO 检测
        yolo_results = yolo(jpg_path, verbose=False)
        if not yolo_results:
            continue
        
        result = yolo_results[0]
        bird_class = 14
        bird_idx = None
        best_conf = 0
        
        if result.boxes is not None:
            for idx, cls in enumerate(result.boxes.cls):
                if int(cls) == bird_class:
                    conf = float(result.boxes.conf[idx])
                    if conf > best_conf:
                        best_conf = conf
                        bird_idx = idx
        
        if bird_idx is None:
            continue
        
        # 裁剪鸟区域
        box = result.boxes.xyxy[bird_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        
        from PIL import Image as PILImage
        import cv2
        img = cv2.imread(jpg_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bird_crop = img_rgb[y1:y2, x1:x2]
        
        if bird_crop.size == 0:
            continue
        
        # 保存裁剪图
        crop_path = os.path.join(tmp_dir, f"crop_{os.path.basename(jpg_path)}")
        PILImage.fromarray(bird_crop).save(crop_path, quality=95)
        
        # 计算分数: Crop vs Full
        topiq_crop = topiq_scorer.calculate_score(crop_path)
        topiq_full = topiq_scorer.calculate_score(jpg_path)
        
        if topiq_crop and topiq_full:
            diff = topiq_full - topiq_crop
            results.append((filename, topiq_crop, topiq_full, diff))
            print(f"{filename[:21]:<22} {topiq_crop:>11.2f} {topiq_full:>11.2f} {diff:>+8.2f}")
    
    print("-" * 80)
    
    if results:
        crop_scores = [r[1] for r in results]
        full_scores = [r[2] for r in results]
        diffs = [r[3] for r in results]
        
        print(f"\n📊 统计结果 ({len(results)} 张有效照片):")
        print(f"   TOPIQ-Crop 范围: [{min(crop_scores):.2f}, {max(crop_scores):.2f}], 均值: {sum(crop_scores)/len(crop_scores):.2f}")
        print(f"   TOPIQ-Full 范围: [{min(full_scores):.2f}, {max(full_scores):.2f}], 均值: {sum(full_scores)/len(full_scores):.2f}")
        print(f"   差值 (Full-Crop): [{min(diffs):.2f}, {max(diffs):.2f}], 均值: {sum(diffs)/len(diffs):+.2f}")
        
        # 按 Crop 分数排序
        by_crop = sorted(results, key=lambda x: x[1], reverse=True)
        # 按 Full 分数排序
        by_full = sorted(results, key=lambda x: x[2], reverse=True)
        
        print("\n" + "=" * 80)
        print("📋 按 Crop 排名 Top 10")
        print("=" * 80)
        print(f"{'排名':<4} {'文件名':<20} {'Crop':>8} {'Full':>8} {'差值':>8}")
        print("-" * 60)
        for i, r in enumerate(by_crop[:10], 1):
            print(f"{i:<4} {r[0][:19]:<20} {r[1]:>8.2f} {r[2]:>8.2f} {r[3]:>+8.2f}")
        
        print("\n" + "=" * 80)
        print("📋 按 Full 排名 Top 10")
        print("=" * 80)
        print(f"{'排名':<4} {'文件名':<20} {'Crop':>8} {'Full':>8} {'差值':>8}")
        print("-" * 60)
        for i, r in enumerate(by_full[:10], 1):
            print(f"{i:<4} {r[0][:19]:<20} {r[1]:>8.2f} {r[2]:>8.2f} {r[3]:>+8.2f}")
        
        print("\n" + "=" * 80)
        print("📋 按 Crop 排名 Bottom 10")
        print("=" * 80)
        print(f"{'排名':<4} {'文件名':<20} {'Crop':>8} {'Full':>8} {'差值':>8}")
        print("-" * 60)
        for i, r in enumerate(by_crop[-10:], len(by_crop)-9):
            print(f"{i:<4} {r[0][:19]:<20} {r[1]:>8.2f} {r[2]:>8.2f} {r[3]:>+8.2f}")
        
        print("\n" + "=" * 80)
        print("📋 按 Full 排名 Bottom 10")
        print("=" * 80)
        print(f"{'排名':<4} {'文件名':<20} {'Crop':>8} {'Full':>8} {'差值':>8}")
        print("-" * 60)
        for i, r in enumerate(by_full[-10:], len(by_full)-9):
            print(f"{i:<4} {r[0][:19]:<20} {r[1]:>8.2f} {r[2]:>8.2f} {r[3]:>+8.2f}")
        
        # 排名差异分析 - Top 10%
        top_n = max(1, len(results) // 10)  # 10%
        crop_top_set = set(r[0] for r in by_crop[:top_n])
        full_top_set = set(r[0] for r in by_full[:top_n])
        overlap = crop_top_set & full_top_set
        
        print("\n" + "=" * 80)
        print(f"🔗 Top 10% ({top_n}张) 排名重叠分析")
        print("=" * 80)
        print(f"   重叠: {len(overlap)}/{top_n} ({len(overlap)*100//top_n}%)")
        print(f"   只在 Crop Top 10%: {len(crop_top_set - full_top_set)} 张")
        print(f"   只在 Full Top 10%: {len(full_top_set - crop_top_set)} 张")
        
        if crop_top_set - full_top_set:
            print(f"\n   只在 Crop 靠前的照片:")
            for f in sorted(crop_top_set - full_top_set):
                r = next(x for x in results if x[0] == f)
                print(f"      {f}: Crop={r[1]:.2f}, Full={r[2]:.2f}")
        
        if full_top_set - crop_top_set:
            print(f"\n   只在 Full 靠前的照片:")
            for f in sorted(full_top_set - crop_top_set):
                r = next(x for x in results if x[0] == f)
                print(f"      {f}: Crop={r[1]:.2f}, Full={r[2]:.2f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
