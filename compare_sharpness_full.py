#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laplacian vs Tenengrad (对数归一化) 完整对比测试

按正常流程处理 /Users/jameszhenyu/Desktop/2025-08-14 目录
同时计算两种锐度，对比三星照片的 Top 25 排名
"""

import os
import sys
import glob
import time
import math
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from core.keypoint_detector import KeypointDetector
from topiq_model import TOPIQScorer


# ============================================================
# 锐度计算函数
# ============================================================

def calculate_laplacian(gray: np.ndarray, mask: np.ndarray) -> float:
    """当前方法: Laplacian 方差"""
    if mask.sum() == 0:
        return 0.0
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    mask_pixels = mask > 0
    if mask_pixels.sum() == 0:
        return 0.0
    return float(laplacian[mask_pixels].var())


def calculate_tenengrad_raw(gray: np.ndarray, mask: np.ndarray) -> float:
    """Tenengrad 原始值"""
    if mask.sum() == 0:
        return 0.0
    
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = gx ** 2 + gy ** 2
    
    mask_pixels = mask > 0
    if mask_pixels.sum() == 0:
        return 0.0
    
    return float(gradient_magnitude[mask_pixels].mean())


def normalize_tenengrad(raw_value: float) -> float:
    """
    对数归一化 Tenengrad 到 0-1000 范围
    
    基于测试数据 [1460, 154016] 映射到 [0, 1000]
    """
    MIN_VAL = 1460
    MAX_VAL = 154016
    
    if raw_value <= MIN_VAL:
        return 0.0
    if raw_value >= MAX_VAL:
        return 1000.0
    
    log_val = math.log(raw_value) - math.log(MIN_VAL)
    log_max = math.log(MAX_VAL) - math.log(MIN_VAL)
    
    return (log_val / log_max) * 1000.0


def calculate_tenengrad_normalized(gray: np.ndarray, mask: np.ndarray) -> float:
    """Tenengrad 对数归一化到 0-1000"""
    raw = calculate_tenengrad_raw(gray, mask)
    return normalize_tenengrad(raw)


# ============================================================
# 主测试流程
# ============================================================

@dataclass
class PhotoResult:
    filename: str
    laplacian: float
    tenengrad_raw: float
    tenengrad_norm: float
    nima: float
    rating: int  # 0, 1, 2, 3


def extract_preview_jpg(nef_path: str, output_dir: str) -> Optional[str]:
    """从 NEF 提取预览 JPG"""
    import subprocess
    
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


def process_directory(directory: str, max_files: int = 200) -> List[PhotoResult]:
    """处理目录，返回结果列表"""
    
    print("\n" + "=" * 80)
    print("🔬 Laplacian vs Tenengrad (对数归一化) 完整对比测试")
    print("=" * 80)
    print(f"\n📁 测试目录: {directory}")
    
    # 准备临时目录
    tmp_dir = "/tmp/sharpness_test"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 查找 NEF 文件 (包括子目录)
    nef_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.upper().endswith('.NEF'):
                nef_files.append(os.path.join(root, f))
    nef_files = sorted(nef_files)
    if not nef_files:
        print("❌ 未找到 NEF 文件")
        return []
    
    import random
    random.seed(42)
    # Quick Check Mode: Limit to 10 files
    nef_files = nef_files[:10]
    
    print(f"📊 抽样 {len(nef_files)} 张照片\n")
    
    # 提取预览
    print("📥 提取预览图...")
    jpg_files = []
    for nef in nef_files:
        jpg = extract_preview_jpg(nef, tmp_dir)
        if jpg:
            jpg_files.append((nef, jpg))
    print(f"   ✅ 提取完成: {len(jpg_files)} 张\n")
    
    # 加载模型
    print("📥 加载模型...")
    yolo_model = YOLO('yolo11m-seg.pt')
    kp_detector = KeypointDetector()
    topiq_scorer = TOPIQScorer(device='mps')
    print("✅ 模型就绪\n")
    
    # 处理
    results = []
    no_bird = 0
    no_eye = 0
    
    NIMA_THRESHOLD = 5.4  # TOPIQ 阈值
    SHARPNESS_THRESHOLD = 500  # 锐度阈值
    VIS_THRESH = 0.5
    
    print("🔄 处理中...")
    print("-" * 80)
    
    for i, (nef_path, jpg_path) in enumerate(jpg_files):
        filename = os.path.basename(nef_path)
        
        # YOLO 检测
        yolo_results = yolo_model(jpg_path, verbose=False)
        if not yolo_results or len(yolo_results) == 0:
            no_bird += 1
            continue
        
        result = yolo_results[0]
        
        # 找鸟
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
            no_bird += 1
            continue
        
        # 裁剪鸟区域
        box = result.boxes.xyxy[bird_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        
        img = cv2.imread(jpg_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]
        bird_crop = img_rgb[y1:y2, x1:x2]
        
        # 提取 Mask
        bird_crop_mask = None
        if hasattr(result, 'masks') and result.masks is not None:
             try:
                # 获取 mask (原始尺寸可能不同，ultralytics通常返回原图尺寸或缩放后的)
                raw_mask = result.masks.data[bird_idx].cpu().numpy()
                if raw_mask.shape != (h_orig, w_orig):
                     raw_mask = cv2.resize(raw_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                
                bird_mask = (raw_mask > 0.5).astype(np.uint8) * 255
                bird_crop_mask = bird_mask[y1:y2, x1:x2]
             except:
                pass
        
        if bird_crop.size == 0:
            continue
        
        # Keypoint 检测 (传入 seg_mask)
        kp_result = kp_detector.detect(bird_crop, seg_mask=bird_crop_mask)
        
        if kp_result is None or kp_result.visible_eye is None:
            no_eye += 1
            continue
        
        # 构建头部掩码
        h, w = bird_crop.shape[:2]
        gray = cv2.cvtColor(bird_crop, cv2.COLOR_RGB2GRAY)
        
        left_vis = kp_result.left_eye_vis >= VIS_THRESH
        right_vis = kp_result.right_eye_vis >= VIS_THRESH
        
        if left_vis:
            eye = kp_result.left_eye
        elif right_vis:
            eye = kp_result.right_eye
        else:
            no_eye += 1
            continue
        
        eye_px = (int(eye[0] * w), int(eye[1] * h))
        
        beak_visible = kp_result.beak_vis >= VIS_THRESH
        if beak_visible:
            beak = kp_result.beak
            beak_px = (int(beak[0] * w), int(beak[1] * h))
            radius = int(np.sqrt((eye_px[0] - beak_px[0])**2 + (eye_px[1] - beak_px[1])**2) * 1.2)
        else:
            radius = int(max(w, h) * 0.15)
        
        radius = max(10, min(radius, min(w, h) // 2))
        
        head_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(head_mask, eye_px, radius, 255, -1)
        
        # 计算锐度
        lap = calculate_laplacian(gray, head_mask)
        tene_raw = calculate_tenengrad_raw(gray, head_mask)
        tene_norm = calculate_tenengrad_normalized(gray, head_mask)
        
        # 保存裁剪图用于 TOPIQ
        crop_path = os.path.join(tmp_dir, f"crop_{os.path.basename(jpg_path)}")
        Image.fromarray(bird_crop).save(crop_path, quality=95)
        
        # TOPIQ 评分
        nima = topiq_scorer.calculate_score(crop_path)
        if nima is None:
            nima = 0.0
        
        # 评星 (使用 Laplacian 作为基准)
        if lap >= SHARPNESS_THRESHOLD and nima >= NIMA_THRESHOLD:
            rating = 3
        elif lap >= SHARPNESS_THRESHOLD or nima >= NIMA_THRESHOLD:
            rating = 2
        else:
            rating = 1
        
        results.append(PhotoResult(
            filename=filename,
            laplacian=lap,
            tenengrad_raw=tene_raw,
            tenengrad_norm=tene_norm,
            nima=nima,
            rating=rating
        ))
        
        if (i + 1) % 20 == 0:
            print(f"[{i+1:3d}/{len(jpg_files)}] 已处理...")
    
    print("-" * 80)
    print(f"✅ 完成: {len(results)} 有效, {no_bird} 无鸟, {no_eye} 无眼\n")
    
    return results


def analyze_results(results: List[PhotoResult]):
    """分析结果"""
    
    if not results:
        print("❌ 无结果可分析")
        return
    
    print("=" * 80)
    print("📊 分析结果")
    print("=" * 80)
    
    # 筛选三星照片
    three_star = [r for r in results if r.rating == 3]
    print(f"\n⭐⭐⭐ 三星照片: {len(three_star)} 张")
    
    if len(three_star) < 5:
        print("   三星照片太少，降低标准重新计算...")
        # 按美学和锐度排序取前 50%
        results_sorted = sorted(results, key=lambda r: (r.nima, r.laplacian), reverse=True)
        three_star = results_sorted[:max(25, len(results)//2)]
        print(f"   使用 Top {len(three_star)} 张作为三星\n")
    
    # 按 Laplacian 排序
    lap_sorted = sorted(three_star, key=lambda r: r.laplacian, reverse=True)[:25]
    
    # 按 Tenengrad (归一化) 排序
    tene_sorted = sorted(three_star, key=lambda r: r.tenengrad_norm, reverse=True)[:25]
    
    # 打印 Laplacian Top 25
    print("\n" + "=" * 80)
    print("📋 Laplacian 锐度 Top 25")
    print("=" * 80)
    print(f"{'排名':<4} {'文件名':<20} {'Laplacian':>10} {'Tene(归一)':>12} {'NIMA':>8}")
    print("-" * 60)
    for i, r in enumerate(lap_sorted, 1):
        print(f"{i:<4} {r.filename[:19]:<20} {r.laplacian:>10.0f} {r.tenengrad_norm:>12.0f} {r.nima:>8.2f}")
    
    # 打印 Tenengrad Top 25
    print("\n" + "=" * 80)
    print("📋 Tenengrad (归一化) 锐度 Top 25")
    print("=" * 80)
    print(f"{'排名':<4} {'文件名':<20} {'Tene(归一)':>12} {'Laplacian':>10} {'NIMA':>8}")
    print("-" * 60)
    for i, r in enumerate(tene_sorted, 1):
        print(f"{i:<4} {r.filename[:19]:<20} {r.tenengrad_norm:>12.0f} {r.laplacian:>10.0f} {r.nima:>8.2f}")
    
    # 计算 Top 25 重叠
    lap_set = set(r.filename for r in lap_sorted)
    tene_set = set(r.filename for r in tene_sorted)
    overlap = lap_set & tene_set
    
    only_lap = lap_set - tene_set
    only_tene = tene_set - lap_set
    
    print("\n" + "=" * 80)
    print("🔗 排名对比分析")
    print("=" * 80)
    print(f"   Top 25 重叠: {len(overlap)}/25 ({len(overlap)/25*100:.0f}%)")
    print(f"   只在 Laplacian Top 25: {len(only_lap)} 张")
    print(f"   只在 Tenengrad Top 25: {len(only_tene)} 张")
    
    if only_lap:
        print(f"\n   只在 Laplacian 排名靠前:")
        for f in list(only_lap)[:5]:
            r = next(x for x in results if x.filename == f)
            print(f"      {f}: Lap={r.laplacian:.0f}, Tene={r.tenengrad_norm:.0f}")
    
    if only_tene:
        print(f"\n   只在 Tenengrad 排名靠前:")
        for f in list(only_tene)[:5]:
            r = next(x for x in results if x.filename == f)
            print(f"      {f}: Lap={r.laplacian:.0f}, Tene={r.tenengrad_norm:.0f}")
    
    # 统计
    print("\n" + "=" * 80)
    print("📈 归一化效果验证")
    print("=" * 80)
    
    all_lap = [r.laplacian for r in results]
    all_tene = [r.tenengrad_norm for r in results]
    
    print(f"   Laplacian 范围:    [{min(all_lap):.0f}, {max(all_lap):.0f}]")
    print(f"   Tenengrad 归一化:  [{min(all_tene):.0f}, {max(all_tene):.0f}]")
    print(f"   阈值=500 时:")
    print(f"      Laplacian >= 500: {sum(1 for x in all_lap if x >= 500)} 张")
    print(f"      Tenengrad >= 500: {sum(1 for x in all_tene if x >= 500)} 张")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    directory = "/Users/jameszhenyu/Desktop/2025-08-14"
    results = process_directory(directory, max_files=200)
    analyze_results(results)
