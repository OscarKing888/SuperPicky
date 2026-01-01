#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laplacian vs Tenengrad 锐度算法对比测试

使用 YOLO + Keypoint Detector 在真实鸟头部区域上对比
"""

import os
import sys
import glob
import time
import cv2
import numpy as np
from typing import List, Dict, Tuple
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from core.keypoint_detector import KeypointDetector


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


def calculate_laplacian_denoised(gray: np.ndarray, mask: np.ndarray) -> float:
    """方案A: Laplacian + 高斯降噪"""
    if mask.sum() == 0:
        return 0.0
    # 先降噪
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    masked_gray = cv2.bitwise_and(blurred, blurred, mask=mask)
    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    mask_pixels = mask > 0
    if mask_pixels.sum() == 0:
        return 0.0
    return float(laplacian[mask_pixels].var())


def calculate_tenengrad(gray: np.ndarray, mask: np.ndarray) -> float:
    """方案B: Tenengrad (Sobel 梯度)"""
    if mask.sum() == 0:
        return 0.0
    
    # Sobel 梯度
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 梯度幅值平方
    gradient_magnitude = gx ** 2 + gy ** 2
    
    # 只取掩码区域
    mask_pixels = mask > 0
    if mask_pixels.sum() == 0:
        return 0.0
    
    return float(gradient_magnitude[mask_pixels].mean())


def calculate_tenengrad_threshold(gray: np.ndarray, mask: np.ndarray, threshold: float = 0) -> float:
    """方案B变体: Tenengrad 带阈值 (更抗噪)"""
    if mask.sum() == 0:
        return 0.0
    
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_magnitude = gx ** 2 + gy ** 2
    
    # 应用阈值忽略小梯度（噪声）
    if threshold > 0:
        gradient_magnitude[gradient_magnitude < threshold] = 0
    
    mask_pixels = mask > 0
    if mask_pixels.sum() == 0:
        return 0.0
    
    return float(gradient_magnitude[mask_pixels].mean())


def run_comparison(image_paths: List[str]):
    """运行对比测试"""
    print("\n" + "=" * 80)
    print("🔬 Laplacian vs Tenengrad 锐度算法对比")
    print("=" * 80)
    print(f"\n📁 测试图片: {len(image_paths)} 张\n")
    
    # 加载模型
    print("📥 加载模型...")
    yolo_model = YOLO('yolo11m-seg.pt')
    kp_detector = KeypointDetector()
    print("✅ 模型就绪\n")
    
    # 收集结果
    results = []
    no_bird_count = 0
    no_eye_count = 0
    
    laplacian_times = []
    laplacian_dn_times = []
    tenengrad_times = []
    
    print("🔄 测试中...")
    print("-" * 80)
    
    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        
        # YOLO 检测
        yolo_results = yolo_model(img_path, verbose=False)
        if not yolo_results or len(yolo_results) == 0:
            no_bird_count += 1
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
            no_bird_count += 1
            continue
        
        # 获取检测框和裁剪图
        box = result.boxes.xyxy[bird_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bird_crop = img_rgb[y1:y2, x1:x2]
        
        if bird_crop.size == 0:
            continue
        
        # Keypoint 检测
        kp_result = kp_detector.detect(bird_crop)
        
        if not kp_result.visible_eye:
            no_eye_count += 1
            continue
        
        # 获取头部掩码
        h, w = bird_crop.shape[:2]
        gray = cv2.cvtColor(bird_crop, cv2.COLOR_RGB2GRAY)
        
        # 构建头部掩码 (使用眼睛位置)
        VIS_THRESH = 0.5
        left_vis = kp_result.left_eye_vis >= VIS_THRESH
        right_vis = kp_result.right_eye_vis >= VIS_THRESH
        
        if left_vis:
            eye = kp_result.left_eye
        elif right_vis:
            eye = kp_result.right_eye
        else:
            no_eye_count += 1
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
        
        # 计算三种锐度
        # 1. Laplacian (当前)
        t0 = time.time()
        lap = calculate_laplacian(gray, head_mask)
        laplacian_times.append((time.time() - t0) * 1000)
        
        # 2. Laplacian + 降噪
        t0 = time.time()
        lap_dn = calculate_laplacian_denoised(gray, head_mask)
        laplacian_dn_times.append((time.time() - t0) * 1000)
        
        # 3. Tenengrad
        t0 = time.time()
        tene = calculate_tenengrad(gray, head_mask)
        tenengrad_times.append((time.time() - t0) * 1000)
        
        results.append({
            'file': filename,
            'laplacian': lap,
            'laplacian_dn': lap_dn,
            'tenengrad': tene,
        })
        
        if (i + 1) % 10 == 0:
            print(f"[{i+1:3d}/{len(image_paths)}] 已处理...")
    
    print("-" * 80)
    print(f"✅ 完成: {len(results)} 有效, {no_bird_count} 无鸟, {no_eye_count} 无眼\n")
    
    if not results:
        print("❌ 没有有效结果")
        return
    
    # 分析结果
    print_analysis(results, laplacian_times, laplacian_dn_times, tenengrad_times)


def print_analysis(results: List[Dict], lap_times, lap_dn_times, tene_times):
    """打印分析结果"""
    print("=" * 80)
    print("📊 分析结果")
    print("=" * 80)
    
    lap_scores = [r['laplacian'] for r in results]
    lap_dn_scores = [r['laplacian_dn'] for r in results]
    tene_scores = [r['tenengrad'] for r in results]
    
    # 速度对比
    print("\n⏱️  速度对比 (平均):")
    print(f"   Laplacian:        {statistics.mean(lap_times):.3f} ms")
    print(f"   Laplacian+降噪:   {statistics.mean(lap_dn_times):.3f} ms")
    print(f"   Tenengrad:        {statistics.mean(tene_times):.3f} ms")
    
    # 数值范围
    print("\n📈 数值范围:")
    print(f"   Laplacian:        [{min(lap_scores):.0f}, {max(lap_scores):.0f}], 均值={statistics.mean(lap_scores):.0f}")
    print(f"   Laplacian+降噪:   [{min(lap_dn_scores):.0f}, {max(lap_dn_scores):.0f}], 均值={statistics.mean(lap_dn_scores):.0f}")
    print(f"   Tenengrad:        [{min(tene_scores):.0f}, {max(tene_scores):.0f}], 均值={statistics.mean(tene_scores):.0f}")
    
    # 相关性分析
    def pearson(x, y):
        n = len(x)
        if n < 2:
            return 0
        mx, my = sum(x)/n, sum(y)/n
        num = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))
        dx = sum((xi-mx)**2 for xi in x) ** 0.5
        dy = sum((yi-my)**2 for yi in y) ** 0.5
        if dx == 0 or dy == 0:
            return 0
        return num / (dx * dy)
    
    corr_lap_tene = pearson(lap_scores, tene_scores)
    corr_lap_lapdn = pearson(lap_scores, lap_dn_scores)
    
    print("\n🔗 相关性:")
    print(f"   Laplacian vs Tenengrad:    r = {corr_lap_tene:.4f}")
    print(f"   Laplacian vs Laplacian+降噪: r = {corr_lap_lapdn:.4f}")
    
    # 样本展示
    print("\n📝 样本对比 (前10张):")
    print(f"{'文件':<25} {'Laplacian':>12} {'Lap+降噪':>12} {'Tenengrad':>12}")
    print("-" * 65)
    for r in results[:10]:
        print(f"{r['file'][:24]:<25} {r['laplacian']:>12.0f} {r['laplacian_dn']:>12.0f} {r['tenengrad']:>12.0f}")
    
    # 结论
    print("\n" + "=" * 80)
    print("💡 结论")
    print("=" * 80)
    
    if corr_lap_tene > 0.9:
        print("   两种算法高度相关，可以互相替代")
    elif corr_lap_tene > 0.7:
        print("   两种算法有良好相关性，但排序可能略有不同")
    else:
        print("   两种算法相关性较低，会产生不同的排序结果")
    
    if statistics.mean(tene_times) <= statistics.mean(lap_times) * 1.2:
        print("   Tenengrad 速度可接受")
    else:
        print("   Tenengrad 明显更慢")
    
    print("=" * 80)


if __name__ == "__main__":
    # 使用之前的测试图片
    test_dir = "/tmp/topiq_test"
    
    images = glob.glob(os.path.join(test_dir, "*.jpg"))
    if not images:
        print(f"❌ 未找到测试图片: {test_dir}")
        print("   请先运行 compare_iqa_crop.py 生成测试图片")
        sys.exit(1)
    
    images = sorted(images)[:50]  # 取50张
    run_comparison(images)
