#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIMA vs TOPIQ 对比测试 - 使用 YOLO 裁剪鸟类区域

这是更准确的测试方式，因为实际使用时 IQA 评估的是裁剪后的鸟类区域
"""

import os
import sys
import time
import glob
import statistics
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import torch
import torchvision.transforms as T

from topiq_model import TOPIQScorer
from iqa_scorer import get_iqa_scorer


def load_yolo_model():
    """加载 YOLO 分割模型"""
    print("📥 加载 YOLO 分割模型...")
    model_path = os.path.join(os.path.dirname(__file__), 'yolo11m-seg.pt')
    model = YOLO(model_path)
    print("✅ YOLO 模型就绪")
    return model


def detect_and_crop_bird(yolo_model, image_path: str, output_dir: str) -> str:
    """
    使用 YOLO 检测鸟类并裁剪
    
    Returns:
        裁剪后图片的路径，或 None 如果没检测到鸟
    """
    results = yolo_model(image_path, verbose=False)
    
    if not results or len(results) == 0:
        return None
    
    result = results[0]
    
    # 查找鸟类 (class 14 in COCO)
    bird_class = 14
    bird_boxes = []
    
    if result.boxes is not None:
        for i, cls in enumerate(result.boxes.cls):
            if int(cls) == bird_class:
                box = result.boxes.xyxy[i].cpu().numpy()
                conf = float(result.boxes.conf[i])
                bird_boxes.append((box, conf))
    
    if not bird_boxes:
        return None
    
    # 选择置信度最高的鸟
    bird_boxes.sort(key=lambda x: x[1], reverse=True)
    best_box = bird_boxes[0][0]
    
    # 裁剪
    x1, y1, x2, y2 = map(int, best_box)
    
    # 扩展边界 10%
    img = Image.open(image_path)
    w, h = img.size
    margin_x = int((x2 - x1) * 0.1)
    margin_y = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)
    
    crop = img.crop((x1, y1, x2, y2))
    
    # 保存裁剪图
    basename = os.path.basename(image_path)
    crop_path = os.path.join(output_dir, f"crop_{basename}")
    crop.save(crop_path, quality=95)
    
    return crop_path


def run_cropped_comparison(image_paths: List[str], crop_dir: str) -> Dict:
    """运行裁剪图对比测试"""
    print("\n" + "=" * 80)
    print("📊 NIMA vs TOPIQ 对比测试 (YOLO 裁剪鸟类区域)")
    print("=" * 80)
    print(f"\n📁 源图片数量: {len(image_paths)}")
    
    # 加载模型
    yolo_model = load_yolo_model()
    
    print("\n📥 加载 IQA 模型...")
    nima_scorer = get_iqa_scorer(device='mps')
    topiq_scorer = TOPIQScorer(device='mps')
    
    # 预热
    if os.path.exists(image_paths[0]):
        nima_scorer.calculate_nima(image_paths[0])
        topiq_scorer.calculate_score(image_paths[0])
    
    print("✅ 所有模型就绪\n")
    
    # 创建裁剪目录
    os.makedirs(crop_dir, exist_ok=True)
    
    # 收集结果
    results = []
    nima_times = []
    topiq_times = []
    no_bird_count = 0
    
    print("🔄 检测鸟类并裁剪...")
    print("-" * 80)
    
    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        
        # YOLO 检测和裁剪
        crop_path = detect_and_crop_bird(yolo_model, img_path, crop_dir)
        
        if crop_path is None:
            no_bird_count += 1
            continue
        
        # NIMA 评分
        start = time.time()
        nima_score = nima_scorer.calculate_nima(crop_path)
        nima_time = (time.time() - start) * 1000
        
        # TOPIQ 评分
        start = time.time()
        topiq_score = topiq_scorer.calculate_score(crop_path)
        topiq_time = (time.time() - start) * 1000
        
        if nima_score and topiq_score:
            pct = ((topiq_score - nima_score) / nima_score) * 100
            
            # 分类
            if abs(pct) <= 5:
                cat = "stable"
            elif pct > 25:
                cat = "extreme_high"
            elif pct < -25:
                cat = "extreme_low"
            elif pct > 5:
                cat = "higher"
            else:
                cat = "lower"
            
            results.append({
                'file': filename,
                'nima': nima_score,
                'topiq': topiq_score,
                'pct_change': pct,
                'category': cat
            })
            
            nima_times.append(nima_time)
            topiq_times.append(topiq_time)
        
        # 进度
        if (i + 1) % 10 == 0:
            print(f"[{i+1:3d}/{len(image_paths)}] 已处理...")
    
    print("-" * 80)
    print(f"✅ 完成: {len(results)} 张有鸟, {no_bird_count} 张无鸟\n")
    
    return {
        'results': results,
        'nima_times': nima_times,
        'topiq_times': topiq_times,
        'no_bird_count': no_bird_count
    }


def print_report(data: Dict):
    """打印报告"""
    results = data['results']
    nima_times = data['nima_times']
    topiq_times = data['topiq_times']
    
    if not results:
        print("❌ 没有有效结果")
        return
    
    total = len(results)
    
    # 分类统计
    cats = {}
    for r in results:
        cats[r['category']] = cats.get(r['category'], 0) + 1
    
    print("=" * 80)
    print("📋 裁剪图对比报告")
    print("=" * 80)
    
    stable = cats.get('stable', 0)
    higher = cats.get('higher', 0)
    lower = cats.get('lower', 0)
    extreme_high = cats.get('extreme_high', 0)
    extreme_low = cats.get('extreme_low', 0)
    
    print(f"\n📊 分区统计 (以 NIMA 为基准):")
    print(f"   🟢 ±5% 稳定:    {stable:3d} 张 ({stable/total*100:5.1f}%)")
    print(f"   🔵 +5%~+25%:    {higher:3d} 张 ({higher/total*100:5.1f}%)")
    print(f"   🟡 -5%~-25%:    {lower:3d} 张 ({lower/total*100:5.1f}%)")
    print(f"   🔴 >+25%:       {extreme_high:3d} 张 ({extreme_high/total*100:5.1f}%)")
    print(f"   🔴 <-25%:       {extreme_low:3d} 张 ({extreme_low/total*100:5.1f}%)")
    
    # 速度
    print(f"\n⏱️  速度:")
    print(f"   NIMA:  {statistics.mean(nima_times):5.1f} ms")
    print(f"   TOPIQ: {statistics.mean(topiq_times):5.1f} ms")
    
    # 评分
    nima_scores = [r['nima'] for r in results]
    topiq_scores = [r['topiq'] for r in results]
    
    print(f"\n📈 评分统计:")
    print(f"   NIMA:  均值={statistics.mean(nima_scores):.2f}, "
          f"范围=[{min(nima_scores):.2f}, {max(nima_scores):.2f}]")
    print(f"   TOPIQ: 均值={statistics.mean(topiq_scores):.2f}, "
          f"范围=[{min(topiq_scores):.2f}, {max(topiq_scores):.2f}]")
    
    # 排名相关性
    nima_rank = sorted(range(total), key=lambda i: nima_scores[i], reverse=True)
    topiq_rank = sorted(range(total), key=lambda i: topiq_scores[i], reverse=True)
    
    rank_nima = [0] * total
    rank_topiq = [0] * total
    for i, idx in enumerate(nima_rank):
        rank_nima[idx] = i
    for i, idx in enumerate(topiq_rank):
        rank_topiq[idx] = i
    
    d_squared = sum((rank_nima[i] - rank_topiq[i])**2 for i in range(total))
    spearman = 1 - (6 * d_squared) / (total * (total**2 - 1))
    
    print(f"\n🔗 排名相关性 (Spearman): ρ = {spearman:.4f}")
    
    # Top 10 重叠
    top10_nima = set(nima_rank[:10])
    top10_topiq = set(topiq_rank[:10])
    overlap = len(top10_nima & top10_topiq)
    print(f"🏆 Top 10 重叠: {overlap}/10")
    
    # 保存 CSV
    with open('compare_crop_results.csv', 'w') as f:
        f.write("filename,nima,topiq,pct_change,category\n")
        for r in results:
            f.write(f"{r['file']},{r['nima']:.2f},{r['topiq']:.2f},"
                    f"{r['pct_change']:.2f},{r['category']}\n")
    
    print(f"\n📄 已保存: compare_crop_results.csv")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO 裁剪鸟类区域对比测试')
    parser.add_argument('directory', help='测试图片目录')
    parser.add_argument('-n', '--max-images', type=int, default=100,
                        help='最大测试数量')
    
    args = parser.parse_args()
    
    # 查找图片
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(glob.glob(os.path.join(args.directory, ext)))
    images = sorted(list(set(images)))[:args.max_images]
    
    if not images:
        print(f"❌ 未找到图片: {args.directory}")
        sys.exit(1)
    
    # 运行测试
    crop_dir = "/tmp/topiq_crops"
    data = run_cropped_comparison(images, crop_dir)
    print_report(data)
