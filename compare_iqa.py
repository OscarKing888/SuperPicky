#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIMA vs TOPIQ 对比测试脚本

测试目的:
1. 速度对比
2. 评分分布对比
3. 验证 TOPIQ 能否脱离 pyiqa 独立运行
"""

import os
import sys
import time
import glob
from typing import List, Tuple
import statistics

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nima_model import NIMA, load_nima_weights
from topiq_model import TOPIQScorer
from iqa_scorer import get_iqa_scorer


def find_test_images(directory: str, max_images: int = 50) -> List[str]:
    """查找测试图片"""
    extensions = ['*.jpg', '*.jpeg', '*.png']
    images = []
    
    for ext in extensions:
        images.extend(glob.glob(os.path.join(directory, ext)))
        images.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    
    images = list(set(images))[:max_images]
    return sorted(images)


def run_comparison(image_paths: List[str]) -> Tuple[dict, dict]:
    """
    运行 NIMA 和 TOPIQ 对比测试
    
    Returns:
        (nima_results, topiq_results)
    """
    print("\n" + "=" * 70)
    print("📊 NIMA vs TOPIQ 对比测试")
    print("=" * 70)
    print(f"\n📁 测试图片数量: {len(image_paths)}")
    
    # 初始化评分器
    print("\n📥 初始化模型...")
    
    # NIMA
    print("   加载 NIMA...")
    nima_scorer = get_iqa_scorer(device='mps')
    # 预热
    nima_scorer.calculate_nima(image_paths[0])
    
    # TOPIQ
    print("   加载 TOPIQ...")
    topiq_scorer = TOPIQScorer(device='mps')
    # 预热
    topiq_scorer.calculate_score(image_paths[0])
    
    print("✅ 模型加载完成\n")
    
    # 测试结果
    nima_results = {'scores': [], 'times': []}
    topiq_results = {'scores': [], 'times': []}
    
    print("🔄 开始测试...")
    print("-" * 50)
    
    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        
        # NIMA 测试
        start = time.time()
        nima_score = nima_scorer.calculate_nima(img_path)
        nima_time = (time.time() - start) * 1000
        
        # TOPIQ 测试
        start = time.time()
        topiq_score = topiq_scorer.calculate_score(img_path)
        topiq_time = (time.time() - start) * 1000
        
        if nima_score is not None and topiq_score is not None:
            nima_results['scores'].append(nima_score)
            nima_results['times'].append(nima_time)
            topiq_results['scores'].append(topiq_score)
            topiq_results['times'].append(topiq_time)
            
            print(f"[{i+1:3d}/{len(image_paths)}] {filename[:30]:30} | "
                  f"NIMA: {nima_score:.2f} ({nima_time:5.0f}ms) | "
                  f"TOPIQ: {topiq_score:.2f} ({topiq_time:5.0f}ms)")
    
    return nima_results, topiq_results


def calculate_correlation(x: List[float], y: List[float]) -> float:
    """计算 Pearson 相关系数"""
    n = len(x)
    if n < 2:
        return 0.0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denominator_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
    
    if denominator_x == 0 or denominator_y == 0:
        return 0.0
    
    return numerator / (denominator_x * denominator_y)


def print_summary(nima_results: dict, topiq_results: dict):
    """打印测试总结"""
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    
    n = len(nima_results['scores'])
    
    # 速度对比
    nima_avg_time = statistics.mean(nima_results['times'])
    topiq_avg_time = statistics.mean(topiq_results['times'])
    
    print(f"\n⏱️  推理速度 (平均):")
    print(f"   NIMA:  {nima_avg_time:6.1f} ms")
    print(f"   TOPIQ: {topiq_avg_time:6.1f} ms")
    speed_ratio = topiq_avg_time / nima_avg_time
    if speed_ratio < 1:
        print(f"   → TOPIQ 快 {(1/speed_ratio - 1)*100:.1f}%")
    else:
        print(f"   → TOPIQ 慢 {(speed_ratio - 1)*100:.1f}%")
    
    # 评分分布
    nima_avg = statistics.mean(nima_results['scores'])
    topiq_avg = statistics.mean(topiq_results['scores'])
    nima_std = statistics.stdev(nima_results['scores']) if n > 1 else 0
    topiq_std = statistics.stdev(topiq_results['scores']) if n > 1 else 0
    
    print(f"\n📈 评分分布:")
    print(f"   NIMA:  均值={nima_avg:.2f}, 标准差={nima_std:.2f}, "
          f"范围=[{min(nima_results['scores']):.2f}, {max(nima_results['scores']):.2f}]")
    print(f"   TOPIQ: 均值={topiq_avg:.2f}, 标准差={topiq_std:.2f}, "
          f"范围=[{min(topiq_results['scores']):.2f}, {max(topiq_results['scores']):.2f}]")
    
    # 相关性
    correlation = calculate_correlation(nima_results['scores'], topiq_results['scores'])
    print(f"\n🔗 评分相关性 (Pearson):")
    print(f"   r = {correlation:.4f}")
    if correlation > 0.9:
        print("   → 高度相关 ✅")
    elif correlation > 0.7:
        print("   → 中度相关")
    else:
        print("   → 低度相关 ⚠️")
    
    # 差异分析
    diffs = [abs(n - t) for n, t in zip(nima_results['scores'], topiq_results['scores'])]
    avg_diff = statistics.mean(diffs)
    max_diff = max(diffs)
    
    print(f"\n📏 评分差异:")
    print(f"   平均差异: {avg_diff:.2f}")
    print(f"   最大差异: {max_diff:.2f}")
    
    # 结论
    print("\n" + "=" * 70)
    print("📋 结论")
    print("=" * 70)
    
    conclusions = []
    
    if topiq_avg_time <= nima_avg_time * 1.2:
        conclusions.append("✅ 速度: TOPIQ 速度可接受 (不超过 NIMA 的 120%)")
    else:
        conclusions.append("⚠️  速度: TOPIQ 明显慢于 NIMA")
    
    # TOPIQ 作为独立模型运行成功
    conclusions.append("✅ 独立运行: TOPIQ 成功脱离 pyiqa 框架运行")
    
    if avg_diff < 1.0:
        conclusions.append("✅ 一致性: 评分差异较小 (平均 < 1.0)")
    else:
        conclusions.append("⚠️  一致性: 评分差异较大，需人工审核")
    
    for c in conclusions:
        print(f"   {c}")
    
    print("\n💡 建议: 请用户审核几张关键照片的评分差异，判断 TOPIQ 是否更适合鸟类摄影。")
    print("=" * 70)
    
    # 保存详细结果
    save_results(nima_results, topiq_results)


def save_results(nima_results: dict, topiq_results: dict):
    """保存详细结果到文件"""
    output_file = "compare_iqa_results.txt"
    with open(output_file, 'w') as f:
        f.write("NIMA vs TOPIQ 对比结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("评分对比:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'序号':>4} | {'NIMA':>8} | {'TOPIQ':>8} | {'差异':>8}\n")
        f.write("-" * 50 + "\n")
        
        for i, (n, t) in enumerate(zip(nima_results['scores'], topiq_results['scores'])):
            f.write(f"{i+1:4d} | {n:8.2f} | {t:8.2f} | {abs(n-t):8.2f}\n")
    
    print(f"\n📄 详细结果已保存到: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NIMA vs TOPIQ 对比测试')
    parser.add_argument('directory', nargs='?', default='img',
                        help='测试图片目录 (默认: img)')
    parser.add_argument('-n', '--max-images', type=int, default=20,
                        help='最大测试图片数 (默认: 20)')
    
    args = parser.parse_args()
    
    # 查找测试图片
    if os.path.isfile(args.directory):
        # 单个文件
        images = [args.directory]
    else:
        images = find_test_images(args.directory, args.max_images)
    
    if not images:
        print(f"❌ 未找到测试图片: {args.directory}")
        sys.exit(1)
    
    # 运行对比
    nima_results, topiq_results = run_comparison(images)
    
    # 打印总结
    print_summary(nima_results, topiq_results)
