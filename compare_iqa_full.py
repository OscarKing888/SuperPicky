#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIMA vs TOPIQ 完整对比测试 - 带百分比分区统计

按分数变化百分比分区:
- ±5% 以内 (稳定)
- +5% 以上 (TOPIQ 更高)
- -5% 以下 (TOPIQ 更低)
- ±25% 以上 (极端差异)
"""

import os
import sys
import time
import glob
from typing import List, Dict
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topiq_model import TOPIQScorer
from iqa_scorer import get_iqa_scorer


def find_test_images(directory: str, max_images: int = 100) -> List[str]:
    extensions = ['*.jpg', '*.jpeg', '*.png']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(directory, ext)))
    images = list(set(images))[:max_images]
    return sorted(images)


def categorize_by_percentage(nima: float, topiq: float) -> str:
    """按百分比变化分类"""
    if nima == 0:
        return "error"
    
    pct_change = ((topiq - nima) / nima) * 100
    
    if abs(pct_change) <= 5:
        return "stable"  # ±5%
    elif pct_change > 25:
        return "extreme_high"  # +25% 以上
    elif pct_change < -25:
        return "extreme_low"  # -25% 以下
    elif pct_change > 5:
        return "higher"  # +5% 到 +25%
    else:
        return "lower"  # -5% 到 -25%


def run_full_comparison(image_paths: List[str]) -> Dict:
    """运行完整对比测试"""
    print("\n" + "=" * 80)
    print("📊 NIMA vs TOPIQ 完整对比测试 (百分比分区)")
    print("=" * 80)
    print(f"\n📁 测试图片数量: {len(image_paths)}")
    
    # 初始化
    print("\n📥 加载模型...")
    nima_scorer = get_iqa_scorer(device='mps')
    nima_scorer.calculate_nima(image_paths[0])  # 预热
    
    topiq_scorer = TOPIQScorer(device='mps')
    topiq_scorer.calculate_score(image_paths[0])  # 预热
    
    print("✅ 模型就绪\n")
    
    # 收集结果
    results = []
    categories = {
        'stable': [],      # ±5%
        'higher': [],      # +5% ~ +25%
        'lower': [],       # -5% ~ -25%
        'extreme_high': [],  # >+25%
        'extreme_low': [],   # <-25%
    }
    
    nima_times = []
    topiq_times = []
    
    print("🔄 测试中...")
    print("-" * 80)
    
    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        
        # NIMA
        start = time.time()
        nima_score = nima_scorer.calculate_nima(img_path)
        nima_time = (time.time() - start) * 1000
        
        # TOPIQ
        start = time.time()
        topiq_score = topiq_scorer.calculate_score(img_path)
        topiq_time = (time.time() - start) * 1000
        
        if nima_score and topiq_score:
            pct = ((topiq_score - nima_score) / nima_score) * 100
            cat = categorize_by_percentage(nima_score, topiq_score)
            
            result = {
                'file': filename,
                'nima': nima_score,
                'topiq': topiq_score,
                'pct_change': pct,
                'category': cat
            }
            results.append(result)
            categories[cat].append(result)
            
            nima_times.append(nima_time)
            topiq_times.append(topiq_time)
            
            # 每10张打印进度
            if (i + 1) % 10 == 0:
                print(f"[{i+1:3d}/{len(image_paths)}] 已完成...")
    
    print("-" * 80)
    print(f"✅ 测试完成: {len(results)} 张图片\n")
    
    return {
        'results': results,
        'categories': categories,
        'nima_times': nima_times,
        'topiq_times': topiq_times
    }


def print_detailed_report(data: Dict):
    """打印详细报告"""
    results = data['results']
    categories = data['categories']
    nima_times = data['nima_times']
    topiq_times = data['topiq_times']
    
    total = len(results)
    
    print("=" * 80)
    print("📋 分区统计报告")
    print("=" * 80)
    
    # ===== 分区统计 =====
    print("\n📊 分数变化分区统计 (以 NIMA 为基准)")
    print("-" * 60)
    
    stable = len(categories['stable'])
    higher = len(categories['higher'])
    lower = len(categories['lower'])
    extreme_high = len(categories['extreme_high'])
    extreme_low = len(categories['extreme_low'])
    
    print(f"\n🟢 ±5% 稳定区:   {stable:3d} 张 ({stable/total*100:5.1f}%)")
    print(f"   (NIMA 和 TOPIQ 评分非常接近)")
    
    print(f"\n🔵 +5%~+25%:     {higher:3d} 张 ({higher/total*100:5.1f}%)")
    print(f"   (TOPIQ 评分略高于 NIMA)")
    
    print(f"\n🟡 -5%~-25%:     {lower:3d} 张 ({lower/total*100:5.1f}%)")
    print(f"   (TOPIQ 评分略低于 NIMA)")
    
    print(f"\n🔴 极端差异 (>±25%):")
    print(f"   +25% 以上:   {extreme_high:3d} 张 ({extreme_high/total*100:5.1f}%)")
    print(f"   -25% 以下:   {extreme_low:3d} 张 ({extreme_low/total*100:5.1f}%)")
    
    # ===== 速度统计 =====
    print("\n" + "-" * 60)
    print("⏱️  速度统计")
    print("-" * 60)
    print(f"   NIMA 平均:  {statistics.mean(nima_times):6.1f} ms")
    print(f"   TOPIQ 平均: {statistics.mean(topiq_times):6.1f} ms")
    ratio = statistics.mean(topiq_times) / statistics.mean(nima_times)
    if ratio < 1:
        print(f"   → TOPIQ 快 {(1 - ratio) * 100:.1f}%")
    else:
        print(f"   → TOPIQ 慢 {(ratio - 1) * 100:.1f}%")
    
    # ===== 评分统计 =====
    print("\n" + "-" * 60)
    print("📈 评分统计")
    print("-" * 60)
    nima_scores = [r['nima'] for r in results]
    topiq_scores = [r['topiq'] for r in results]
    
    print(f"   NIMA:  均值={statistics.mean(nima_scores):.2f}, "
          f"标准差={statistics.stdev(nima_scores):.2f}")
    print(f"   TOPIQ: 均值={statistics.mean(topiq_scores):.2f}, "
          f"标准差={statistics.stdev(topiq_scores):.2f}")
    
    # ===== 极端案例 =====
    if extreme_high or extreme_low:
        print("\n" + "-" * 60)
        print("⚠️  极端差异案例 (>±25%)")
        print("-" * 60)
        
        extremes = categories['extreme_high'] + categories['extreme_low']
        extremes.sort(key=lambda x: abs(x['pct_change']), reverse=True)
        
        for r in extremes[:10]:
            print(f"   {r['file'][:35]:35} | NIMA: {r['nima']:.2f} → TOPIQ: {r['topiq']:.2f} ({r['pct_change']:+.1f}%)")
    
    # ===== 样本展示 =====
    print("\n" + "-" * 60)
    print("📝 各分区样本 (每区最多5张)")
    print("-" * 60)
    
    for cat_name, cat_label in [
        ('stable', '±5% 稳定'),
        ('higher', '+5%~+25%'),
        ('lower', '-5%~-25%')
    ]:
        cat_items = categories[cat_name][:5]
        if cat_items:
            print(f"\n【{cat_label}】")
            for r in cat_items:
                print(f"   {r['file'][:30]:30} | NIMA: {r['nima']:.2f} | TOPIQ: {r['topiq']:.2f} | {r['pct_change']:+.1f}%")
    
    # ===== 结论 =====
    print("\n" + "=" * 80)
    print("📋 结论")
    print("=" * 80)
    
    stable_pct = stable / total * 100
    topiq_higher_pct = (higher + extreme_high) / total * 100
    topiq_lower_pct = (lower + extreme_low) / total * 100
    
    print(f"\n   📊 整体趋势:")
    print(f"      - {stable_pct:.1f}% 的照片评分差异在 ±5% 内 (稳定)")
    print(f"      - {topiq_higher_pct:.1f}% 的照片 TOPIQ 评分更高")
    print(f"      - {topiq_lower_pct:.1f}% 的照片 TOPIQ 评分更低")
    
    if topiq_higher_pct > topiq_lower_pct:
        print(f"\n   💡 TOPIQ 整体倾向给出更高的美学评分。")
    else:
        print(f"\n   💡 TOPIQ 整体倾向给出更低的美学评分。")
    
    print("=" * 80)
    
    # 保存详细 CSV
    save_csv(results)


def save_csv(results: List[Dict]):
    """保存详细 CSV 结果"""
    output = "compare_full_results.csv"
    with open(output, 'w') as f:
        f.write("filename,nima,topiq,pct_change,category\n")
        for r in results:
            f.write(f"{r['file']},{r['nima']:.2f},{r['topiq']:.2f},{r['pct_change']:.2f},{r['category']}\n")
    print(f"\n📄 详细结果已保存: {output}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NIMA vs TOPIQ 完整对比测试')
    parser.add_argument('directory', help='测试图片目录')
    parser.add_argument('-n', '--max-images', type=int, default=100,
                        help='最大测试图片数 (默认: 100)')
    
    args = parser.parse_args()
    
    images = find_test_images(args.directory, args.max_images)
    
    if not images:
        print(f"❌ 未找到测试图片: {args.directory}")
        sys.exit(1)
    
    data = run_full_comparison(images)
    print_detailed_report(data)
