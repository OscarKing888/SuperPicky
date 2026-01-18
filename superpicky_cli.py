#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky CLI - 命令行入口
完整功能版本 - 支持处理、重置、重新评星

Usage:
    python superpicky_cli.py process /path/to/photos [options]
    python superpicky_cli.py reset /path/to/photos
    python superpicky_cli.py restar /path/to/photos [options]
    python superpicky_cli.py info /path/to/photos

Examples:
    # 基本处理
    python superpicky_cli.py process ~/Photos/Birds
    
    # 自定义阈值
    python superpicky_cli.py process ~/Photos/Birds --sharpness 600 --nima 5.2
    
    # 不移动文件，只写EXIF
    python superpicky_cli.py process ~/Photos/Birds --no-organize
    
    # 重置目录
    python superpicky_cli.py reset ~/Photos/Birds
    
    # 重新评星
    python superpicky_cli.py restar ~/Photos/Birds --sharpness 700 --nima 5.5
"""

import argparse
import sys
import os
from pathlib import Path

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """打印 CLI 横幅"""
    print("\n" + "━" * 60)
    print("  🐦 SuperPicky CLI v3.9.3 - 慧眼选鸟 (命令行版)")
    print("━" * 60)


def cmd_burst(args):
    """连拍检测与分组"""
    from core.burst_detector import BurstDetector
    from exiftool_manager import ExifToolManager
    
    print_banner()
    print(f"\n📁 目标目录: {args.directory}")
    print(f"⚙️  最小连拍张数: {args.min_count}")
    print(f"⚙️  时间阈值: {args.threshold}ms")
    print(f"⚙️  pHash验证: {'启用' if args.phash else '禁用'}")
    print(f"⚙️  执行模式: {'实际处理' if args.execute else '仅预览'}")
    print()
    
    # 创建检测器
    detector = BurstDetector(use_phash=args.phash)
    detector.MIN_BURST_COUNT = args.min_count
    detector.TIME_THRESHOLD_MS = args.threshold
    
    # 运行检测
    print("🔍 正在检测连拍组...")
    results = detector.run_full_detection(args.directory)
    
    # 显示结果
    print(f"\n{'═' * 50}")
    print("  连拍检测结果")
    print(f"{'═' * 50}")
    print(f"\n📊 总览:")
    print(f"  总照片数: {results['total_photos']}")
    print(f"  有毫秒时间戳: {results['photos_with_subsec']}")
    print(f"  连拍组数: {results['groups_detected']}")
    
    for dir_name, data in results['groups_by_dir'].items():
        print(f"\n📂 {dir_name}:")
        print(f"  照片数: {data['photos']}")
        print(f"  连拍组: {data['groups']}")
        
        for g in data['group_details']:
            print(f"    组 #{g['id']}: {g['count']} 张, 最佳: {g['best']}")
    
    # 执行模式
    if args.execute and results['groups_detected'] > 0:
        print(f"\n🚀 开始处理连拍组...")
        
        exiftool_mgr = ExifToolManager()
        total_stats = {'groups_processed': 0, 'photos_moved': 0, 'best_marked': 0}
        
        rating_dirs = ['3星_优选', '2星_良好']
        for rating_dir in rating_dirs:
            subdir = os.path.join(args.directory, rating_dir)
            if not os.path.exists(subdir):
                continue
            
            # 重新获取该目录的 groups
            extensions = {'.nef', '.rw2', '.arw', '.cr2', '.cr3', '.orf', '.dng'}
            filepaths = []
            for entry in os.scandir(subdir):
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in extensions:
                        filepaths.append(entry.path)
            
            if not filepaths:
                continue
            
            photos = detector.read_timestamps(filepaths)
            csv_path = os.path.join(args.directory, '.superpicky', 'report.csv')
            photos = detector.enrich_from_csv(photos, csv_path)
            groups = detector.detect_groups(photos)
            groups = detector.select_best_in_groups(groups)
            
            # 处理
            stats = detector.process_burst_groups(groups, subdir, exiftool_mgr)
            total_stats['groups_processed'] += stats['groups_processed']
            total_stats['photos_moved'] += stats['photos_moved']
            total_stats['best_marked'] += stats['best_marked']
        
        print(f"\n✅ 处理完成!")
        print(f"  处理组数: {total_stats['groups_processed']}")
        print(f"  移动照片: {total_stats['photos_moved']}")
        print(f"  紫色标记: {total_stats['best_marked']}")
    elif not args.execute:
        print(f"\n💡 预览模式，未实际处理。添加 --execute 参数执行实际处理。")
    
    print()
    return 0


def cmd_process(args):
    """处理照片目录"""
    from cli_processor import CLIProcessor
    from core.config_manager import UISettings
    
    print_banner()
    print(f"\n📁 目标目录: {args.directory}")
    print(f"⚙️  锐度阈值: {args.sharpness}")
    print(f"  🎨 美学阈值: {args.nima_threshold} (默认: 5.0, TOPIQ)")
    print(f"⚙️  识别飞鸟: {'是' if args.flight else '否'}")
    print(f"⚙️  连拍检测: {'是' if args.burst else '否'}")
    print(f"⚙️  整理文件: {'是' if args.organize else '否'}")
    print(f"⚙️  清理临时: {'是' if args.cleanup else '否'}")
    print()
    
    # 创建处理器
    ui_settings = UISettings(
        ai_confidence=args.confidence,
        sharpness_threshold=args.sharpness,
        nima_threshold=args.nima_threshold,
        save_crop=False,
        normalization_mode='log_compression',
        detect_flight=args.flight,
        detect_exposure=True,   # V3.9.4: 默认开启曝光检测，与 GUI 一致
        detect_burst=args.burst,
        use_job_workers=True,
        use_cpu_workers=True,
        use_gpu_workers=True
        
    )
    
    processor = CLIProcessor(
        dir_path=args.directory,
        ui_settings=ui_settings,
        verbose=not args.quiet,
        detect_flight=args.flight
    )
    
    # 执行处理
    stats = processor.process(
        organize_files=args.organize,
        cleanup_temp=args.cleanup
    )
    
    # V4.0: 连拍检测（处理完成后执行）
    if args.burst and args.organize:
        from core.burst_detector import BurstDetector
        from exiftool_manager import get_exiftool_manager
        
        print("\n📷 正在执行连拍检测...")
        detector = BurstDetector(use_phash=True)
        
        rating_dirs = ['3星_优选', '2星_良好']
        total_groups = 0
        total_moved = 0
        
        exiftool_mgr = get_exiftool_manager()
        
        for rating_dir in rating_dirs:
            subdir = os.path.join(args.directory, rating_dir)
            if not os.path.exists(subdir):
                continue
            
            # 获取文件列表
            extensions = {'.nef', '.rw2', '.arw', '.cr2', '.cr3', '.orf', '.dng'}
            filepaths = []
            for entry in os.scandir(subdir):
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in extensions:
                        filepaths.append(entry.path)
            
            if not filepaths:
                continue
            
            photos = detector.read_timestamps(filepaths)
            csv_path = os.path.join(args.directory, '.superpicky', 'report.csv')
            photos = detector.enrich_from_csv(photos, csv_path)
            groups = detector.detect_groups(photos)
            groups = detector.select_best_in_groups(groups)
            
            burst_stats = detector.process_burst_groups(groups, subdir, exiftool_mgr)
            total_groups += burst_stats['groups_processed']
            total_moved += burst_stats['photos_moved']
        
        if total_groups > 0:
            print(f"  ✅ 连拍检测完成: {total_groups} 组, 移动 {total_moved} 张照片")
        else:
            print("  ℹ️  未检测到连拍组")
    
    print("\n✅ 处理完成!")
    return 0


def cmd_reset(args):
    """重置目录"""
    from find_bird_util import reset
    from exiftool_manager import get_exiftool_manager
    from i18n import get_i18n
    import shutil
    
    print_banner()
    print(f"\n🔄 重置目录: {args.directory}")
    
    if not args.yes:
        confirm = input("\n⚠️  这将重置所有评分和文件位置，确定继续? [y/N]: ")
        if confirm.lower() not in ['y', 'yes']:
            print("❌ 已取消")
            return 1
    
    # V4.0: 先处理 burst_XXX 子目录（将文件移回评分目录）
    print("\n📂 步骤0: 清理连拍子目录...")
    rating_dirs = ['3星_优选', '2星_良好', '1星_普通', '0星_放弃']
    burst_stats = {'dirs_removed': 0, 'files_restored': 0}
    
    for rating_dir in rating_dirs:
        rating_path = os.path.join(args.directory, rating_dir)
        if not os.path.exists(rating_path):
            continue
        
        # 查找 burst_XXX 子目录
        for entry in os.listdir(rating_path):
            if entry.startswith('burst_'):
                burst_path = os.path.join(rating_path, entry)
                if os.path.isdir(burst_path):
                    # 将文件移回评分目录
                    for filename in os.listdir(burst_path):
                        src = os.path.join(burst_path, filename)
                        dst = os.path.join(rating_path, filename)
                        if os.path.isfile(src):
                            try:
                                if os.path.exists(dst):
                                    os.remove(dst)
                                shutil.move(src, dst)
                                burst_stats['files_restored'] += 1
                            except Exception as e:
                                print(f"    ⚠️ 移动失败: {filename}: {e}")
                    
                    # 删除空的 burst 目录
                    try:
                        if not os.listdir(burst_path):
                            os.rmdir(burst_path)
                        else:
                            shutil.rmtree(burst_path)
                        burst_stats['dirs_removed'] += 1
                    except Exception as e:
                        print(f"    ⚠️ 删除目录失败: {entry}: {e}")
    
    if burst_stats['dirs_removed'] > 0:
        print(f"  ✅ 已清理 {burst_stats['dirs_removed']} 个连拍目录，恢复 {burst_stats['files_restored']} 个文件")
    else:
        print("  ℹ️  无连拍子目录需要清理")
    
    print("\n📂 步骤1: 恢复文件到主目录...")
    exiftool_mgr = get_exiftool_manager()
    restore_stats = exiftool_mgr.restore_files_from_manifest(args.directory)
    
    restored = restore_stats.get('restored', 0)
    if restored > 0:
        print(f"  ✅ 已恢复 {restored} 个文件")
    else:
        print("  ℹ️  无需恢复文件")
    
    print("\n📝 步骤2: 清理并重置 EXIF 元数据...")
    i18n = get_i18n('zh_CN')
    success = reset(args.directory, i18n=i18n)
    
    # V3.9: 删除空的评分目录
    print("\n🗑️  步骤3: 清理空目录...")
    deleted_dirs = 0
    for rating_dir in rating_dirs:
        rating_path = os.path.join(args.directory, rating_dir)
        if os.path.exists(rating_path) and os.path.isdir(rating_path):
            # 检查是否为空（或只包含隐藏文件/目录）
            contents = [f for f in os.listdir(rating_path) if not f.startswith('.')]
            if len(contents) == 0:
                try:
                    shutil.rmtree(rating_path)
                    print(f"  🗑️ 已删除空目录: {rating_dir}")
                    deleted_dirs += 1
                except Exception as e:
                    print(f"  ⚠️ 删除目录失败: {rating_dir}: {e}")
    
    if deleted_dirs > 0:
        print(f"  ✅ 已清理 {deleted_dirs} 个空评分目录")
    else:
        print("  ℹ️  无空目录需要清理")
    
    if success:
        print("\n✅ 目录重置完成!")
        return 0
    else:
        print("\n❌ 重置失败")
        return 1


def cmd_restar(args):
    """重新评星"""
    from post_adjustment_engine import PostAdjustmentEngine
    from exiftool_manager import get_exiftool_manager
    from advanced_config import get_advanced_config
    import shutil
    
    print_banner()
    print(f"\n🔄 重新评星: {args.directory}")
    print(f"⚙️  新锐度阈值: {args.sharpness}")
    print(f"⚙️  新美学阈值: {args.nima_threshold}")
    print(f"⚙️  连拍检测: {'是' if args.burst else '否'}")
    
    # V4.0: 先清理 burst 子目录（将文件移回评分目录）
    print("\n📂 步骤0: 清理连拍子目录...")
    rating_dirs = ['3星_优选', '2星_良好', '1星_普通', '0星_放弃']
    burst_stats = {'dirs_removed': 0, 'files_restored': 0}
    
    for rating_dir in rating_dirs:
        rating_path = os.path.join(args.directory, rating_dir)
        if not os.path.exists(rating_path):
            continue
        
        for entry in os.listdir(rating_path):
            if entry.startswith('burst_'):
                burst_path = os.path.join(rating_path, entry)
                if os.path.isdir(burst_path):
                    for filename in os.listdir(burst_path):
                        src = os.path.join(burst_path, filename)
                        dst = os.path.join(rating_path, filename)
                        if os.path.isfile(src):
                            try:
                                if os.path.exists(dst):
                                    os.remove(dst)
                                shutil.move(src, dst)
                                burst_stats['files_restored'] += 1
                            except Exception as e:
                                print(f"    ⚠️ 移动失败: {filename}: {e}")
                    
                    try:
                        if not os.listdir(burst_path):
                            os.rmdir(burst_path)
                        else:
                            shutil.rmtree(burst_path)
                        burst_stats['dirs_removed'] += 1
                    except Exception as e:
                        print(f"    ⚠️ 删除目录失败: {entry}: {e}")
    
    if burst_stats['dirs_removed'] > 0:
        print(f"  ✅ 已清理 {burst_stats['dirs_removed']} 个连拍目录，恢复 {burst_stats['files_restored']} 个文件")
    else:
        print("  ℹ️  无连拍子目录需要清理")
    
    # 检查 report.csv 是否存在（可能在根目录或 .superpicky 子目录）
    report_path = os.path.join(args.directory, 'report.csv')
    report_path_alt = os.path.join(args.directory, '.superpicky', 'report.csv')
    if not os.path.exists(report_path) and not os.path.exists(report_path_alt):
        print("\n❌ 未找到 report.csv，请先运行 process 命令")
        return 1
    
    # 初始化引擎
    engine = PostAdjustmentEngine(args.directory)
    
    # 加载报告
    success, msg = engine.load_report()
    if not success:
        print(f"\n❌ 加载数据失败: {msg}")
        return 1
    
    print(f"\n📊 {msg}")
    
    # 获取高级配置的 0 星阈值
    adv_config = get_advanced_config()
    min_confidence = getattr(adv_config, 'min_confidence', 0.5)
    min_sharpness = getattr(adv_config, 'min_sharpness', 250)
    min_nima = getattr(adv_config, 'min_nima', 4.0)
    
    # 重新计算评分
    new_photos = engine.recalculate_ratings(
        photos=engine.photos_data,
        min_confidence=min_confidence,
        min_sharpness=min_sharpness,
        min_nima=min_nima,
        sharpness_threshold=args.sharpness,
        nima_threshold=args.nima_threshold
    )
    
    # 统计变化
    changed_photos = []
    old_stats = {'star_3': 0, 'star_2': 0, 'star_1': 0, 'star_0': 0}
    for photo in new_photos:
        old_rating = int(photo.get('rating', 0))
        new_rating = photo.get('新星级', 0)
        
        # 统计原始评分
        if old_rating == 3:
            old_stats['star_3'] += 1
        elif old_rating == 2:
            old_stats['star_2'] += 1
        elif old_rating == 1:
            old_stats['star_1'] += 1
        else:
            old_stats['star_0'] += 1
        
        if old_rating != new_rating:
            photo['filename'] = photo.get('filename', '')
            changed_photos.append(photo)
    
    # 统计新评分
    new_stats = engine.get_statistics(new_photos)
    
    # 使用共享格式化模块输出对比
    from core.stats_formatter import format_restar_comparison, print_summary
    lines = format_restar_comparison(old_stats, new_stats, len(changed_photos))
    print_summary(lines)
    
    if len(changed_photos) == 0:
        print("\n✅ 无需更新任何照片")
        # 即使评分无变化，如果开启了连拍检测，仍然运行
        if args.burst and args.organize:
            _run_burst_detection_restar(args.directory)
        return 0
    
    if not args.yes:
        confirm = input("\n确定应用新评分? [y/N]: ")
        if confirm.lower() not in ['y', 'yes']:
            print("❌ 已取消")
            return 1
    
    # 准备 EXIF 批量更新数据
    exiftool_mgr = get_exiftool_manager()
    batch_data = []
    
    for photo in changed_photos:
        filename = photo.get('filename', '')
        file_path = engine.find_image_file(filename)
        if file_path:
            rating = photo.get('新星级', 0)
            batch_data.append({
                'file': file_path,
                'rating': rating,
                'pick': 0
            })
    
    # 写入 EXIF
    print("\n📝 写入 EXIF 元数据...")
    exif_stats = exiftool_mgr.batch_set_metadata(batch_data)
    print(f"  ✅ 成功: {exif_stats.get('success', 0)}, 失败: {exif_stats.get('failed', 0)}")
    
    # 更新 CSV
    print("\n📊 更新 report.csv...")
    picked_files = set()  # CLI 模式暂不支持精选计算
    engine.update_report_csv(new_photos, picked_files)
    
    # 文件重分配
    if args.organize:
        print("\n📂 重新分配文件目录...")
        RATING_FOLDER_NAMES = {
            3: "3星_优选",
            2: "2星_良好",
            1: "1星_普通",
            0: "0星_放弃",
            -1: "0星_放弃"
        }
        
        moved_count = 0
        for photo in changed_photos:
            filename = photo.get('filename', '')
            file_path = engine.find_image_file(filename)
            if not file_path:
                continue
            
            new_rating = photo.get('新星级', 0)
            target_folder = RATING_FOLDER_NAMES.get(new_rating, "0星_放弃")
            target_dir = os.path.join(args.directory, target_folder)
            target_path = os.path.join(target_dir, os.path.basename(file_path))
            
            if os.path.dirname(file_path) == target_dir:
                continue
            
            try:
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                if not os.path.exists(target_path):
                    shutil.move(file_path, target_path)
                    moved_count += 1
            except Exception:
                pass
        
        if moved_count > 0:
            print(f"  ✅ 已移动 {moved_count} 个文件")
        
        # V4.0: 重新运行连拍检测
        if args.burst:
            _run_burst_detection_restar(args.directory)
    
    print("\n✅ 重新评星完成!")
    return 0


def _run_burst_detection_restar(directory: str):
    """Restar 后运行连拍检测"""
    from core.burst_detector import BurstDetector
    from exiftool_manager import get_exiftool_manager
    
    print("\n📷 正在执行连拍检测...")
    detector = BurstDetector(use_phash=True)
    
    rating_dirs = ['3星_优选', '2星_良好']
    total_groups = 0
    total_moved = 0
    
    exiftool_mgr = get_exiftool_manager()
    
    for rating_dir in rating_dirs:
        subdir = os.path.join(directory, rating_dir)
        if not os.path.exists(subdir):
            continue
        
        extensions = {'.nef', '.rw2', '.arw', '.cr2', '.cr3', '.orf', '.dng'}
        filepaths = []
        for entry in os.scandir(subdir):
            if entry.is_file():
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in extensions:
                    filepaths.append(entry.path)
        
        if not filepaths:
            continue
        
        photos = detector.read_timestamps(filepaths)
        csv_path = os.path.join(directory, '.superpicky', 'report.csv')
        photos = detector.enrich_from_csv(photos, csv_path)
        groups = detector.detect_groups(photos)
        groups = detector.select_best_in_groups(groups)
        
        burst_stats = detector.process_burst_groups(groups, subdir, exiftool_mgr)
        total_groups += burst_stats['groups_processed']
        total_moved += burst_stats['photos_moved']
    
    if total_groups > 0:
        print(f"  ✅ 连拍检测完成: {total_groups} 组, 移动 {total_moved} 张照片")
    else:
        print("  ℹ️  未检测到连拍组")


def cmd_info(args):
    """显示目录信息"""
    import pandas as pd
    
    print_banner()
    print(f"\n📁 目录: {args.directory}")
    
    # 检查各种文件
    report_path = os.path.join(args.directory, 'report.csv')
    manifest_path = os.path.join(args.directory, '.superpicky_manifest.json')
    
    print("\n📋 文件状态:")
    
    if os.path.exists(report_path):
        print("  ✅ report.csv 存在")
        try:
            df = pd.read_csv(report_path)
            total = len(df)
            print(f"     共 {total} 条记录")
            
            if 'rating' in df.columns:
                rating_counts = df['rating'].value_counts().sort_index()
                print("\n📊 评分分布:")
                for rating, count in rating_counts.items():
                    stars = "⭐" * max(0, int(rating)) if rating >= 0 else "❌"
                    print(f"     {stars} {rating}星: {count} 张")
            
            if 'is_flying' in df.columns:
                flying = df[df['is_flying'] == 'yes'].shape[0]
                if flying > 0:
                    print(f"\n🦅 飞鸟照片: {flying} 张")
                    
        except Exception as e:
            print(f"     读取失败: {e}")
    else:
        print("  ❌ report.csv 不存在")
    
    if os.path.exists(manifest_path):
        print("  ✅ manifest 文件存在 (可重置)")
    else:
        print("  ℹ️  manifest 文件不存在")
    
    # 检查分类文件夹
    folders = ['3星_优选', '2星_良好', '1星_普通', '0星_放弃']
    existing_folders = []
    for folder in folders:
        folder_path = os.path.join(args.directory, folder)
        if os.path.exists(folder_path):
            count = len([f for f in os.listdir(folder_path) 
                        if f.lower().endswith(('.nef', '.cr2', '.arw', '.jpg', '.jpeg'))])
            existing_folders.append((folder, count))
    
    if existing_folders:
        print("\n📂 分类文件夹:")
        for folder, count in existing_folders:
            print(f"     {folder}/: {count} 张")
    
    print()
    return 0


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        prog='superpicky_cli',
        description='SuperPicky CLI - 慧眼选鸟命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s process ~/Photos/Birds              # 处理照片
  %(prog)s process ~/Photos/Birds -s 600       # 自定义锐度阈值
  %(prog)s reset ~/Photos/Birds -y             # 重置目录(无确认)
  %(prog)s restar ~/Photos/Birds -s 700 -n 5.5 # 重新评星
  %(prog)s info ~/Photos/Birds                 # 查看目录信息
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # ===== process 命令 =====
    p_process = subparsers.add_parser('process', help='处理照片目录')
    p_process.add_argument('directory', help='照片目录路径')
    p_process.add_argument('-s', '--sharpness', type=int, default=400,
                          help='锐度阈值 (默认: 400, 范围: 200-600)')
    p_process.add_argument('-n', '--nima-threshold', type=float, default=5.0,
                          help='美学阈值 (TOPIQ, 默认: 5.0, 范围: 4.0-7.0)')
    p_process.add_argument('-c', '--confidence', type=int, default=50,
                          help='AI置信度阈值 (默认: 50)')
    # 飞鸟检测（使用 store_true/store_false 组合，通过 set_defaults 设置默认值为 True）
    p_process.add_argument('--flight', action='store_true', dest='flight',
                          help='识别飞鸟 (默认: 开启)')
    p_process.add_argument('--no-flight', action='store_false', dest='flight',
                          help='禁用飞鸟识别')
    # 连拍检测（同样方式）
    p_process.add_argument('--burst', action='store_true', dest='burst',
                          help='连拍检测 (默认: 开启)')
    p_process.add_argument('--no-burst', action='store_false', dest='burst',
                          help='禁用连拍检测')
    p_process.add_argument('--no-organize', action='store_false', dest='organize',
                          help='不移动文件到分类文件夹')
    p_process.add_argument('--no-cleanup', action='store_false', dest='cleanup',
                          help='不清理临时JPG文件')
    p_process.add_argument('-q', '--quiet', action='store_true',
                          help='静默模式')
    # V3.9: 使用 set_defaults 确保 flight, burst 默认为 True
    p_process.set_defaults(organize=True, cleanup=True, burst=True, flight=True)
    
    # ===== reset 命令 =====
    p_reset = subparsers.add_parser('reset', help='重置目录')
    p_reset.add_argument('directory', help='照片目录路径')
    p_reset.add_argument('-y', '--yes', action='store_true',
                        help='跳过确认提示')
    
    # ===== restar 命令 =====
    p_restar = subparsers.add_parser('restar', help='重新评星')
    p_restar.add_argument('directory', help='照片目录路径')
    p_restar.add_argument('-s', '--sharpness', type=int, default=400,
                         help='新锐度阈值 (默认: 400, 范围: 200-600)')
    p_restar.add_argument('-n', '--nima-threshold', type=float, default=5.0,
                         help='TOPIQ 美学评分阈值 (默认: 5.0, 范围: 4.0-7.0)')
    p_restar.add_argument('-c', '--confidence', type=int, default=50,
                         help='AI置信度阈值 (默认: 50)')
    p_restar.add_argument('--burst', action='store_true', default=True,
                         help='连拍检测 (默认: 开启)')
    p_restar.add_argument('--no-burst', action='store_false', dest='burst',
                         help='禁用连拍检测')
    p_restar.add_argument('--no-organize', action='store_false', dest='organize',
                         help='不重新分配文件目录')
    p_restar.add_argument('-y', '--yes', action='store_true',
                         help='跳过确认提示')
    p_restar.set_defaults(organize=True, burst=True)
    
    # ===== info 命令 =====
    p_info = subparsers.add_parser('info', help='查看目录信息')
    p_info.add_argument('directory', help='照片目录路径')
    
    # ===== burst 命令 =====
    p_burst = subparsers.add_parser('burst', help='连拍检测与分组')
    p_burst.add_argument('directory', help='照片目录路径')
    p_burst.add_argument('-m', '--min-count', type=int, default=4,
                         help='最小连拍张数 (默认: 4)')
    p_burst.add_argument('-t', '--threshold', type=int, default=250,
                         help='时间阈值(ms) (默认: 250)')
    p_burst.add_argument('--no-phash', action='store_false', dest='phash',
                         help='禁用 pHash 验证（默认启用）')
    p_burst.add_argument('--execute', action='store_true',
                         help='实际执行处理（默认仅预览）')
    p_burst.set_defaults(phash=True)
    
    # 解析参数
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 验证目录
    if not os.path.isdir(args.directory):
        print(f"❌ 目录不存在: {args.directory}")
        return 1
    
    # 转换为绝对路径
    args.directory = os.path.abspath(args.directory)
    
    # 执行命令
    if args.command == 'process':
        return cmd_process(args)
    elif args.command == 'reset':
        return cmd_reset(args)
    elif args.command == 'restar':
        return cmd_restar(args)
    elif args.command == 'info':
        return cmd_info(args)
    elif args.command == 'burst':
        return cmd_burst(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
