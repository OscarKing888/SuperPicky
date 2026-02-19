#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky CLI - 命令行入口
完整功能版本 - 支持处理、重置、重新评星、鸟类识别

Usage:
    python superpicky_cli.py process /path/to/photos [options]
    python superpicky_cli.py reset /path/to/photos
    python superpicky_cli.py restar /path/to/photos [options]
    python superpicky_cli.py info /path/to/photos
    python superpicky_cli.py identify /path/to/bird.jpg [options]

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

    # 鸟类识别
    python superpicky_cli.py identify ~/Photos/bird.jpg
    python superpicky_cli.py identify ~/Photos/bird.NEF --top 10
    python superpicky_cli.py identify ~/Photos/bird.jpg --write-exif
"""

import argparse
import sys
import os
from pathlib import Path
from tools.i18n import t

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """打印 CLI 横幅"""
    print("\n" + "━" * 60)
    print(t("cli.banner", version="4.0.6"))
    print("━" * 60)


def cmd_burst(args):
    """连拍检测与分组"""
    from core.burst_detector import BurstDetector
    from exiftool_manager import ExifToolManager
    
    print_banner()
    print(t("cli.target_dir", directory=args.directory))
    print(t("cli.min_burst", count=args.min_count))
    print(t("cli.time_threshold", ms=args.threshold))
    print(t("cli.phash", status=t("cli.enabled") if args.phash else t("cli.disabled")))
    print(t("cli.execute_mode", mode=t("cli.mode_real") if args.execute else t("cli.mode_preview")))
    print()
    
    # 创建检测器
    detector = BurstDetector(use_phash=args.phash)
    detector.MIN_BURST_COUNT = args.min_count
    detector.TIME_THRESHOLD_MS = args.threshold
    
    # 运行检测
    print(t("cli.detecting_burst"))
    results = detector.run_full_detection(args.directory)
    
    # 显示结果
    print(f"\n{'═' * 50}")
    print(t("cli.burst_result_title"))
    print(f"{'═' * 50}")
    print(t("cli.total_overview"))
    print(t("cli.total_photos", count=results['total_photos']))
    print(t("cli.photos_subsec", count=results['photos_with_subsec']))
    print(t("cli.groups_detected", count=results['groups_detected']))
    
    for dir_name, data in results['groups_by_dir'].items():
        print(f"\n📂 {dir_name}:")
        print(f"  照片数: {data['photos']}")
        print(f"  连拍组: {data['groups']}")
        
        for g in data['group_details']:
            print(f"    组 #{g['id']}: {g['count']} 张, 最佳: {g['best']}")
    
    # 执行模式
    if args.execute and results['groups_detected'] > 0:
        print(t("cli.processing_burst"))
        
        exiftool_mgr = ExifToolManager()
        total_stats = {'groups_processed': 0, 'photos_moved': 0, 'best_marked': 0}
        
        rating_dirs = ['3star_excellent', '2star_good', '3星_优选', '2星_良好']  # Support both languages
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
            photos = detector.enrich_from_db(photos, args.directory)
            groups = detector.detect_groups(photos)
            groups = detector.select_best_in_groups(groups)
            
            # 处理
            stats = detector.process_burst_groups(groups, subdir, exiftool_mgr)
            total_stats['groups_processed'] += stats['groups_processed']
            total_stats['photos_moved'] += stats['photos_moved']
            total_stats['best_marked'] += stats['best_marked']
        
        print(t("cli.processing_complete"))
        print(t("cli.processed_groups", count=total_stats['groups_processed']))
        print(t("cli.moved_photos", count=total_stats['photos_moved']))
        print(t("cli.marked_purple", count=total_stats['best_marked']))
    elif not args.execute:
        print(t("cli.preview_hint"))
    
    print()
    return 0


def cmd_process(args):
    """处理照片目录"""
    from tools.cli_processor import CLIProcessor
    from core.photo_processor import ProcessingSettings
    from advanced_config import get_advanced_config
    
    print_banner()
    print(t("cli.target_dir", directory=args.directory))
    print(t("cli.sharpness", value=args.sharpness))
    print(t("cli.aesthetics", value=args.nima_threshold))
    print(t("cli.detect_flight", value=t("cli.enabled") if args.flight else t("cli.disabled")))
    print(t("cli.xmp", value=t("cli.enabled") if args.xmp else t("cli.disabled")))
    print(t("cli.detect_burst", value=t("cli.enabled") if args.burst else t("cli.disabled")))
    print(t("cli.organize_files", value=t("cli.enabled") if args.organize else t("cli.disabled")))
    print(f"⚙️  清理临时: {'是' if args.cleanup else '否'}")
    
    # V4.0: 显示自动识鸟设置
    auto_identify = getattr(args, 'auto_identify', False)
    if auto_identify:
        print(f"⚙️  自动识鸟: 是 (2★+ 照片)")
        if getattr(args, 'birdid_country', None):
            print(f"  └─ 国家: {args.birdid_country}")
        if getattr(args, 'birdid_region', None):
            print(f"  └─ 区域: {args.birdid_region}")
        print(f"  └─ 置信度阈值: {getattr(args, 'birdid_threshold', 70.0)}%")
    print()
    
    # 更新 ARW 写入策略
    adv_config = get_advanced_config()
    adv_config.config["arw_write_mode"] = "sidecar" if args.xmp else "embedded"
    
    # V4.1: 更新临时文件配置
    # 如果指定了 --keep-temp-files 或 --no-keep-temp-files，优先使用
    # 如果没指定，检查 --no-cleanup (args.cleanup=False) -> keep_temp=True
    if hasattr(args, 'keep_temp'):
        adv_config.config["keep_temp_files"] = args.keep_temp
    elif not args.cleanup:
        # 兼容旧参数：--no-cleanup 意味着保留临时文件
        adv_config.config["keep_temp_files"] = True
        
    if hasattr(args, 'cleanup_days'):
        adv_config.config["auto_cleanup_days"] = args.cleanup_days
        
    adv_config.save()

    # V4.0: 构建 ProcessingSettings（与 GUI 完全一致）
    settings = ProcessingSettings(
        ai_confidence=args.confidence,
        sharpness_threshold=args.sharpness,
        nima_threshold=args.nima_threshold,
        normalization_mode='log_compression',
        detect_flight=args.flight,
        detect_exposure=True,
        detect_burst=args.burst,
        # V4.0: BirdID 自动识别设置
        auto_identify=auto_identify,
        # V4.1: Crop
        save_crop=args.save_crop,
        birdid_use_ebird=True,
        birdid_country_code=getattr(args, 'birdid_country', None),
        birdid_region_code=getattr(args, 'birdid_region', None),
        birdid_confidence_threshold=getattr(args, 'birdid_threshold', 70.0)
    )
    
    # 创建处理器（直接传入 ProcessingSettings）
    processor = CLIProcessor(
        dir_path=args.directory,
        verbose=not args.quiet,
        settings=settings
    )
    
    # 执行处理（PhotoProcessor 内部会处理自动识鸟）
    # 执行处理（PhotoProcessor 内部会处理自动识鸟）
    # V4.1: cleanup_temp 参数现在由 AdvancedConfig.keep_temp_files 控制
    # 但为了兼容性，如果显式传递了参数，我们还是传递下去，不过 PhotoProcessor 内部会使用统一逻辑
    stats = processor.process(
        organize_files=args.organize,
        cleanup_temp=not adv_config.keep_temp_files  # 如果保留，则不清理
    )
    
    # V4.0.5: 连拍检测已移至 PhotoProcessor 内部
    # - 早期检测: _detect_bursts_early() 在文件扫描后执行
    # - 跨目录合并: _consolidate_burst_groups() 在文件整理后执行
    # 这样可以实现跨星级目录的连拍合并，将所有连拍照片移至最高星级目录
    
    print("\n✅ 处理完成!")
    return 0


def cmd_reset(args):
    """重置目录"""
    from tools.find_bird_util import reset
    from tools.exiftool_manager import get_exiftool_manager
    from tools.i18n import get_i18n
    import shutil
    
    print_banner()
    print(f"\n🔄 重置目录: {args.directory}")
    
    if not args.yes:
        confirm = input("\n⚠️  这将重置所有评分和文件位置，确定继续? [y/N]: ")
        if confirm.lower() not in ['y', 'yes']:
            print("❌ 已取消")
            return 1
    
    # V4.0.5: 先处理所有子目录（burst_XXX、鸟种 Other_Birds 等）
    # 将文件移回评分目录，然后由步骤1的 manifest 恢复到根目录
    print("\n📂 步骤0: 清理评分目录中的子目录...")
    rating_dirs = ['3star_excellent', '2star_good', '1star_average', '0star_reject',
                   '3星_优选', '2星_良好', '1星_普通', '0星_放弃']  # Support both languages
    subdir_stats = {'dirs_removed': 0, 'files_restored': 0}
    
    for rating_dir in rating_dirs:
        rating_path = os.path.join(args.directory, rating_dir)
        if not os.path.exists(rating_path):
            continue
        
        # 查找所有子目录（burst_XXX、鸟种目录等）
        for entry in os.listdir(rating_path):
            entry_path = os.path.join(rating_path, entry)
            if os.path.isdir(entry_path):
                print(f"  📁 打平子目录: {rating_dir}/{entry}")
                # 递归将所有文件移回评分目录
                for root, dirs, files in os.walk(entry_path):
                    for filename in files:
                        src = os.path.join(root, filename)
                        dst = os.path.join(rating_path, filename)
                        if os.path.isfile(src):
                            try:
                                if os.path.exists(dst):
                                    os.remove(dst)
                                shutil.move(src, dst)
                                subdir_stats['files_restored'] += 1
                            except Exception as e:
                                print(f"    ⚠️ 移动失败: {filename}: {e}")
                    
                # 删除子目录
                try:
                    if os.path.exists(entry_path):
                        shutil.rmtree(entry_path)
                    subdir_stats['dirs_removed'] += 1
                except Exception as e:
                    print(f"    ⚠️ 删除目录失败: {entry}: {e}")
    
    if subdir_stats['dirs_removed'] > 0:
        print(f"  ✅ 已清理 {subdir_stats['dirs_removed']} 个子目录，恢复 {subdir_stats['files_restored']} 个文件")
    else:
        print("  ℹ️  无子目录需要清理")
    
    print("\n📂 步骤1: 恢复文件到主目录...")
    exiftool_mgr = get_exiftool_manager()
    restore_stats = exiftool_mgr.restore_files_from_manifest(args.directory)
    
    restored = restore_stats.get('restored', 0)
    if restored > 0:
        print(f"  ✅ 已通过 Manifest 恢复 {restored} 个文件")
    
    # V4.0.5: Manifest 可能不包含所有文件（来自上次运行的残留文件）
    # 扫描评分目录，将所有文件强制移回根目录
    fallback_restored = 0
    for rating_dir in rating_dirs:
        rating_path = os.path.join(args.directory, rating_dir)
        if not os.path.exists(rating_path):
            continue
        
        for filename in os.listdir(rating_path):
            src = os.path.join(rating_path, filename)
            dst = os.path.join(args.directory, filename)
            if os.path.isfile(src):
                try:
                    if os.path.exists(dst):
                        os.remove(dst)
                    shutil.move(src, dst)
                    fallback_restored += 1
                except Exception as e:
                    print(f"    ⚠️ 回迁失败: {filename}: {e}")
    
    if fallback_restored > 0:
        print(f"  ✅ 额外恢复了 {fallback_restored} 个残留文件到根目录")
    
    total_restored = restored + fallback_restored
    if total_restored == 0:
        print("  ℹ️  无需恢复文件")
    
    print("\n📝 步骤2: 清理并重置 EXIF 元数据...")
    i18n = get_i18n('zh_CN')
    success = reset(args.directory, i18n=i18n)
    
    # V4.0.5: 删除评分目录（所有文件已移走）
    print("\n🗑️  步骤3: 清理目录...")
    deleted_dirs = 0
    for rating_dir in rating_dirs:
        rating_path = os.path.join(args.directory, rating_dir)
        if os.path.exists(rating_path) and os.path.isdir(rating_path):
            try:
                shutil.rmtree(rating_path)
                print(f"  🗑️ 已删除: {rating_dir}")
                deleted_dirs += 1
            except Exception as e:
                print(f"  ⚠️ 删除目录失败: {rating_dir}: {e}")
    
    # V4.0.5: 清理 .superpicky 隐藏目录和 manifest 文件
    superpicky_dir = os.path.join(args.directory, ".superpicky")
    if os.path.exists(superpicky_dir):
        try:
            shutil.rmtree(superpicky_dir)
            print("  🗑️ 已删除: .superpicky/")
            deleted_dirs += 1
        except Exception:
            try:
                import subprocess
                subprocess.run(['rm', '-rf', superpicky_dir], check=True)
                print("  🗑️ 已删除: .superpicky/ (force)")
                deleted_dirs += 1
            except Exception as e2:
                print(f"  ⚠️ .superpicky 删除失败: {e2}")
    
    manifest_file = os.path.join(args.directory, ".superpicky_manifest.json")
    if os.path.exists(manifest_file):
        try:
            os.remove(manifest_file)
            print("  🗑️ 已删除: .superpicky_manifest.json")
        except Exception as e:
            print(f"  ⚠️ manifest 删除失败: {e}")
    
    # 清理 macOS ._burst_XXX 残留文件
    for filename in os.listdir(args.directory):
        if filename.startswith('._burst_') or filename.startswith('._其他') or filename.startswith('._栗'):
            try:
                os.remove(os.path.join(args.directory, filename))
            except Exception:
                pass
    
    if deleted_dirs > 0:
        print(f"  ✅ 已清理 {deleted_dirs} 个目录")
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
    from tools.exiftool_manager import get_exiftool_manager
    from advanced_config import get_advanced_config
    import shutil
    
    print_banner()
    print(f"\n🔄 重新评星: {args.directory}")
    print(f"⚙️  新锐度阈值: {args.sharpness}")
    print(f"⚙️  新美学阈值: {args.nima_threshold}")
    print(f"⚙️  连拍检测: {'是' if args.burst else '否'}")
    print(t("cli.xmp", value=t("cli.enabled") if args.xmp else t("cli.disabled")))

    # 更新 ARW 写入策略
    adv_config = get_advanced_config()
    adv_config.config["arw_write_mode"] = "sidecar" if args.xmp else "embedded"
    adv_config.save()
    
    # V4.0: 先清理 burst 子目录（将文件移回评分目录）
    print("\n📂 步骤0: 清理连拍子目录...")
    rating_dirs = ['3star_excellent', '2star_good', '1star_average', '0star_reject',
                   '3星_优选', '2星_良好', '1星_普通', '0星_放弃']  # Support both languages
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
    
    # 检查 report.db 是否存在
    db_path = os.path.join(args.directory, '.superpicky', 'report.db')
    if not os.path.exists(db_path):
        print("\n❌ 未找到 report.db，请先运行 process 命令")
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
    
    # 更新数据库
    print("\n📊 更新 report.db...")
    picked_files = set()  # CLI 模式暂不支持精选计算
    engine.update_report_csv(new_photos, picked_files)
    
    # 文件重分配
    if args.organize:
        from constants import get_rating_folder_name
        
        moved_count = 0
        for photo in changed_photos:
            filename = photo.get('filename', '')
            file_path = engine.find_image_file(filename)
            if not file_path:
                continue
            
            new_rating = photo.get('新星级', 0)
            target_folder = get_rating_folder_name(new_rating)
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
    from tools.exiftool_manager import get_exiftool_manager
    
    print("\n📷 正在执行连拍检测...")
    detector = BurstDetector(use_phash=True)
    
    rating_dirs = ['3star_excellent', '2star_good', '3星_优选', '2星_良好']  # Support both languages
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
        photos = detector.enrich_from_db(photos, directory)
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
    from tools.report_db import ReportDB
    
    print_banner()
    print(f"\n📁 目录: {args.directory}")
    
    # 检查各种文件
    db_path = os.path.join(args.directory, '.superpicky', 'report.db')
    manifest_path = os.path.join(args.directory, '.superpicky_manifest.json')
    
    print("\n📋 文件状态:")
    
    if os.path.exists(db_path):
        print("  ✅ report.db 存在")
        try:
            db = ReportDB(args.directory)
            stats = db.get_statistics()
            total = stats['total']
            print(f"     共 {total} 条记录")
            
            print("\n📊 评分分布:")
            for rating, count in sorted(stats['by_rating'].items()):
                stars = "⭐" * max(0, int(rating)) if rating >= 0 else "❌"
                print(f"     {stars} {rating}星: {count} 张")
            
            if stats['flying'] > 0:
                print(f"\n🦅 飞鸟照片: {stats['flying']} 张")
            
            db.close()
        except Exception as e:
            print(f"     读取失败: {e}")
    else:
        print("  ❌ report.db 不存在")
    
    if os.path.exists(manifest_path):
        print("  ✅ manifest 文件存在 (可重置)")
    else:
        print("  ℹ️  manifest 文件不存在")
    
    # 检查分类文件夹
    folders = ['3star_excellent', '2star_good', '1star_average', '0star_reject',
               '3星_优选', '2星_良好', '1星_普通', '0星_放弃']  # Support both languages
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


def cmd_identify(args):
    """识别鸟类"""
    from birdid.bird_identifier import identify_bird, YOLO_AVAILABLE, RAW_SUPPORT

    print_banner()
    print(f"\n🐦 鸟类识别")
    print(f"📸 图片: {args.image}")
    print(f"⚙️  YOLO裁剪: {'是' if args.yolo else '否'}")
    print(f"⚙️  GPS过滤: {'是' if args.gps else '否'}")
    print(f"⚙️  返回数量: {args.top}")
    print()

    if not YOLO_AVAILABLE:
        print("⚠️  YOLO 模块未安装，将使用完整图像识别")

    # 执行识别
    print("🔍 正在识别...")
    result = identify_bird(
        args.image,
        use_yolo=args.yolo,
        use_gps=args.gps,
        top_k=args.top
    )

    if not result['success']:
        print(f"\n❌ 识别失败: {result.get('error', '未知错误')}")
        return 1

    # 显示结果
    print(f"\n{'═' * 50}")
    print("  识别结果")
    print(f"{'═' * 50}")

    if result.get('yolo_info'):
        print(f"\n📍 YOLO检测: {result['yolo_info']}")

    if result.get('gps_info'):
        gps = result['gps_info']
        print(f"🌍 GPS位置: {gps['info']}")

    results = result.get('results', [])
    if not results:
        print("\n⚠️  未能识别出鸟类")
        return 0

    print(f"\n🐦 Top-{len(results)} 识别结果:")
    for i, r in enumerate(results, 1):
        cn_name = r.get('cn_name', '未知')
        en_name = r.get('en_name', 'Unknown')
        confidence = r.get('confidence', 0)
        ebird_match = "✓" if r.get('ebird_match') else ""

        print(f"  {i}. {cn_name} ({en_name})")
        print(f"     置信度: {confidence:.1f}% {ebird_match}")

    # 写入 EXIF（如果启用）
    if args.write_exif and results:
        from exiftool_manager import get_exiftool_manager

        best = results[0]
        bird_name = f"{best['cn_name']} ({best['en_name']})"

        print(f"\n📝 写入 EXIF Title...")
        exiftool_mgr = get_exiftool_manager()
        success = exiftool_mgr.set_metadata(args.image, {'Title': bird_name})

        if success:
            print(f"  ✅ 已写入: {bird_name}")
        else:
            print(f"  ❌ 写入失败")

    print()
    return 0


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        prog='superpicky_cli',
        description=t("cli.sp_description"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s process ~/Photos/Birds              # 处理照片
  %(prog)s process ~/Photos/Birds -s 600       # 自定义锐度阈值
  %(prog)s reset ~/Photos/Birds -y             # 重置目录(无确认)
  %(prog)s restar ~/Photos/Birds -s 700 -n 5.5 # 重新评星
  %(prog)s info ~/Photos/Birds                 # 查看目录信息
  %(prog)s identify ~/Photos/bird.jpg          # 识别鸟类
  %(prog)s identify bird.NEF --write-exif      # 识别并写入EXIF
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # ===== process 命令 =====
    p_process = subparsers.add_parser('process', help=t("cli.cmd_process"))
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
    # XMP 侧车写入
    p_process.add_argument('--xmp', action='store_true', dest='xmp',
                          help='写入XMP侧车(不改RAW)')
    p_process.add_argument('--no-xmp', action='store_false', dest='xmp',
                          help='直接写入RAW(默认)')
    p_process.add_argument('--no-organize', action='store_false', dest='organize',
                          help='不移动文件到分类文件夹')
    p_process.add_argument('--no-cleanup', action='store_false', dest='cleanup',
                          help='不清理临时JPG文件')
    p_process.add_argument('-q', '--quiet', action='store_true',
                          help='静默模式')
    # V4.0: 自动识鸟选项
    p_process.add_argument('--auto-identify', '-i', action='store_true',
                          help='自动识别 2★+ 照片的鸟种并按鸟种分目录')
    p_process.add_argument('--birdid-country', type=str, default=None,
                          help='BirdID 国家代码 (如 AU, CN, US)')
    p_process.add_argument('--birdid-region', type=str, default=None,
                          help='BirdID 区域代码 (如 AU-SA, CN-31)')
    p_process.add_argument('--birdid-threshold', type=float, default=70.0,
                          help='BirdID 置信度阈值 (默认: 70%%)')
    # V4.1: 临时文件管理
    p_process.add_argument('--keep-temp-files', action='store_true', dest='keep_temp',
                          help='保留临时预览图片（默认: 开启）')
    p_process.add_argument('--no-keep-temp-files', action='store_false', dest='keep_temp',
                          help='不保留临时预览图片')
    p_process.add_argument('--cleanup-days', type=int, default=30,
                          help='自动清理周期（天），0=永久 (默认: 30)')
    p_process.add_argument('--save-crop', action='store_true',
                          help='保留 bird/debug 裁剪图片 (保存到 .superpicky/cache/debug)')
                          
    # V3.9: 使用 set_defaults 确保 flight, burst 默认为 True
    # V4.1: keep_temp 默认为 True
    p_process.set_defaults(organize=True, cleanup=True, burst=True, flight=True, auto_identify=False, xmp=False, keep_temp=True)
    
    # ===== reset 命令 =====
    p_reset = subparsers.add_parser('reset', help=t("cli.cmd_reset"))
    p_reset.add_argument('directory', help='照片目录路径')
    p_reset.add_argument('-y', '--yes', action='store_true',
                        help='跳过确认提示')
    
    # ===== restar 命令 =====
    p_restar = subparsers.add_parser('restar', help=t("cli.cmd_restar"))
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
    # XMP 侧车写入
    p_restar.add_argument('--xmp', action='store_true', dest='xmp',
                         help='写入XMP侧车(不改RAW)')
    p_restar.add_argument('--no-xmp', action='store_false', dest='xmp',
                         help='直接写入RAW(默认)')
    p_restar.add_argument('--no-organize', action='store_false', dest='organize',
                         help='不重新分配文件目录')
    p_restar.add_argument('-y', '--yes', action='store_true',
                         help='跳过确认提示')
    p_restar.set_defaults(organize=True, burst=True, xmp=False)
    
    # ===== info 命令 =====
    p_info = subparsers.add_parser('info', help=t("cli.cmd_info"))
    p_info.add_argument('directory', help='照片目录路径')
    
    # ===== burst 命令 =====
    p_burst = subparsers.add_parser('burst', help=t("cli.cmd_burst"))
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

    # ===== identify 命令 =====
    p_identify = subparsers.add_parser('identify', help=t("cli.cmd_identify"))
    p_identify.add_argument('image', help='图片文件路径')
    p_identify.add_argument('-t', '--top', type=int, default=5,
                           help='返回前 N 个结果 (默认: 5)')
    p_identify.add_argument('--no-yolo', action='store_false', dest='yolo',
                           help='禁用 YOLO 裁剪')
    p_identify.add_argument('--no-gps', action='store_false', dest='gps',
                           help='禁用 GPS 过滤')
    p_identify.add_argument('--write-exif', action='store_true',
                           help='将识别结果写入 EXIF Title')
    p_identify.set_defaults(yolo=True, gps=True)

    # 解析参数
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # identify 命令验证文件，其他命令验证目录
    if args.command == 'identify':
        if not os.path.isfile(args.image):
            print(t("cli.file_not_found", path=args.image))
            return 1
        args.image = os.path.abspath(args.image)
    else:
        # 验证目录
        if not os.path.isdir(args.directory):
            print(t("cli.dir_not_found", path=args.directory))
            return 1
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
    elif args.command == 'identify':
        return cmd_identify(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
