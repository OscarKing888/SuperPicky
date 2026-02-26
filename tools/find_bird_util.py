import os
import rawpy
import imageio
import subprocess
import sys
from .utils import log_message
from .exiftool_manager import get_exiftool_manager
import glob
import shutil

from .file_utils import ensure_hidden_directory
from constants import RAW_EXTENSIONS, JPG_EXTENSIONS


HEIF_EXTENSIONS = {'.heic', '.heif', '.hif'}


def _extract_preview_with_exiftool_to_jpeg(source_path: str, target_jpg_path: str) -> bool:
    """
    使用 ExifTool 从 HEIF/RAW-like 文件中提取 JPEG 预览图。

    依次尝试 PreviewImage / JpgFromRaw / ThumbnailImage。
    """
    mgr = get_exiftool_manager()
    exiftool_path = mgr.exiftool_path
    exiftool_cwd = os.path.dirname(os.path.abspath(exiftool_path))
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith('win') else 0

    for tag in ('PreviewImage', 'JpgFromRaw', 'ThumbnailImage'):
        cmd = [exiftool_path, '-b', f'-{tag}', source_path]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=False,
                timeout=30,
                creationflags=creationflags,
                cwd=exiftool_cwd,
            )
        except Exception:
            continue

        preview_bytes = result.stdout or b""
        if result.returncode != 0 or not preview_bytes:
            continue

        try:
            with open(target_jpg_path, 'wb') as f:
                f.write(preview_bytes)
            return True
        except Exception:
            continue

    return False


def _heif_to_jpeg(heif_file_path: str, jpg_file_path: str):
    """
    将 HEIF/HEIC/HIF 转为临时 JPEG。

    优先尝试 Pillow（如环境安装了 pillow-heif），失败则回退 ExifTool 提取预览图。
    """
    # 1) Pillow 路径（支持安装 pillow-heif 的环境）
    try:
        try:
            import pillow_heif  # type: ignore
            pillow_heif.register_heif_opener()
        except Exception:
            pass

        from PIL import Image
        with Image.open(heif_file_path) as im:
            im.convert("RGB").save(jpg_file_path, format="JPEG", quality=95)
        return
    except Exception:
        pass

    # 2) ExifTool 预览提取路径（更接近 RAW 逻辑）
    if _extract_preview_with_exiftool_to_jpeg(heif_file_path, jpg_file_path):
        return

    raise RuntimeError("无法解码 HEIF/HEIC/HIF（Pillow/ExifTool 预览提取均失败）")

def raw_to_jpeg(raw_file_path):
    filename = os.path.basename(raw_file_path)
    file_prefix, file_ext = os.path.splitext(filename)
    directory_path = os.path.dirname(raw_file_path)
    
    # V4.1.0: 使用 .superpicky/cache 目录存储临时 JPEG
    superpicky_dir = os.path.join(directory_path, ".superpicky")
    cache_dir = os.path.join(superpicky_dir, "cache", "temp_preview")
    
    # 确保目录存在并隐藏
    ensure_hidden_directory(superpicky_dir)
    ensure_hidden_directory(cache_dir)
    
    # 文件名不带 tmp_ 前缀，直接使用原名前缀
    jpg_file_path = os.path.join(cache_dir, f"{file_prefix}.jpg")
    
    if os.path.exists(jpg_file_path):
        return jpg_file_path  # 返回完整路径（缓存命中，无需日志）
        
    if not os.path.exists(raw_file_path):
        log_message(f"ERROR, file [{filename}] cannot be found in RAW form", directory_path)
        return None

    try:
        if file_ext.lower() in HEIF_EXTENSIONS:
            _heif_to_jpeg(raw_file_path, jpg_file_path)
            return jpg_file_path

        with rawpy.imread(raw_file_path) as raw:
            thumbnail = raw.extract_thumb()
            if thumbnail is None:
                return None
            if thumbnail.format == rawpy.ThumbFormat.JPEG:
                with open(jpg_file_path, 'wb') as f:
                    f.write(thumbnail.data)
            elif thumbnail.format == rawpy.ThumbFormat.BITMAP:
                imageio.imsave(jpg_file_path, thumbnail.data)
            # 成功转换——已由 photo_processor 的批量日志统计，无需逐文件记录
            
            return jpg_file_path  # 返回完整路径
    except Exception as e:
        log_message(f"Error occurred while converting the RAW-like file:{raw_file_path}, Error: {e}", directory_path)
        raise e  # 抛出异常供调用者捕获

def reset(directory, log_callback=None, i18n=None):
    """
    重置工作目录：
    1. 清理临时文件和日志
    2. 重置所有照片的EXIF元数据（Rating、Pick、Label）

    Args:
        directory: 工作目录
        log_callback: 日志回调函数（可选，用于UI显示）
        i18n: I18n instance for internationalization (optional)
    """
    def log(msg):
        """统一日志输出"""
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    if not os.path.exists(directory):
        if i18n:
            log(i18n.t("errors.dir_not_exist", directory=directory))
        else:
            log(f"ERROR: {directory} does not exist")
        return False

    if i18n:
        log(i18n.t("logs.reset_start"))
        log(i18n.t("logs.reset_dir", directory=directory))
    else:
        log(f"🔄 开始重置目录: {directory}")

    # 1. 清理临时文件、日志和Crop图片
    if i18n:
        log("\n" + i18n.t("logs.clean_tmp"))
    else:
        log("\n📁 清理临时文件...")

    # 1.1 清理 _tmp 目录（包含所有临时文件、日志、crop图片等）
    tmp_dir = os.path.join(directory, ".superpicky")
    if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
        try:
            # 先逐文件清空（含 ExFAT 上的 ._* 资源分叉文件），再删目录
            import stat
            for dirpath, dirnames, filenames in os.walk(tmp_dir, topdown=False):
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    try:
                        os.remove(fpath)
                    except Exception:
                        try:
                            os.chmod(fpath, stat.S_IWRITE | stat.S_IREAD)
                            os.remove(fpath)
                        except Exception:
                            pass
                for dname in dirnames:
                    dpath = os.path.join(dirpath, dname)
                    try:
                        os.rmdir(dpath)
                    except Exception:
                        pass
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if i18n:
                log(i18n.t("logs.tmp_deleted"))
            else:
                log(f"  ✅ 已删除 _tmp 目录及其所有内容")
        except Exception as e:
            if i18n:
                log(i18n.t("logs.tmp_delete_failed", error=str(e)))
            else:
                log(f"  ❌ 删除 _tmp 目录失败: {e}")
            # 尝试使用系统命令强制删除（macOS/Linux）
            try:
                import subprocess
                if os.name == 'nt':
                     subprocess.run(['cmd', '/c', 'rd', '/s', '/q', tmp_dir], check=True)
                else:
                    subprocess.run(['rm', '-rf', tmp_dir], check=True)
                if i18n:
                    log(i18n.t("logs.tmp_force_delete"))
                else:
                    log(f"  ✅ 使用系统命令强制删除 _tmp 成功")
            except Exception as e2:
                if i18n:
                    log(i18n.t("logs.tmp_force_failed", error=str(e2)))
                else:
                    log(f"  ❌ 强制删除也失败: {e2}")

    # 1.2 清理旧版本的日志和CSV文件（如果存在于根目录）
    files_to_clean = [".report.csv", ".report.db", ".process_log.txt", "superpicky.log"]
    for name in files_to_clean:
        path = os.path.join(directory, name)
        if os.path.exists(path) and os.path.isfile(path):
            try:
                os.remove(path)
                log(f"  ✅ 已删除: {name}")
            except Exception as e:
                log(f"  ❌ 删除失败 {name}: {e}")

    # 1.3 清理临时JPEG文件（tmp_*.jpg，如果有遗留在根目录的）
    tmp_jpg_pattern = os.path.join(directory, "tmp_*.jpg")
    tmp_jpg_files = glob.glob(tmp_jpg_pattern)
    tmp_jpg_files = [f for f in tmp_jpg_files if not os.path.basename(f).startswith('.')]
    if tmp_jpg_files:
        log(f"  发现 {len(tmp_jpg_files)} 个临时JPEG文件（tmp_*.jpg），正在删除...")
        deleted_tmp = 0
        for tmp_file in tmp_jpg_files:
            try:
                os.remove(tmp_file)
                deleted_tmp += 1
            except Exception as e:
                log(f"  ❌ 删除失败 {os.path.basename(tmp_file)}: {e}")
        if deleted_tmp > 0:
            log(f"  ✅ 临时JPEG删除完成: {deleted_tmp} 成功")

    # 2. 删除所有XMP侧车文件（Lightroom会优先读取XMP）
    if i18n:
        log("\n" + i18n.t("logs.delete_xmp"))
    else:
        log("\n🗑️  删除XMP侧车文件...")
    xmp_pattern = os.path.join(directory, "**/*.xmp")
    xmp_files = glob.glob(xmp_pattern, recursive=True)
    # 过滤掉隐藏文件
    xmp_files = [f for f in xmp_files if not os.path.basename(f).startswith('.')]
    if xmp_files:
        if i18n:
            log(i18n.t("logs.xmp_found", count=len(xmp_files)))
        else:
            log(f"  发现 {len(xmp_files)} 个XMP文件，正在删除...")
        deleted_xmp = 0
        for xmp_file in xmp_files:
            try:
                os.remove(xmp_file)
                deleted_xmp += 1
            except Exception as e:
                log(f"  ❌ 删除失败 {os.path.basename(xmp_file)}: {e}")
        if i18n:
            log(i18n.t("logs.xmp_deleted", count=deleted_xmp))
        else:
            log(f"  ✅ XMP文件删除完成: {deleted_xmp} 成功")
    else:
        if i18n:
            log(i18n.t("logs.xmp_not_found"))
        else:
            log("  ℹ️  未找到XMP文件")

    # 3. 重置所有图片文件的EXIF元数据
    if i18n:
        log("\n" + i18n.t("logs.reset_exif"))
    else:
        log("\n🏷️  重置EXIF元数据...")

    # 支持的图片格式
    image_extensions = []
    for ext in RAW_EXTENSIONS + JPG_EXTENSIONS:
        image_extensions.append(f"*{ext.lower()}")
        image_extensions.append(f"*{ext.upper()}")

    # 收集所有图片文件（跳过隐藏文件）
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(directory, ext)
        files = glob.glob(pattern)
        # 过滤掉隐藏文件（以.开头的文件）
        files = [f for f in files if not os.path.basename(f).startswith('.')]
        image_files.extend(files)

    # V3.9.4: 对文件列表执行去重（Windows 下 *.NEF 和 *.nef 匹配结果相同，会导致计数翻倍）
    image_files = sorted(list(set(os.path.abspath(f) for f in image_files)))

    if image_files:
        if i18n:
            log(i18n.t("logs.images_found", count=len(image_files)))
        else:
            log(f"  发现 {len(image_files)} 个图片文件")

        try:
            # 使用批量重置功能（传递log_callback和i18n）
            manager = get_exiftool_manager()
            stats = manager.batch_reset_metadata(image_files, log_callback=log_callback, i18n=i18n)

            if i18n:
                log(i18n.t("logs.batch_complete", success=stats['success'], skipped=stats.get('skipped', 0), failed=stats['failed']))
            else:
                log(f"  ✅ EXIF重置完成: {stats['success']} 成功, {stats.get('skipped', 0)} 跳过(4-5星), {stats['failed']} 失败")

        except Exception as e:
            if i18n:
                log(i18n.t("logs.exif_reset_failed", error=str(e)))
            else:
                log(f"  ❌ EXIF重置失败: {e}")
            return False
    else:
        if i18n:
            log(i18n.t("logs.no_images"))
        else:
            log("  ⚠️  未找到图片文件")

    if i18n:
        log("\n" + i18n.t("logs.reset_complete"))
    else:
        log("\n✅ 目录重置完成！")
    return True
