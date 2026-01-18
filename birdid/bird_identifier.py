#!/usr/bin/env python3
"""
鸟类识别核心模块
从 SuperBirdID 移植，提供鸟类检测与分类识别功能
"""

__version__ = "1.0.0"

import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from PIL.ExifTags import TAGS, GPSTAGS
import json
import cv2
import os
import sys
from typing import Optional, List, Dict, Tuple, Set

# ==================== 设备配置 ====================
CLASSIFIER_DEVICE = torch.device("cpu")

# ==================== 可选依赖检测 ====================

# RAW格式支持
try:
    import rawpy
    import imageio
    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False

# YOLO检测支持
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ==================== 路径配置 ====================

# birdid 模块目录
BIRDID_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
PROJECT_ROOT = os.path.dirname(BIRDID_DIR)


def get_birdid_path(relative_path: str) -> str:
    """获取 birdid 模块内的资源路径"""
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包环境
        return os.path.join(sys._MEIPASS, 'birdid', relative_path)
    return os.path.join(BIRDID_DIR, relative_path)


def get_project_path(relative_path: str) -> str:
    """获取项目根目录下的资源路径"""
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(PROJECT_ROOT, relative_path)


def get_user_data_dir() -> str:
    """获取用户数据目录"""
    if sys.platform == 'darwin':
        user_data_dir = os.path.expanduser('~/Documents/SuperPicky_Data')
    elif sys.platform == 'win32':
        user_data_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'SuperPicky_Data')
    else:
        user_data_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'SuperPicky_Data')
    os.makedirs(user_data_dir, exist_ok=True)
    return user_data_dir


# ==================== 模型路径 ====================
# 鸟类识别专用模型和数据（在 birdid/ 目录下）
MODEL_PATH = get_birdid_path('models/birdid2024.pt')
MODEL_PATH_ENC = get_birdid_path('models/birdid2024.pt.enc')
BIRD_INFO_PATH = get_birdid_path('data/birdinfo.json')
DATABASE_PATH = get_birdid_path('data/bird_reference.sqlite')
OFFLINE_EBIRD_DIR = get_birdid_path('data/offline_ebird_data')

# YOLO 模型（共用项目根目录的模型）
YOLO_MODEL_PATH = get_project_path('models/yolo11l-seg.pt')

# ==================== 全局变量（懒加载）====================
_classifier = None
_bird_info = None
_db_manager = None
_yolo_detector = None


# ==================== 模型加密解密 ====================

def decrypt_model(encrypted_path: str, password: str) -> bytes:
    """解密模型文件"""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    with open(encrypted_path, 'rb') as f:
        encrypted_data = f.read()

    salt = encrypted_data[:16]
    iv = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(password.encode())

    cipher = Cipher(
        algorithms.AES(key),
        modes.CBC(iv),
        backend=default_backend()
    )
    decryptor = cipher.decryptor()
    plaintext_padded = decryptor.update(ciphertext) + decryptor.finalize()

    padding_length = plaintext_padded[-1]
    return plaintext_padded[:-padding_length]


# ==================== 懒加载函数 ====================

def get_classifier():
    """懒加载分类模型"""
    global _classifier
    if _classifier is None:
        SECRET_PASSWORD = "SuperBirdID_2024_AI_Model_Encryption_Key_v1"

        if os.path.exists(MODEL_PATH_ENC):
            # 加载加密模型
            model_data = decrypt_model(MODEL_PATH_ENC, SECRET_PASSWORD)
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                tmp.write(model_data)
                tmp_path = tmp.name
            _classifier = torch.jit.load(tmp_path, map_location='cpu')
            try:
                os.unlink(tmp_path)
            except:
                pass
        elif os.path.exists(MODEL_PATH):
            _classifier = torch.jit.load(MODEL_PATH, map_location='cpu')
        else:
            raise RuntimeError(f"未找到分类模型: {MODEL_PATH}")

        _classifier.eval()
    return _classifier


def get_bird_info() -> List:
    """懒加载鸟类信息"""
    global _bird_info
    if _bird_info is None:
        if os.path.exists(BIRD_INFO_PATH):
            with open(BIRD_INFO_PATH, 'r', encoding='utf-8') as f:
                _bird_info = json.load(f)
        else:
            _bird_info = []
    return _bird_info


def get_database_manager():
    """懒加载数据库管理器"""
    global _db_manager
    if _db_manager is None:
        try:
            from birdid.bird_database_manager import BirdDatabaseManager
            if os.path.exists(DATABASE_PATH):
                _db_manager = BirdDatabaseManager(DATABASE_PATH)
        except Exception as e:
            print(f"数据库加载失败: {e}")
            _db_manager = False
    return _db_manager if _db_manager is not False else None


def get_yolo_detector():
    """懒加载YOLO检测器"""
    global _yolo_detector
    if _yolo_detector is None and YOLO_AVAILABLE:
        if os.path.exists(YOLO_MODEL_PATH):
            _yolo_detector = YOLOBirdDetector(YOLO_MODEL_PATH)
    return _yolo_detector


# ==================== YOLO 鸟类检测器 ====================

class YOLOBirdDetector:
    """YOLO 鸟类检测器"""

    def __init__(self, model_path: str = None):
        if not YOLO_AVAILABLE:
            self.model = None
            return

        if model_path is None:
            model_path = YOLO_MODEL_PATH

        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            self.model = None

    def detect_and_crop_bird(
        self,
        image_input,
        confidence_threshold: float = 0.25,
        padding: int = 150
    ) -> Tuple[Optional[Image.Image], str]:
        """
        检测并裁剪鸟类区域

        Args:
            image_input: 文件路径或 PIL Image
            confidence_threshold: 置信度阈值
            padding: 裁剪边距

        Returns:
            (裁剪后的图像, 检测信息) 或 (None, 错误信息)
        """
        if self.model is None:
            return None, "YOLO模型未可用"

        try:
            if isinstance(image_input, str):
                image = load_image(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                return None, "不支持的图像输入类型"

            img_array = np.array(image)
            results = self.model(img_array, conf=confidence_threshold)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        # COCO 数据集中鸟类的 class_id 是 14
                        if class_id == 14:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence)
                            })

            if not detections:
                return None, "未检测到鸟类"

            best = max(detections, key=lambda x: x['confidence'])
            img_width, img_height = image.size

            x1, y1, x2, y2 = best['bbox']
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(img_width, x2 + padding)
            y2_padded = min(img_height, y2 + padding)

            cropped = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
            info = f"置信度{best['confidence']:.3f}, 尺寸{cropped.size}"

            return cropped, info

        except Exception as e:
            return None, f"检测失败: {e}"


# ==================== 图像加载 ====================

def load_image(image_path: str) -> Image.Image:
    """
    加载图像，支持标准格式和 RAW 格式
    对 RAW 文件优先提取内嵌 JPEG 预览图（更适合 YOLO 检测）
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"文件不存在: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()

    raw_extensions = [
        '.cr2', '.cr3', '.nef', '.nrw', '.arw', '.srf', '.dng',
        '.raf', '.orf', '.rw2', '.pef', '.srw', '.raw', '.rwl',
        '.3fr', '.fff', '.erf', '.mef', '.mos', '.mrw', '.x3f'
    ]

    if ext in raw_extensions:
        if RAW_SUPPORT:
            try:
                with rawpy.imread(image_path) as raw:
                    # 优先尝试提取内嵌的 JPEG 预览图
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            # 直接使用内嵌的 JPEG
                            from io import BytesIO
                            img = Image.open(BytesIO(thumb.data)).convert("RGB")
                            print(f"[RAW] 使用内嵌 JPEG 预览: {img.size[0]}x{img.size[1]}")
                            return img
                        elif thumb.format == rawpy.ThumbFormat.BITMAP:
                            # 位图格式
                            img = Image.fromarray(thumb.data).convert("RGB")
                            print(f"[RAW] 使用内嵌位图预览: {img.size[0]}x{img.size[1]}")
                            return img
                    except Exception as e:
                        print(f"[RAW] 提取预览失败，使用半尺寸后处理: {e}")
                    
                    # 如果无法提取预览，使用半尺寸后处理
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        output_bps=8,
                        no_auto_bright=False,
                        auto_bright_thr=0.01,
                        half_size=True  # 使用半尺寸，加快处理
                    )
                    img = Image.fromarray(rgb)
                    print(f"[RAW] 使用半尺寸后处理: {img.size[0]}x{img.size[1]}")
                    return img
            except Exception as e:
                raise Exception(f"RAW处理失败: {e}")
        else:
            raise ImportError("需要安装 rawpy 来处理 RAW 格式")
    else:
        return Image.open(image_path).convert("RGB")


# ==================== GPS 提取 ====================

def extract_gps_from_exif(image_path: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    从图像 EXIF 提取 GPS 坐标
    支持 RAW 文件（使用 exiftool）

    Returns:
        (纬度, 经度, 信息) 或 (None, None, 错误信息)
    """
    import subprocess
    import json as json_module
    
    # 首先尝试使用 exiftool（支持 RAW 格式）
    try:
        # 查找 exiftool
        exiftool_paths = [
            '/usr/local/bin/exiftool',
            '/opt/homebrew/bin/exiftool',
            'exiftool',  # 在 PATH 中查找
        ]
        
        exiftool_path = None
        for path in exiftool_paths:
            try:
                result = subprocess.run([path, '-ver'], capture_output=True, timeout=5)
                if result.returncode == 0:
                    exiftool_path = path
                    break
            except:
                continue
        
        if exiftool_path:
            # 使用 exiftool 提取 GPS 信息
            result = subprocess.run(
                [exiftool_path, '-j', '-GPSLatitude', '-GPSLongitude', '-GPSLatitudeRef', '-GPSLongitudeRef', image_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                data = json_module.loads(result.stdout)
                if data and len(data) > 0:
                    gps_data = data[0]
                    
                    lat_str = gps_data.get('GPSLatitude', '')
                    lon_str = gps_data.get('GPSLongitude', '')
                    lat_ref = gps_data.get('GPSLatitudeRef', 'N')
                    lon_ref = gps_data.get('GPSLongitudeRef', 'E')
                    
                    if lat_str and lon_str:
                        # 解析度分秒格式，如 "27 deg 25' 0.53\" S"
                        def parse_dms(dms_str):
                            import re
                            match = re.search(r'(\d+)\s*deg\s*(\d+)\'\s*([\d.]+)"?', str(dms_str))
                            if match:
                                d, m, s = float(match.group(1)), float(match.group(2)), float(match.group(3))
                                return d + m/60 + s/3600
                            # 尝试直接作为数字解析
                            try:
                                return float(dms_str)
                            except:
                                return None
                        
                        lat = parse_dms(lat_str)
                        lon = parse_dms(lon_str)
                        
                        if lat is not None and lon is not None:
                            # 处理南纬 (S 或 South)
                            if lat_ref and lat_ref.upper().startswith('S'):
                                lat = -lat
                            # 处理西经 (W 或 West)
                            if lon_ref and lon_ref.upper().startswith('W'):
                                lon = -lon
                            print(f"[GPS] 从 exiftool 提取: {lat:.6f}, {lon:.6f}")
                            return lat, lon, f"GPS: {lat:.6f}, {lon:.6f}"
    except Exception as e:
        print(f"[GPS] exiftool 提取失败: {e}")
    
    # 回退到 PIL（仅支持 JPEG 等常规格式）
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if not exif_data:
            return None, None, "无EXIF数据"

        gps_info = {}
        for tag, value in exif_data.items():
            decoded_tag = TAGS.get(tag, tag)
            if decoded_tag == "GPSInfo":
                for gps_tag in value:
                    gps_decoded = GPSTAGS.get(gps_tag, gps_tag)
                    gps_info[gps_decoded] = value[gps_tag]
                break

        if not gps_info:
            return None, None, "无GPS数据"

        def convert_to_degrees(coord, ref):
            d, m, s = coord
            decimal = d + (m / 60.0) + (s / 3600.0)
            if ref in ['S', 'W']:
                decimal = -decimal
            return decimal

        lat = None
        lon = None

        if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
            lat = convert_to_degrees(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])

        if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
            lon = convert_to_degrees(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])

        if lat is not None and lon is not None:
            return lat, lon, f"GPS: {lat:.6f}, {lon:.6f}"

        return None, None, "GPS坐标不完整"

    except Exception as e:
        return None, None, f"GPS解析失败: {e}"


# ==================== 图像预处理 ====================

def smart_resize(image: Image.Image, target_size: int = 224) -> Image.Image:
    """智能图像尺寸调整"""
    width, height = image.size
    max_dim = max(width, height)

    if max_dim < 1000:
        return image.resize((target_size, target_size), Image.LANCZOS)

    resized = image.resize((256, 256), Image.LANCZOS)
    left = (256 - target_size) // 2
    top = (256 - target_size) // 2
    return resized.crop((left, top, left + target_size, top + target_size))


def apply_enhancement(image: Image.Image, method: str = "unsharp_mask") -> Image.Image:
    """应用图像增强"""
    if method == "unsharp_mask":
        return image.filter(ImageFilter.UnsharpMask())
    elif method == "edge_enhance_more":
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif method == "contrast_edge":
        enhanced = ImageEnhance.Brightness(image).enhance(1.2)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.3)
        return enhanced.filter(ImageFilter.EDGE_ENHANCE)
    elif method == "desaturate":
        return ImageEnhance.Color(image).enhance(0.5)
    return image


# ==================== 核心识别函数 ====================

def predict_bird(
    image: Image.Image,
    top_k: int = 5,
    ebird_species_set: Optional[Set[str]] = None
) -> List[Dict]:
    """
    识别鸟类

    Args:
        image: PIL Image 对象
        top_k: 返回前 K 个结果
        ebird_species_set: eBird 物种代码集合（用于过滤）

    Returns:
        识别结果列表 [{cn_name, en_name, confidence, ebird_code, ...}, ...]
    """
    model = get_classifier()
    bird_data = get_bird_info()
    db_manager = get_database_manager()

    # 多种增强方法测试
    enhancement_methods = [
        ("none", None),
        ("edge_enhance_more", "edge_enhance_more"),
        ("unsharp_mask", "unsharp_mask"),
        ("contrast_edge", "contrast_edge"),
        ("desaturate", "desaturate")
    ]

    all_logits = []  # 收集所有增强方法的logits用于融合

    for name, method in enhancement_methods:
        if method:
            enhanced = apply_enhancement(image, method)
        else:
            enhanced = image

        processed = smart_resize(enhanced, 224)

        # 转换为 tensor
        img_array = np.array(processed)
        bgr_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        mean = np.array([0.406, 0.456, 0.485])
        std = np.array([0.225, 0.224, 0.229])

        normalized = (bgr_array / 255.0 - mean) / std
        input_tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            output = model(input_tensor)

        # 收集原始logits用于融合
        all_logits.append(output[0])

    # 多增强融合：对所有增强方法的logits取平均
    if len(all_logits) == 0:
        return []
    
    stacked_logits = torch.stack(all_logits)
    fused_logits = stacked_logits.mean(dim=0)
    
    # 对融合后的logits应用温度缩放和softmax
    TEMPERATURE = 0.5
    best_probs = torch.nn.functional.softmax(fused_logits / TEMPERATURE, dim=0)

    # 获取 top-k 结果
    k = min(100 if ebird_species_set else top_k, len(best_probs))
    top_probs, top_indices = torch.topk(best_probs, k)

    results = []
    for i in range(len(top_indices)):
        class_id = top_indices[i].item()
        confidence = top_probs[i].item() * 100
        # 置信度阈值：使用 eBird 过滤时降低阈值以保留更多候选
        min_confidence = 0.3 if ebird_species_set else 1.0
        if confidence < min_confidence:
            continue

        cn_name = None
        en_name = None
        scientific_name = None
        ebird_code = None

        # 优先从数据库获取
        if db_manager:
            info = db_manager.get_bird_by_class_id(class_id)
            if info:
                cn_name = info.get('chinese_simplified')
                en_name = info.get('english_name')
                scientific_name = info.get('scientific_name')
                ebird_code = info.get('ebird_code')

        # 回退到 bird_data
        if not cn_name and class_id < len(bird_data) and len(bird_data[class_id]) >= 2:
            cn_name = bird_data[class_id][0]
            en_name = bird_data[class_id][1]

        if not cn_name:
            cn_name = f"Unknown (ID: {class_id})"
            en_name = f"Unknown (ID: {class_id})"

        # eBird 过滤
        ebird_match = False
        if ebird_species_set:
            if not ebird_code and db_manager and en_name:
                ebird_code = db_manager.get_ebird_code_by_english_name(en_name)

            if ebird_code and ebird_code in ebird_species_set:
                ebird_match = True
            elif ebird_species_set:
                # 调试：显示被过滤掉的候选
                if i < 5:  # 只显示前5个被过滤的
                    print(f"[eBird过滤] 跳过: {cn_name} ({en_name}), ebird_code={ebird_code}, 置信度={confidence:.1f}%")
                continue  # 不在列表中，跳过

        results.append({
            'class_id': class_id,
            'cn_name': cn_name,
            'en_name': en_name,
            'scientific_name': scientific_name,
            'confidence': confidence,
            'ebird_code': ebird_code,
            'ebird_match': ebird_match
        })

        if len(results) >= top_k:
            break

    return results


def identify_bird(
    image_path: str,
    use_yolo: bool = True,
    use_gps: bool = True,
    use_ebird: bool = True,
    country_code: str = None,
    region_code: str = None,
    top_k: int = 5
) -> Dict:
    """
    端到端鸟类识别

    Args:
        image_path: 图像路径
        use_yolo: 是否使用 YOLO 裁剪
        use_gps: 是否使用 GPS 自动检测区域
        use_ebird: 是否启用 eBird 区域过滤
        country_code: 手动指定国家代码（如 "AU"）
        region_code: 手动指定区域代码（如 "AU-SA"）
        top_k: 返回前 K 个结果

    Returns:
        识别结果字典
    """
    result = {
        'success': False,
        'image_path': image_path,
        'results': [],
        'yolo_info': None,
        'gps_info': None,
        'ebird_info': None,
        'error': None
    }

    try:
        # 加载图像
        image = load_image(image_path)

        # YOLO 裁剪
        print(f"[YOLO调试] use_yolo={use_yolo}, YOLO_AVAILABLE={YOLO_AVAILABLE}")
        if use_yolo and YOLO_AVAILABLE:
            width, height = image.size
            print(f"[YOLO调试] 图片尺寸: {width}x{height}")
            if max(width, height) > 640:
                detector = get_yolo_detector()
                print(f"[YOLO调试] detector={detector is not None}")
                if detector:
                    cropped, info = detector.detect_and_crop_bird(image)
                    print(f"[YOLO调试] 检测结果: cropped={cropped is not None}, info={info}")
                    if cropped:
                        image = cropped
                        result['yolo_info'] = info
                        print(f"[YOLO调试] ✅ 已裁剪鸟类区域")
                    else:
                        print(f"[YOLO调试] ⚠️ 未检测到鸟类")
            else:
                print(f"[YOLO调试] 图片太小，跳过YOLO裁剪")
        else:
            print(f"[YOLO调试] YOLO未启用或不可用")

        # eBird 区域过滤
        ebird_species_set = None
        detected_region = None
        
        if use_ebird:
            try:
                from birdid.ebird_country_filter import eBirdCountryFilter
                import os as os_module
                
                api_key = os_module.environ.get('EBIRD_API_KEY', '60nan25sogpo')
                cache_dir = os_module.path.join(get_user_data_dir(), 'ebird_cache')
                offline_dir = get_birdid_path('data/offline_ebird_data')
                
                ebird_filter = eBirdCountryFilter(api_key, cache_dir=cache_dir, offline_dir=offline_dir)
                
                # 确定要使用的区域
                effective_region = None
                data_source = None
                
                # 优先级：region_code > country_code > GPS 自动检测
                if region_code:
                    effective_region = region_code
                    data_source = f"手动选择: {region_code}"
                elif country_code:
                    effective_region = country_code
                    data_source = f"手动选择: {country_code}"
                elif use_gps:
                    # 尝试从 EXIF 提取 GPS 并自动检测区域
                    lat, lon, gps_msg = extract_gps_from_exif(image_path)
                    if lat and lon:
                        result['gps_info'] = {
                            'latitude': lat,
                            'longitude': lon,
                            'info': gps_msg
                        }
                        # 只用 GPS 判定国家/区域代码（不调用 eBird API）
                        detected_region, region_name = ebird_filter.get_region_code_from_gps(lat, lon)
                        if detected_region:
                            effective_region = detected_region
                            data_source = f"GPS自动检测: {detected_region}"
                            print(f"[GPS] 自动检测区域: {detected_region} ({region_name})")
                
                # 使用区域代码获取物种列表（优先离线数据）
                if effective_region:
                    species_set = ebird_filter.get_country_species_list(effective_region)
                    if species_set:
                        ebird_species_set = species_set
                        print(f"[eBird] 使用区域 {effective_region} 的物种列表: {len(species_set)} 个物种")
                
                # 记录 eBird 过滤信息
                if ebird_species_set:
                    result['ebird_info'] = {
                        'enabled': True,
                        'region': effective_region,
                        'species_count': len(ebird_species_set),
                        'data_source': data_source
                    }
                    
            except ImportError:
                print("eBird 过滤模块不可用")
            except Exception as e:
                print(f"eBird 过滤初始化失败: {e}")

        # 执行识别
        results = predict_bird(image, top_k=top_k, ebird_species_set=ebird_species_set)

        result['success'] = True
        result['results'] = results

    except Exception as e:
        result['error'] = str(e)

    return result


# ==================== 便捷函数 ====================

def quick_identify(image_path: str, top_k: int = 3) -> List[Dict]:
    """
    快速识别（简化接口）

    Returns:
        识别结果列表
    """
    result = identify_bird(image_path, top_k=top_k)
    return result.get('results', [])


# ==================== 测试 ====================

if __name__ == "__main__":
    print("BirdIdentifier 模块测试")
    print(f"YOLO 可用: {YOLO_AVAILABLE}")
    print(f"RAW 支持: {RAW_SUPPORT}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"数据库路径: {DATABASE_PATH}")
