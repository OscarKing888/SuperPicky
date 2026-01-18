import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from utils import log_message, write_to_csv
from config import config
# V3.2: 移除未使用的 sharpness 计算器导入
from iqa_scorer import get_iqa_scorer
from advanced_config import get_advanced_config

# 禁用 Ultralytics 设置警告
os.environ['YOLO_VERBOSE'] = 'False'

def _resolve_device(device):
    """Resolve requested device string to a supported value."""
    if device is None:
        device = 'auto'

    if isinstance(device, str):
        device = device.lower()

    try:
        import torch
        has_cuda = torch.cuda.is_available()
        has_mps = torch.backends.mps.is_available()
    except Exception:
        has_cuda = False
        has_mps = False

    if device in ('', 'auto'):
        if has_cuda:
            return 'cuda'
        if has_mps:
            return 'mps'
        return 'cpu'

    if isinstance(device, str) and device.startswith('cuda'):
        return 'cuda' if has_cuda else 'cpu'

    if device == 'mps':
        return 'mps' if has_mps else 'cpu'

    if device == 'cpu':
        return 'cpu'

    return 'cpu'



def load_yolo_model(device='auto'):
    """?? YOLO ?????????"""
    model_path = config.ai.get_model_path()
    model = YOLO(str(model_path))

    resolved_device = _resolve_device(device)
    model._sp_device = resolved_device

    return model


def preprocess_image(image_path, target_size=None):
    """预处理图像"""
    if target_size is None:
        target_size = config.ai.TARGET_IMAGE_SIZE
    
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    scale = target_size / max(w, h)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# V3.2: 移除 _get_sharpness_calculator（锐度现在由 keypoint_detector 计算）

# 初始化全局 IQA 评分器（延迟加载）
_iqa_scorer = None


def _get_iqa_scorer():
    """获取 IQA 评分器单例"""
    global _iqa_scorer
    if _iqa_scorer is None:
        _iqa_scorer = get_iqa_scorer(device='mps')
    return _iqa_scorer


def detect_and_draw_birds(image_path, model, output_path, dir, ui_settings, i18n=None, skip_nima=False, device=None):
    """
    检测并标记鸟类（V3.1 - 简化版，移除预览功能）

    Args:
        image_path: 图片路径
        model: YOLO模型
        output_path: 输出路径（带框图片）
        dir: 工作目录
        ui_settings: [ai_confidence, sharpness_threshold, nima_threshold, save_crop, normalization_mode]
        i18n: I18n instance for internationalization (optional)
        skip_nima: 如果为True，跳过NIMA计算（用于双眼不可见的情况）
    """
    # V3.1: 从 ui_settings 获取参数
    from core.config_manager import UISettings
    # 支持 UISettings 数据类和列表（向后兼容）
    if isinstance(ui_settings, UISettings):
        ai_confidence = ui_settings.ai_confidence / 100  # AI置信度：50-100 -> 0.5-1.0（仅用于过滤）
        sharpness_threshold = ui_settings.sharpness_threshold  # 锐度阈值：6000-9000
        nima_threshold = ui_settings.nima_threshold  # NIMA美学阈值：5.0-6.0
    else:
        # 向后兼容列表格式
        ai_confidence = ui_settings[0] / 100  # AI置信度：50-100 -> 0.5-1.0（仅用于过滤）
        sharpness_threshold = ui_settings[1]  # 锐度阈值：6000-9000
        nima_threshold = ui_settings[2]       # NIMA美学阈值：5.0-6.0

    # V3.1: 不再保存Crop图片（移除预览功能）
    save_crop = False

    # V3.2: 移除未使用的 normalization_mode 和 sharpness_calculator
    # 锐度现在由 photo_processor 中的 keypoint_detector 计算

    found_bird = False
    bird_sharp = False
    bird_result = False
    nima_score = None  # 美学评分
    # V3.2: 移除 BRISQUE（不再使用）

    # 使用配置检查文件类型
    if not config.is_jpg_file(image_path):
        log_message("ERROR: not a jpg file", dir)
        return None

    if not os.path.exists(image_path):
        log_message(f"ERROR: in detect_and_draw_birds, {image_path} not found", dir)
        return None

    # 记录总处理开始时间
    total_start = time.time()

    # Step 1: 图像预处理
    step_start = time.time()
    image = preprocess_image(image_path)
    height, width, _ = image.shape
    preprocess_time = (time.time() - step_start) * 1000
    # V3.3: 简化日志，移除步骤详情
    # log_message(f"  ⏱️  [1/4] 图像预处理: {preprocess_time:.1f}ms", dir)

    # Step 2: YOLO??
    step_start = time.time()
    requested_device = device
    if requested_device:
        requested_device = _resolve_device(requested_device)
    else:
        requested_device = getattr(model, '_sp_device', None)
    if requested_device in ('', 'auto'):
        requested_device = None
    try:
        if requested_device:
            results = model(image, device=requested_device)
        else:
            results = model(image)
    except Exception as primary_error:
        if requested_device and requested_device != 'cpu':
            log_message(f"WARNING: {requested_device.upper()} inference failed, fallback to CPU: {primary_error}", dir)
            try:
                results = model(image, device='cpu')
            except Exception as cpu_error:
                log_message(f"ERROR: AI inference failed: {cpu_error}", dir)
                data = {
                    "filename": os.path.splitext(os.path.basename(image_path))[0],
                    "has_bird": "no",
                    "confidence": 0.0,
                    "head_sharp": "-",
                    "left_eye": "-",
                    "right_eye": "-",
                    "beak": "-",
                    "nima_score": "-",
                    "rating": -1
                }
                write_to_csv(data, dir, False)
                return found_bird, bird_result, 0.0, 0.0, None, None, None, None
        else:
            log_message(f"ERROR: AI inference failed: {primary_error}", dir)
            data = {
                "filename": os.path.splitext(os.path.basename(image_path))[0],
                "has_bird": "no",
                "confidence": 0.0,
                "head_sharp": "-",
                "left_eye": "-",
                "right_eye": "-",
                "beak": "-",
                "nima_score": "-",
                "rating": -1
            }
            write_to_csv(data, dir, False)
            return found_bird, bird_result, 0.0, 0.0, None, None, None, None
    
    yolo_time = (time.time() - step_start) * 1000
    # V3.3: 简化日志，移除步骤详情
    # if i18n:
    #     log_message(i18n.t("logs.yolo_inference", time=yolo_time), dir)
    # else:
    #     log_message(f"  ⏱️  [2/4] YOLO推理: {yolo_time:.1f}ms", dir)

    # Step 3: 解析检测结果
    step_start = time.time()
    detections = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    # 获取掩码数据（如果是分割模型）
    masks = None
    if hasattr(results[0], 'masks') and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

    # 只处理置信度最高的鸟
    bird_idx = -1
    max_conf = 0

    for idx, (detection, conf, class_id) in enumerate(zip(detections, confidences, class_ids)):
        if int(class_id) == config.ai.BIRD_CLASS_ID:
            if conf > max_conf:
                max_conf = conf
                bird_idx = idx

    parse_time = (time.time() - step_start) * 1000
    # V3.3: 简化日志，移除步骤详情
    # if i18n:
    #     log_message(i18n.t("logs.result_parsing", time=parse_time), dir)
    # else:
    #     log_message(f"  ⏱️  [3/4] 结果解析: {parse_time:.1f}ms", dir)

    # 如果没有找到鸟，记录到CSV并返回（V3.1）
    if bird_idx == -1:
        # V3.3: 使用英文列名
        data = {
            "filename": os.path.splitext(os.path.basename(image_path))[0],
            "has_bird": "no",
            "confidence": 0.0,
            "head_sharp": "-",
            "left_eye": "-",
            "right_eye": "-",
            "beak": "-",
            "nima_score": "-",
            "rating": -1
        }
        write_to_csv(data, dir, False)
        return found_bird, bird_result, 0.0, 0.0, None, None, None, None  # V3.7: 8 values including mask
    # V3.2: 移除 NIMA 计算（现在由 photo_processor 在裁剪区域上计算）
    # nima_score 设为 None，photo_processor 会重新计算
    nima_score = None
    
    # V3.9.3: 提前声明默认值，避免 continue 后变量未定义
    sharpness = 0.0
    x, y, w, h = 0, 0, 0, 0

    # 只处理面积最大的那只鸟
    for idx, (detection, conf, class_id) in enumerate(zip(detections, confidences, class_ids)):
        # 跳过非鸟类或非最大面积的鸟
        if idx != bird_idx:
            continue
        x1, y1, x2, y2 = detection

        x = int(x1)
        y = int(y1)
        w = int(x2 - x1)
        h = int(y2 - y1)
        class_id = int(class_id)

        # 使用配置中的鸟类类别 ID
        if class_id == config.ai.BIRD_CLASS_ID:
            found_bird = True
            area_ratio = (w * h) / (width * height)
            filename = os.path.basename(image_path)

            # V3.1: 不再保存Crop图片
            crop_path = None

            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = min(w, width - x)
            h = min(h, height - y)

            if w <= 0 or h <= 0:
                log_message(f"ERROR: Invalid crop region for {image_path}", dir)
                continue

            crop_img = image[y:y + h, x:x + w]

            if crop_img is None or crop_img.size == 0:
                log_message(f"ERROR: Crop image is empty for {image_path}", dir)
                continue

            # V3.2: 移除 Step 5 锐度计算（现在由 photo_processor 中的 keypoint_detector 计算 head_sharpness）
            # 设置占位值以保持 CSV 兼容性
            real_sharpness = 0.0
            sharpness = 0.0
            effective_pixels = 0

            # V3.2: 移除 BRISQUE 评估（不再使用）

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # V3.1: 新的评分逻辑
            # 计算中心坐标（仅用于日志输出）
            center_x = (x + w / 2) / width
            center_y = (y + h / 2) / height

            # V3.3: 简化日志，移除AI详情输出
            # log_message(f" AI: {conf:.2f} - Class: {class_id} "
            #             f"- Area:{area_ratio * 100:.2f}% - Pixels:{effective_pixels:,d}"
            #             f" - Center_x:{center_x:.2f} - Center_y:{center_y:.2f}", dir)

            # V3.2: 移除评分逻辑（现在由 photo_processor 的 RatingEngine 计算）
            # rating_value 设为占位值，photo_processor 会重新计算
            rating_value = 0

            # V3.3: 使用英文列名
            data = {
                "filename": os.path.splitext(os.path.basename(image_path))[0],
                "has_bird": "yes" if found_bird else "no",
                "confidence": float(f"{conf:.2f}"),
                "head_sharp": "-",        # 将由 photo_processor 填充
                "left_eye": "-",          # 将由 photo_processor 填充
                "right_eye": "-",         # 将由 photo_processor 填充
                "beak": "-",              # 将由 photo_processor 填充
                "nima_score": float(f"{nima_score:.2f}") if nima_score is not None else "-",
                "rating": rating_value
            }

            # Step 5: CSV写入
            step_start = time.time()
            write_to_csv(data, dir, False)
            csv_time = (time.time() - step_start) * 1000
            # V3.3: 简化日志
            # log_message(f"  ⏱️  [4/4] CSV写入: {csv_time:.1f}ms", dir)

    # --- 修改开始 ---
    # 只有在 found_bird 为 True 且 output_path 有效时，才保存带框的图片
    if found_bird and output_path:
        cv2.imwrite(output_path, image)
    # --- 修改结束 ---

    # 计算总处理时间 (V3.3: 移除此处日志, 由 photo_processor 输出真正总耗时)
    total_time = (time.time() - total_start) * 1000
    # log_message(f"  ⏱️  ========== 总耗时: {total_time:.1f}ms ==========", dir)

    # 返回 found_bird, bird_result, AI置信度, 归一化锐度, NIMA分数, bbox, 图像尺寸, 分割掩码
    bird_confidence = float(confidences[bird_idx]) if bird_idx != -1 else 0.0
    bird_sharpness = sharpness if bird_idx != -1 else 0.0
    # bbox 格式: (x, y, w, h) - 在缩放后的图像上
    # img_dims 格式: (width, height) - 缩放后图像的尺寸，用于计算缩放比例
    bird_bbox = (x, y, w, h) if found_bird else None
    img_dims = (width, height) if found_bird else None
    
    # 获取对应鸟的掩码
    bird_mask = None
    if found_bird and masks is not None:
        # masks shape: (N, H, W) where N is number of detections
        # YOLO masks are usually same size as input image (or smaller and upscaled)
        # Ultralytics results.masks.data is usually (N, H, W) 
        # But we need to be careful about resizing if it's smaller
        # results.masks.data contains masks for all detections
        # We need the one corresponding to bird_idx
        try:
            # Mask is already resized to image size by ultralytics by default in modern versions
            # But let's verify if we need to resize
            # results[0].masks.data is a torch tensor on GPU/CPU
            raw_mask = results[0].masks.data[bird_idx].cpu().numpy()
            
            # Ensure mask is same size as processed image (width, height)
            if raw_mask.shape != (height, width):
                raw_mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Convert to binary uint8 mask (0 or 255)
            # YOLO masks are float [0,1], threshold at 0.5
            bird_mask = (raw_mask > 0.5).astype(np.uint8) * 255
        except Exception as e:
            # Mask processing failed, ignore
            pass

    return found_bird, bird_result, bird_confidence, bird_sharpness, nima_score, bird_bbox, img_dims, bird_mask