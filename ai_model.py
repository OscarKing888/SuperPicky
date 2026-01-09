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


def load_yolo_model():
    """加载 YOLO 模型（自动启用 GPU 加速：MPS/CUDA）"""
    from utils import get_best_device
    
    model_path = config.ai.get_model_path()
    model = YOLO(str(model_path))

    # 自动检测并使用最佳 GPU 设备
    try:
        import torch
        device = get_best_device('auto')
        
        if device == 'mps':
            if torch.backends.mps.is_available():
                print("✅ 检测到 Apple GPU (MPS)，启用硬件加速")
            else:
                print("⚠️  MPS 不可用，使用 CPU 推理")
                device = 'cpu'
        elif device == 'cuda':
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "NVIDIA GPU"
                print(f"✅ 检测到 {gpu_name} (CUDA)，启用硬件加速")
            else:
                print("⚠️  CUDA 不可用，使用 CPU 推理")
                device = 'cpu'
        else:
            print("⚠️  使用 CPU 推理")
        
        # 保存设备信息供后续使用
        model._device = device
        
    except Exception as e:
        print(f"⚠️  GPU检测失败: {e}，使用CPU推理")
        model._device = 'cpu'

    return model


def preprocess_image(image_path, target_size=None):
    """预处理图像"""
    from utils import read_image
    
    if target_size is None:
        target_size = config.ai.TARGET_IMAGE_SIZE
    
    img = read_image(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    h, w = img.shape[:2]
    scale = target_size / max(w, h)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# V3.2: 移除 _get_sharpness_calculator（锐度现在由 keypoint_detector 计算）

# 初始化全局 IQA 评分器（延迟加载）
_iqa_scorer = None


def _get_iqa_scorer():
    """获取 IQA 评分器单例"""
    from utils import get_best_device
    global _iqa_scorer
    if _iqa_scorer is None:
        device = get_best_device('auto')
        _iqa_scorer = get_iqa_scorer(device=device)
    return _iqa_scorer


def detect_and_draw_birds(image_path, model, output_path, dir, ui_settings, i18n=None, skip_nima=False):
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

    # 检查文件是否存在
    if not os.path.exists(image_path):
        log_message(f"ERROR: in detect_and_draw_birds, {image_path} not found", dir)
        return None

    # 对于 HEIF/HEIC/HIF 文件，需要先转换为临时 JPG 文件
    # 因为某些 AI 模型可能无法直接处理这些格式
    temp_jpg_path = None
    file_ext = os.path.splitext(image_path)[1].lower()
    is_heif_format = file_ext in ['.heif', '.heic', '.hif']
    
    if is_heif_format:
        try:
            # 创建临时 JPG 文件
            import tempfile
            temp_dir = os.path.join(dir, '.superpicky', 'temp_jpg')
            os.makedirs(temp_dir, exist_ok=True)
            
            file_basename = os.path.splitext(os.path.basename(image_path))[0]
            temp_jpg_path = os.path.join(temp_dir, f"{file_basename}_temp.jpg")
            
            # 使用 PIL + pillow-heif 读取并转换为 JPG
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            except ImportError:
                pass
            
            from PIL import Image
            pil_image = Image.open(image_path).convert('RGB')
            pil_image.save(temp_jpg_path, 'JPEG', quality=95)
            
            # 使用临时 JPG 文件进行后续处理（仅用于 AI 推理）
            # 注意：原始文件路径（image_path 的原始值）不会被修改，
            # 调用者传入的文件路径保持不变，EXIF 会写入原始文件
            original_image_path = image_path  # 保存原始路径
            image_path = temp_jpg_path  # 临时使用 JPG 进行 AI 推理
            log_message(f"🔄 已转换 {file_ext.upper()} 为临时 JPG（仅用于 AI 推理，EXIF 将写入原始文件）", dir)
        except Exception as e:
            log_message(f"⚠️  HEIF 转换失败，尝试直接处理: {e}", dir)
            # 如果转换失败，继续尝试直接处理
    
    # 使用配置检查文件类型（现在应该是 JPG 或已转换的临时文件）
    if not config.is_jpg_file(image_path) and not is_heif_format:
        log_message("ERROR: not a jpg file", dir)
        if temp_jpg_path and os.path.exists(temp_jpg_path):
            try:
                os.remove(temp_jpg_path)
            except:
                pass
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

    # Step 2: YOLO推理
    step_start = time.time()
    # 自动选择最佳设备进行推理（MPS/CUDA/CPU）
    from utils import get_best_device
    
    # 获取模型保存的设备，或自动选择
    device = getattr(model, '_device', None) or get_best_device('auto')
    
    # 尝试使用首选设备
    try:
        results = model(image, device=device)
    except Exception as device_error:
        # 首选设备失败，尝试降级
        log_message(f"⚠️  {device.upper()} 推理失败，尝试降级: {device_error}", dir)
        
        # 降级策略：MPS -> CUDA -> CPU
        fallback_devices = []
        if device == 'mps':
            fallback_devices = ['cuda', 'cpu']
        elif device == 'cuda':
            fallback_devices = ['cpu']
        else:
            fallback_devices = []
        
        success = False
        for fallback_device in fallback_devices:
            try:
                import torch
                if fallback_device == 'cuda' and not torch.cuda.is_available():
                    continue
                if fallback_device == 'mps' and not torch.backends.mps.is_available():
                    continue
                    
                log_message(f"🔄 尝试使用 {fallback_device.upper()} 设备...", dir)
                results = model(image, device=fallback_device)
                success = True
                device = fallback_device
                break
            except Exception:
                continue
        
        if not success:
            # 所有设备都失败，使用 CPU 作为最后尝试
            try:
                log_message(f"🔄 最后尝试使用 CPU 设备...", dir)
                results = model(image, device='cpu')
                device = 'cpu'
            except Exception as cpu_error:
                log_message(f"❌ AI推理完全失败: {cpu_error}", dir)
            # 返回"无鸟"结果（V3.1）
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
            # 清理临时 JPG 文件（如果创建了）
            if 'temp_jpg_path' in locals() and temp_jpg_path and os.path.exists(temp_jpg_path):
                try:
                    os.remove(temp_jpg_path)
                except Exception:
                    pass
            return found_bird, bird_result, 0.0, 0.0, None, None, None, None  # V3.7: 8 values including mask

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

    # 清理临时 JPG 文件（如果创建了）
    if 'temp_jpg_path' in locals() and temp_jpg_path and os.path.exists(temp_jpg_path):
        try:
            os.remove(temp_jpg_path)
        except Exception:
            pass  # 忽略清理错误
    
    return found_bird, bird_result, bird_confidence, bird_sharpness, nima_score, bird_bbox, img_dims, bird_mask