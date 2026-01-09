"""
工具函数模块
提供日志记录和CSV报告功能
"""
import os
import csv
import numpy as np
from datetime import datetime


def log_message(message: str, directory: str = None, file_only: bool = False):
    """
    记录日志消息到控制台和日志文件

    Args:
        message: 日志消息
        directory: 工作目录（可选，如果提供则写入该目录/.superpicky/process_log.txt）
        file_only: 仅写入文件，不打印到控制台（避免重复输出）
    """
    # 打印到控制台（除非指定只写文件）
    if not file_only:
        print(message)

    # 如果提供了目录，写入日志文件到_tmp子目录
    if directory:
        # 确保_tmp目录存在
        tmp_dir = os.path.join(directory, ".superpicky")
        os.makedirs(tmp_dir, exist_ok=True)

        log_file = os.path.join(tmp_dir, "process_log.txt")
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")


def write_to_csv(data: dict, directory: str, header: bool = False):
    """
    将数据写入CSV报告文件

    Args:
        data: 要写入的数据字典（如果为None且header=True，则只创建文件并写表头）
        directory: 工作目录
        header: 是否写入表头（第一次写入时为True）
    """
    # 确保_tmp目录存在
    tmp_dir = os.path.join(directory, ".superpicky")
    os.makedirs(tmp_dir, exist_ok=True)

    report_file = os.path.join(tmp_dir, "report.csv")

    # V3.4: 全英文列名，添加飞版检测字段
    fieldnames = [
        "filename",        # 文件名（不含扩展名）
        "has_bird",        # 是否有鸟 (yes/no)
        "confidence",      # AI置信度 (0-1)
        "head_sharp",      # 头部区域锐度
        "left_eye",        # 左眼可见性 (0-1)
        "right_eye",       # 右眼可见性 (0-1)
        "beak",            # 喙可见性 (0-1)
        "nima_score",      # NIMA美学评分 (0-10)
        "is_flying",       # V3.4: 是否飞行 (yes/no/-)
        "flight_conf",     # V3.4: 飞行置信度 (0-1)
        "rating"           # 最终评分 (-1/0/1/2/3)
    ]

    try:
        # 如果是初始化表头（data为None）
        if data is None and header:
            with open(report_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            return

        file_exists = os.path.exists(report_file)
        mode = 'a' if file_exists else 'w'

        with open(report_file, mode, newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # 如果文件不存在或者明确要求写表头，则写入表头
            if not file_exists or header:
                writer.writeheader()

            if data:
                writer.writerow(data)
    except Exception as e:
        log_message(f"Warning: Could not write to CSV file: {e}", directory)


def get_best_device(preferred_device='auto'):
    """
    获取最佳计算设备（自动选择或使用首选设备）
    
    Args:
        preferred_device: 首选设备 ('auto', 'mps', 'cuda', 'cpu')
                         'auto' 会自动选择最佳可用设备
        
    Returns:
        str: 设备名称 ('mps', 'cuda', 'cpu')
    """
    try:
        import torch
        
        # 如果指定了设备，直接返回（如果可用）
        if preferred_device == 'mps':
            if torch.backends.mps.is_available():
                return 'mps'
            else:
                # MPS 不可用，尝试其他设备
                preferred_device = 'auto'
        
        if preferred_device == 'cuda':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                # CUDA 不可用，尝试其他设备
                preferred_device = 'auto'
        
        # 自动选择最佳设备（优先级：MPS > CUDA > CPU）
        if preferred_device == 'auto':
            # 1. 优先尝试 MPS (Apple GPU)
            if torch.backends.mps.is_available():
                return 'mps'
            
            # 2. 尝试 CUDA (NVIDIA GPU)
            if torch.cuda.is_available():
                return 'cuda'
            
            # 3. 默认使用 CPU
            return 'cpu'
        
        # 如果指定了 CPU 或其他，直接返回
        return preferred_device
        
    except ImportError:
        # 如果没有安装 torch，返回 CPU
        return 'cpu'
    except Exception as e:
        # 任何其他错误，返回 CPU
        return 'cpu'


def read_image(image_path):
    """
    读取图片文件，支持 JPG、PNG、HEIF、HEIC 等格式
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        numpy.ndarray: BGR 格式的图像数组（OpenCV 格式），如果读取失败返回 None
    """
    import cv2
    
    # 检查文件扩展名
    ext = os.path.splitext(image_path)[1].lower()
    
    # 对于 HEIF/HEIC 文件，使用 PIL + pillow-heif 读取
    if ext in ['.heif', '.heic', '.hif']:
        try:
            # 注册 pillow-heif（如果还没有注册）
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            except ImportError:
                pass  # pillow-heif 可能已经注册或未安装
            
            # 使用 PIL 读取 HEIF/HEIC
            from PIL import Image
            pil_image = Image.open(image_path).convert('RGB')
            
            # 转换为 numpy 数组（RGB）
            img_array = np.array(pil_image)
            
            # 转换为 BGR（OpenCV 格式）
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_bgr
        except Exception as e:
            # 如果读取失败，尝试用 OpenCV 读取（某些系统可能支持）
            pass
    
    # 对于其他格式（JPG、PNG 等），使用 OpenCV 读取
    img = cv2.imread(image_path)
    return img
