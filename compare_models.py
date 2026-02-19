#!/usr/bin/env python3
"""
OSEA 模型对比测试工具
对比 birdid2024 (TorchScript) 与 OSEA (ResNet34) 的鸟类识别结果
"""

import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

# 项目路径
PROJECT_ROOT = Path(__file__).parent
BIRDID_DIR = PROJECT_ROOT / "birdid"

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

# OSEA 预处理 (RGB, torchvision transforms)
osea_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def parse_args():
    parser = argparse.ArgumentParser(
        description="对比 birdid2024 与 OSEA 模型的识别结果"
    )
    parser.add_argument("input", help="图片路径或目录")
    parser.add_argument("--top-k", type=int, default=5, help="显示前 K 个结果")
    return parser.parse_args()


# ==================== 数据加载 ====================

def load_bird_info():
    """加载 birdinfo.json (中文名、英文名、学名)"""
    info_path = BIRDID_DIR / "data" / "birdinfo.json"
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_osea_labels():
    """加载 OSEA 标签文件 (学名)"""
    label_path = BIRDID_DIR / "data" / "osea_labels.txt"
    with open(label_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


# ==================== 模型加载 ====================

def decrypt_model(encrypted_path: str, password: str) -> bytes:
    """解密模型文件"""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    with open(encrypted_path, "rb") as f:
        encrypted_data = f.read()

    salt = encrypted_data[:16]
    iv = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    key = kdf.derive(password.encode())

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext_padded = decryptor.update(ciphertext) + decryptor.finalize()

    padding_length = plaintext_padded[-1]
    return plaintext_padded[:-padding_length]


def load_birdid_model():
    """加载 birdid2024 TorchScript 模型"""
    model_path = BIRDID_DIR / "models" / "birdid2024.pt.enc"
    password = "SuperBirdID_2024_AI_Model_Encryption_Key_v1"

    print("加载 birdid2024 模型...")
    model_data = decrypt_model(str(model_path), password)
    model = torch.jit.load(io.BytesIO(model_data), map_location="cpu")
    model.to_eval = lambda: None  # placeholder
    print("  birdid2024 加载完成")
    return model


def load_osea_model():
    """加载 OSEA ResNet34 模型"""
    model_path = PROJECT_ROOT / "models" / "model20240824.pth"

    print("加载 OSEA ResNet34 模型...")
    model = models.resnet34(num_classes=11000)
    state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()  # 切换到推理模式
    print("  OSEA 加载完成")
    return model


# ==================== 图像预处理 ====================

def preprocess_for_birdid(image: Image.Image) -> torch.Tensor:
    """birdid2024 预处理 (BGR 颜色空间)"""
    # 调整尺寸
    image = image.resize((224, 224), Image.LANCZOS)

    # RGB -> BGR
    img_array = np.array(image)
    bgr_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 正规化 (BGR 顺序)
    mean = np.array([0.406, 0.456, 0.485])
    std = np.array([0.225, 0.224, 0.229])
    normalized = (bgr_array / 255.0 - mean) / std

    # 转换为 tensor [1, 3, 224, 224]
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def preprocess_for_osea(image: Image.Image) -> torch.Tensor:
    """OSEA 预处理 (RGB, torchvision transforms)"""
    return osea_transform(image).unsqueeze(0)


# ==================== 推理 ====================

def predict_with_birdid(model, image: Image.Image, bird_info: list, top_k: int = 5):
    """使用 birdid2024 模型预测"""
    input_tensor = preprocess_for_birdid(image)

    with torch.no_grad():
        output = model(input_tensor)[0]

    # softmax
    probs = torch.nn.functional.softmax(output, dim=0)
    top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for i in range(top_k):
        idx = top_indices[i].item()
        prob = top_probs[i].item() * 100
        cn_name = bird_info[idx][0] if idx < len(bird_info) else f"ID:{idx}"
        en_name = bird_info[idx][1] if idx < len(bird_info) else f"ID:{idx}"
        results.append({
            "rank": i + 1,
            "cn_name": cn_name,
            "en_name": en_name,
            "confidence": prob,
        })
    return results


def predict_with_osea(model, image: Image.Image, bird_info: list, top_k: int = 5):
    """使用 OSEA 模型预测"""
    input_tensor = preprocess_for_osea(image)

    with torch.no_grad():
        output = model(input_tensor)  # ResNet34 直接返回 [batch, num_classes]

    # 取 batch 维度的第一个
    output = output[0]

    # OSEA 模型输出 11000 维，但有效标签只有 10964 个
    # 只取前 10964 维进行 softmax
    num_valid_classes = len(bird_info)  # 10964
    output = output[:num_valid_classes]

    # softmax
    probs = torch.nn.functional.softmax(output, dim=0)
    top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for i in range(top_k):
        idx = top_indices[i].item()
        prob = top_probs[i].item() * 100
        cn_name = bird_info[idx][0] if idx < len(bird_info) else f"ID:{idx}"
        en_name = bird_info[idx][1] if idx < len(bird_info) else f"ID:{idx}"
        results.append({
            "rank": i + 1,
            "cn_name": cn_name,
            "en_name": en_name,
            "confidence": prob,
        })
    return results


# ==================== 显示 ====================

def display_comparison(image_name: str, birdid_results: list, osea_results: list):
    """显示对比结果表格"""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()

        table = Table(title=f"[bold]{image_name}[/bold]", show_header=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("birdid2024", style="cyan", width=28)
        table.add_column("置信度", justify="right", width=8)
        table.add_column("OSEA", style="green", width=28)
        table.add_column("置信度", justify="right", width=8)
        table.add_column("匹配", justify="center", width=4)

        for i in range(len(birdid_results)):
            b = birdid_results[i]
            o = osea_results[i]
            match_symbol = "Y" if b["cn_name"] == o["cn_name"] else "N"
            match_style = "green" if match_symbol == "Y" else "red"

            table.add_row(
                str(i + 1),
                b["cn_name"],
                f"{b['confidence']:.1f}%",
                o["cn_name"],
                f"{o['confidence']:.1f}%",
                f"[{match_style}]{match_symbol}[/{match_style}]",
            )

        top1_match = birdid_results[0]["cn_name"] == osea_results[0]["cn_name"]
        status = "[green]Y Top-1 一致[/green]" if top1_match else "[red]N Top-1 不同[/red]"

        console.print(table)
        console.print(status)
        console.print()
        return top1_match

    except ImportError:
        # fallback 到简单输出
        print(f"\n=== {image_name} ===")
        print(f"{'#':<3} {'birdid2024':<25} {'置信度':<10} {'OSEA':<25} {'置信度':<10} {'匹配':<4}")
        print("-" * 85)
        for i in range(len(birdid_results)):
            b = birdid_results[i]
            o = osea_results[i]
            match_symbol = "Y" if b["cn_name"] == o["cn_name"] else "N"
            print(f"{i+1:<3} {b['cn_name']:<25} {b['confidence']:>6.1f}%   {o['cn_name']:<25} {o['confidence']:>6.1f}%   {match_symbol}")

        top1_match = birdid_results[0]["cn_name"] == osea_results[0]["cn_name"]
        print(f"\n{'Y Top-1 一致' if top1_match else 'N Top-1 不同'}\n")
        return top1_match


# ==================== 图片加载 ====================

def get_image_files(input_path: str) -> list:
    """获取图片文件列表"""
    path = Path(input_path)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif",
                  ".arw", ".cr2", ".cr3", ".nef", ".raf", ".rw2", ".dng"}

    if path.is_file():
        return [path] if path.suffix.lower() in extensions else []
    elif path.is_dir():
        files = []
        for ext in extensions:
            files.extend(path.glob(f"*{ext}"))
            files.extend(path.glob(f"*{ext.upper()}"))
        return sorted(set(files))
    return []


def load_image(image_path: Path) -> Image.Image:
    """加载图片，支持 RAW 格式"""
    raw_extensions = {".arw", ".cr2", ".cr3", ".nef", ".raf", ".rw2", ".dng"}

    if image_path.suffix.lower() in raw_extensions:
        try:
            import rawpy
            with rawpy.imread(str(image_path)) as raw:
                rgb = raw.postprocess()
            return Image.fromarray(rgb)
        except ImportError:
            print(f"[警告] 需要 rawpy 来处理 RAW 文件: {image_path.name}")
            return None
    else:
        return Image.open(image_path).convert("RGB")


# ==================== 主流程 ====================

def main():
    args = parse_args()

    # 获取图片列表
    image_files = get_image_files(args.input)
    if not image_files:
        print(f"[错误] 未找到图片: {args.input}")
        sys.exit(1)

    print(f"找到 {len(image_files)} 张图片\n")

    # 加载数据
    bird_info = load_bird_info()
    osea_labels = load_osea_labels()
    print(f"物种数: SuperPicky={len(bird_info)}, OSEA={len(osea_labels)}\n")

    # 加载模型
    birdid_model = load_birdid_model()
    osea_model = load_osea_model()
    print()

    # 统计
    total = len(image_files)
    top1_match_count = 0
    processed = 0

    # 逐张处理
    for image_path in image_files:
        try:
            image = load_image(image_path)
            if image is None:
                continue

            birdid_results = predict_with_birdid(
                birdid_model, image, bird_info, args.top_k
            )
            osea_results = predict_with_osea(
                osea_model, image, bird_info, args.top_k
            )

            if display_comparison(image_path.name, birdid_results, osea_results):
                top1_match_count += 1
            processed += 1

        except Exception as e:
            print(f"[错误] 处理失败 {image_path.name}: {e}")

    # 汇总
    if processed > 1:
        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            console.print(Panel(
                f"[bold]Top-1 一致率: {top1_match_count}/{processed} ({top1_match_count/processed*100:.1f}%)[/bold]",
                title="汇总",
            ))
        except ImportError:
            print(f"\n=== 汇总 ===")
            print(f"Top-1 一致率: {top1_match_count}/{processed} ({top1_match_count/processed*100:.1f}%)")


if __name__ == "__main__":
    main()
