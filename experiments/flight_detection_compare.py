#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞鸟检测对比测试 - EfficientNet-B3 vs Florence-2

对比现有 EfficientNet-B3 飞鸟检测模型和 Florence-2 的准确率。
支持从 RAW 文件（NEF, CR2, ARW 等）中提取内嵌预览进行检测。

使用方法:
    # 从 RAW 目录测试（自动提取预览）
    python experiments/flight_detection_compare.py --test-dir /path/to/raw/photos --limit 50

    # 使用已有 JPG 测试
    python experiments/flight_detection_compare.py --test-dir /path/to/jpgs --use-jpg

    # 带 ground truth 测试（需要 flying/not_flying 子目录）
    python experiments/flight_detection_compare.py --test-dir /path/to/labeled --labeled
"""

import argparse
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from core.flight_detector import FlightDetector, FlightResult


# RAW 文件扩展名
RAW_EXTENSIONS = {'.nef', '.cr2', '.cr3', '.arw', '.raf', '.orf', '.rw2', '.pef', '.dng', '.3fr', '.iiq'}

# Florence-2 飞行相关关键词
FLYING_KEYWORDS = [
    'flying', 'flight', 'soaring', 'gliding', 'hovering',
    'in the air', 'mid-air', 'airborne', 'taking off',
    'wings spread', 'wings extended', 'wings outstretched',
]


@dataclass
class CompareResult:
    """单张图片的对比结果"""
    raw_path: str           # 原始文件路径
    preview_path: str       # 预览图路径（可能是临时文件）
    ground_truth: Optional[bool] = None  # 实际是否飞行（如果有标注）

    # EfficientNet-B3 结果
    efficientnet_pred: bool = False
    efficientnet_conf: float = 0.0
    efficientnet_time: float = 0.0

    # Florence-2 Caption 方法结果
    florence_caption_pred: bool = False
    florence_caption_text: str = ""
    florence_caption_time: float = 0.0

    # Florence-2 Grounding 方法结果（可选）
    florence_grounding_pred: Optional[bool] = None
    florence_grounding_result: Optional[str] = None
    florence_grounding_time: Optional[float] = None

    @property
    def models_agree(self) -> bool:
        """两个模型结论是否一致"""
        return self.efficientnet_pred == self.florence_caption_pred


class RawPreviewExtractor:
    """从 RAW 文件提取内嵌预览图"""

    def __init__(self):
        self.exiftool_path = self._find_exiftool()

    def _find_exiftool(self) -> str:
        """查找 exiftool 路径"""
        # 优先使用系统 exiftool
        system_exiftool = shutil.which('exiftool')
        if system_exiftool:
            return system_exiftool

        # 回退到项目目录
        project_exiftool = project_root / 'exiftool'
        if project_exiftool.exists():
            return str(project_exiftool)

        raise RuntimeError("未找到 exiftool，请安装或确保项目目录下有 exiftool")

    def extract_preview(self, raw_path: str, output_path: str) -> bool:
        """
        从 RAW 文件提取内嵌的 JPG 预览

        Args:
            raw_path: RAW 文件路径
            output_path: 输出 JPG 路径

        Returns:
            是否成功
        """
        try:
            # 使用 exiftool 提取预览图
            # -b: 二进制输出
            # -PreviewImage: 提取预览图（大多数相机）
            # -JpgFromRaw: 备选（部分相机使用此标签）
            cmd = [
                self.exiftool_path,
                '-b',
                '-PreviewImage',
                raw_path
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode == 0 and len(result.stdout) > 1000:
                with open(output_path, 'wb') as f:
                    f.write(result.stdout)
                return True

            # 尝试 JpgFromRaw
            cmd[2] = '-JpgFromRaw'
            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode == 0 and len(result.stdout) > 1000:
                with open(output_path, 'wb') as f:
                    f.write(result.stdout)
                return True

            return False

        except Exception as e:
            print(f"提取预览失败 {raw_path}: {e}")
            return False


class Florence2FlightDetector:
    """
    使用 Florence-2 检测飞鸟

    方法1: DETAILED_CAPTION - 生成图像描述，分析是否包含飞行关键词
    方法2: CAPTION_TO_PHRASE_GROUNDING - 使用 "flying bird" 作为 prompt 定位
    """

    def __init__(self, model_name: str = "microsoft/Florence-2-base"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = None
        self.model_loaded = False

    def load_model(self):
        """加载 Florence-2 模型"""
        print(f"正在加载 Florence-2 模型: {self.model_name}")

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"使用设备: {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
            attn_implementation="eager"  # 禁用 SDPA 以兼容旧模型
        ).to(self.device)

        self.model_loaded = True
        print("Florence-2 模型加载完成")

    def _run_task(self, image: Image.Image, task_prompt: str, text_input: str = None) -> dict:
        """运行 Florence-2 任务"""
        if not self.model_loaded:
            raise RuntimeError("Florence-2 模型未加载")

        prompt = task_prompt if text_input is None else task_prompt + text_input

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
                early_stopping=True
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        return parsed

    def detect_by_caption(self, image_path: str) -> tuple:
        """通过生成详细描述检测飞鸟"""
        image = Image.open(image_path).convert('RGB')
        result = self._run_task(image, "<DETAILED_CAPTION>")
        caption = result.get("<DETAILED_CAPTION>", "")

        caption_lower = caption.lower()
        is_flying = any(keyword in caption_lower for keyword in FLYING_KEYWORDS)

        return is_flying, caption

    def detect_by_grounding(self, image_path: str) -> tuple:
        """通过短语定位检测飞鸟"""
        image = Image.open(image_path).convert('RGB')
        result = self._run_task(image, "<CAPTION_TO_PHRASE_GROUNDING>", "flying bird")
        grounding_result = result.get("<CAPTION_TO_PHRASE_GROUNDING>", {})

        bboxes = grounding_result.get("bboxes", [])
        labels = grounding_result.get("labels", [])

        is_flying = len(bboxes) > 0 and any("flying" in l.lower() for l in labels)
        result_str = f"bboxes: {bboxes}, labels: {labels}"

        return is_flying, result_str


def find_test_files(test_dir: Path, use_jpg: bool, labeled: bool, limit: int) -> list:
    """
    查找测试文件

    Args:
        test_dir: 测试目录
        use_jpg: 是否使用已有的 JPG 文件
        labeled: 是否使用带标注的目录结构
        limit: 最大文件数

    Returns:
        list of (file_path, ground_truth)
    """
    files = []

    if labeled:
        # 带标注模式：flying/ 和 not_flying/ 子目录
        flying_dir = test_dir / "flying"
        not_flying_dir = test_dir / "not_flying"

        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            if flying_dir.exists():
                for f in flying_dir.glob(ext):
                    files.append((str(f), True))
            if not_flying_dir.exists():
                for f in not_flying_dir.glob(ext):
                    files.append((str(f), False))
    else:
        # 无标注模式：直接扫描目录
        if use_jpg:
            # 使用 JPG 文件
            for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
                for f in test_dir.glob(ext):
                    files.append((str(f), None))
        else:
            # 使用 RAW 文件
            for f in test_dir.iterdir():
                if f.is_file() and f.suffix.lower() in RAW_EXTENSIONS:
                    files.append((str(f), None))

    # 按文件名排序并限制数量
    files.sort(key=lambda x: Path(x[0]).name)
    if limit > 0:
        files = files[:limit]

    return files


def print_results_table(results: list, show_ground_truth: bool):
    """打印对比结果表格"""
    print("\n" + "=" * 130)
    print("飞鸟检测对比结果")
    print("=" * 130)

    total = len(results)
    efficientnet_flying = sum(1 for r in results if r.efficientnet_pred)
    florence_flying = sum(1 for r in results if r.florence_caption_pred)
    agree_count = sum(1 for r in results if r.models_agree)

    print(f"\n总测试图片数: {total}")
    print(f"\nEfficientNet-B3 检测为飞行: {efficientnet_flying}/{total} ({efficientnet_flying/total*100:.1f}%)")
    print(f"Florence-2 检测为飞行: {florence_flying}/{total} ({florence_flying/total*100:.1f}%)")
    print(f"两模型一致: {agree_count}/{total} ({agree_count/total*100:.1f}%)")

    if show_ground_truth:
        labeled_results = [r for r in results if r.ground_truth is not None]
        if labeled_results:
            eff_correct = sum(1 for r in labeled_results if r.efficientnet_pred == r.ground_truth)
            flor_correct = sum(1 for r in labeled_results if r.florence_caption_pred == r.ground_truth)
            labeled_total = len(labeled_results)
            print(f"\n带标注样本 ({labeled_total} 张):")
            print(f"  EfficientNet-B3 准确率: {eff_correct}/{labeled_total} ({eff_correct/labeled_total*100:.1f}%)")
            print(f"  Florence-2 准确率: {flor_correct}/{labeled_total} ({flor_correct/labeled_total*100:.1f}%)")

    # 详细结果（只显示差异）
    disagreements = [r for r in results if not r.models_agree]
    if disagreements:
        print(f"\n\n{'='*130}")
        print(f"模型结论不一致的图片 ({len(disagreements)} 张)")
        print("-" * 130)
        print(f"{'文件名':<30} {'EffNet':<20} {'Florence-2':<15} {'Florence描述':<60}")
        print("-" * 130)

        for r in disagreements:
            name = Path(r.raw_path).stem[:28]
            eff = f"{'飞行' if r.efficientnet_pred else '非飞行'} ({r.efficientnet_conf:.2f})"
            flor = "飞行" if r.florence_caption_pred else "非飞行"
            desc = r.florence_caption_text[:58] if r.florence_caption_text else ""
            print(f"{name:<30} {eff:<20} {flor:<15} {desc:<60}")

    # 时间统计
    print("\n" + "-" * 130)
    avg_eff_time = sum(r.efficientnet_time for r in results) / total
    avg_flor_time = sum(r.florence_caption_time for r in results) / total

    print(f"平均推理时间:")
    print(f"  EfficientNet-B3: {avg_eff_time*1000:.1f}ms")
    print(f"  Florence-2 Caption: {avg_flor_time*1000:.1f}ms")

    print("=" * 130)


def save_results(results: list, output_file: Path):
    """保存详细结果到文件"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("飞鸟检测对比测试详细结果\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"文件: {r.raw_path}\n")
            if r.ground_truth is not None:
                f.write(f"真值: {'飞行' if r.ground_truth else '非飞行'}\n")
            f.write(f"EfficientNet: {'飞行' if r.efficientnet_pred else '非飞行'} (conf={r.efficientnet_conf:.3f})\n")
            f.write(f"Florence-2: {'飞行' if r.florence_caption_pred else '非飞行'}\n")
            f.write(f"  描述: {r.florence_caption_text}\n")
            f.write(f"一致: {'是' if r.models_agree else '否'}\n")
            f.write("-" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="飞鸟检测对比测试 - EfficientNet-B3 vs Florence-2"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="测试目录路径"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="最大测试图片数（默认50，0表示无限制）"
    )
    parser.add_argument(
        "--use-jpg",
        action="store_true",
        help="使用目录中已有的 JPG 文件（而非从 RAW 提取）"
    )
    parser.add_argument(
        "--labeled",
        action="store_true",
        help="使用带标注的目录结构（flying/ 和 not_flying/ 子目录）"
    )
    parser.add_argument(
        "--florence-model",
        type=str,
        default="microsoft/Florence-2-base",
        choices=["microsoft/Florence-2-base", "microsoft/Florence-2-large"],
        help="Florence-2 模型版本"
    )
    parser.add_argument(
        "--skip-grounding",
        action="store_true",
        help="跳过 Grounding 方法测试（更快）"
    )

    args = parser.parse_args()
    test_dir = Path(args.test_dir)

    if not test_dir.exists():
        print(f"错误: 测试目录不存在: {test_dir}")
        sys.exit(1)

    # 查找测试文件
    print("正在扫描测试文件...")
    test_files = find_test_files(test_dir, args.use_jpg, args.labeled, args.limit)

    if not test_files:
        print("错误: 未找到测试文件")
        sys.exit(1)

    print(f"找到 {len(test_files)} 个测试文件")

    # 初始化 RAW 预览提取器
    extractor = None
    if not args.use_jpg and not args.labeled:
        print("\n初始化 RAW 预览提取器...")
        extractor = RawPreviewExtractor()
        print(f"使用 exiftool: {extractor.exiftool_path}")

    # 加载 EfficientNet-B3 模型
    print("\n正在加载 EfficientNet-B3 飞鸟检测模型...")
    efficientnet = FlightDetector()
    try:
        efficientnet.load_model()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保模型文件存在: models/superFlier_efficientnet.pth")
        sys.exit(1)

    # 加载 Florence-2 模型
    print()
    florence = Florence2FlightDetector(args.florence_model)
    florence.load_model()

    # 创建临时目录存放提取的预览
    temp_dir = None
    if extractor:
        temp_dir = tempfile.mkdtemp(prefix="flight_compare_")
        print(f"\n临时预览目录: {temp_dir}")

    # 运行对比测试
    print(f"\n开始对比测试...")
    results = []

    try:
        for i, (file_path, ground_truth) in enumerate(test_files, 1):
            file_name = Path(file_path).name
            print(f"\r处理 {i}/{len(test_files)}: {file_name[:50]:<50}", end="", flush=True)

            # 确定预览图路径
            if extractor:
                # 从 RAW 提取预览
                preview_path = Path(temp_dir) / f"{Path(file_path).stem}.jpg"
                if not extractor.extract_preview(file_path, str(preview_path)):
                    print(f"\n  跳过（无法提取预览）: {file_name}")
                    continue
                preview_path = str(preview_path)
            else:
                # 直接使用文件
                preview_path = file_path

            # EfficientNet-B3 检测
            start = time.time()
            eff_result = efficientnet.detect(preview_path)
            eff_time = time.time() - start

            # Florence-2 Caption 检测
            start = time.time()
            flor_caption_pred, flor_caption_text = florence.detect_by_caption(preview_path)
            flor_caption_time = time.time() - start

            results.append(CompareResult(
                raw_path=file_path,
                preview_path=preview_path,
                ground_truth=ground_truth,
                efficientnet_pred=eff_result.is_flying,
                efficientnet_conf=eff_result.confidence,
                efficientnet_time=eff_time,
                florence_caption_pred=flor_caption_pred,
                florence_caption_text=flor_caption_text,
                florence_caption_time=flor_caption_time,
            ))

        print()  # 换行

        if not results:
            print("错误: 没有成功处理任何图片")
            sys.exit(1)

        # 输出结果
        print_results_table(results, args.labeled)

        # 保存详细结果
        output_file = test_dir / "flight_compare_results.txt"
        save_results(results, output_file)
        print(f"\n详细结果已保存到: {output_file}")

    finally:
        # 清理临时目录
        if temp_dir:
            print(f"\n清理临时目录: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
