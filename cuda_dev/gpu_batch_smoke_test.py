# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
if repo_root and repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ai_model import load_yolo_model, detect_and_draw_birds_batch
from core.config_manager import UISettings
from config import config


def _choose_device(requested: str) -> str:
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _iter_images(root: str, recursive: bool):
    if os.path.isfile(root):
        yield root
        return
    if recursive:
        for base, _, files in os.walk(root):
            for name in files:
                if config.is_jpg_file(name):
                    yield os.path.join(base, name)
    else:
        for name in os.listdir(root):
            if config.is_jpg_file(name):
                yield os.path.join(root, name)


def _collect_images(path: str, count: int, recursive: bool):
    images = []
    for item in _iter_images(path, recursive):
        images.append(item)
        if len(images) >= count:
            break
    return images


def main():
    parser = argparse.ArgumentParser(description="GPU batch smoke test for YOLO inference.")
    parser.add_argument("path", nargs="?", default=".", help="Image directory or a single jpg path.")
    parser.add_argument("--count", type=int, default=2, help="Number of images to use per batch.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat batch inference N times.")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively.")
    parser.add_argument("--device", default="", help="Device override: cuda/mps/cpu.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations (single image).")
    args = parser.parse_args()

    if args.count < 1:
        print("count must be >= 1")
        return 2
    if args.repeat < 1:
        print("repeat must be >= 1")
        return 2
    if args.warmup < 0:
        print("warmup must be >= 0")
        return 2

    images = _collect_images(args.path, args.count, args.recursive)
    if len(images) < args.count:
        print(f"Not enough jpg files found. need={args.count}, got={len(images)}")
        return 2

    device = _choose_device(args.device)
    print(f"[SmokeTest] device={device} count={args.count} repeat={args.repeat} warmup={args.warmup}")
    for idx, img in enumerate(images, 1):
        print(f"[SmokeTest] image{idx}={img}")

    try:
        model = load_yolo_model(device=device)
    except TypeError:
        model = load_yolo_model()

    ui_settings = UISettings(
        ai_confidence=50,
        sharpness_threshold=400,
        nima_threshold=5.0,
        normalization_mode="log_compression",
    )

    for _ in range(args.warmup):
        detect_and_draw_birds_batch(
            [images[0]],
            model,
            os.path.dirname(images[0]) or ".",
            ui_settings,
            None,
            skip_nima=True,
            device=device,
        )

    for i in range(args.repeat):
        start = time.time()
        results = detect_and_draw_birds_batch(
            images,
            model,
            os.path.dirname(images[0]) or ".",
            ui_settings,
            None,
            skip_nima=True,
            device=device,
        )
        elapsed = time.time() - start
        ok = sum(1 for item in results if item is not None)
        print(f"[SmokeTest] iter={i+1} batch={len(images)} ok={ok} elapsed={elapsed:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
