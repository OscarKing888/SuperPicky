#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成网站对比图 - 展示鸟类检测 + 头部区域标注
输出: 原图、裁剪图、叠加掩码图
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, '/Users/jameszhenyu/PycharmProjects/SuperPicky_SandBox')

def convert_raw_to_jpg(raw_path):
    """将RAW文件转换为JPG"""
    import rawpy
    import imageio
    
    jpg_path = raw_path.rsplit('.', 1)[0] + '_preview.jpg'
    if not os.path.exists(jpg_path):
        with rawpy.imread(raw_path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=True)
        imageio.imwrite(jpg_path, rgb)
    return jpg_path

def main():
    # 输入和输出路径
    raw_path = "/Users/jameszhenyu/Desktop/flytest/3星_优选/_Z9W7150.NEF"
    output_dir = "/Users/jameszhenyu/PycharmProjects/SuperPicky_SandBox/docs/img"
    
    print("=" * 60)
    print("生成网站对比图")
    print("=" * 60)
    
    # 1. 转换RAW到JPG
    print("\n[1/5] 转换 RAW 到 JPG...")
    jpg_path = convert_raw_to_jpg(raw_path)
    print(f"   -> {jpg_path}")
    
    # 2. 加载YOLO分割模型
    print("\n[2/5] 加载 YOLO-SEG 模型...")
    from ultralytics import YOLO
    seg_model = YOLO('yolo11m-seg.pt')
    
    # 3. 运行分割检测
    print("\n[3/5] 运行分割检测...")
    img = cv2.imread(jpg_path)
    results = seg_model(img, classes=[14], conf=0.25)  # 14 = bird class
    
    if not results or len(results[0].boxes) == 0:
        print("   ❌ 未检测到鸟类!")
        return
    
    # 获取最大的检测框
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in boxes]
    best_idx = np.argmax(areas)
    best_box = boxes[best_idx].astype(int)
    
    print(f"   -> 检测到 {len(boxes)} 只鸟, 选择最大的一只")
    
    # 4. 提取分割掩码
    print("\n[4/5] 生成掩码叠加图...")
    h, w = img.shape[:2]
    
    # 创建叠加层
    overlay = img.copy()
    
    # 绿色半透明身体掩码
    if results[0].masks is not None:
        mask = results[0].masks.data[best_idx].cpu().numpy()
        mask_resized = cv2.resize(mask, (w, h))
        bird_mask = (mask_resized > 0.5).astype(np.uint8)
        
        # 绿色叠加
        green_overlay = np.zeros_like(img)
        green_overlay[:, :] = [0, 200, 0]  # BGR: 绿色
        overlay = np.where(bird_mask[:, :, None] == 1,
                          cv2.addWeighted(img, 0.6, green_overlay, 0.4, 0),
                          overlay)
    
    # 5. 计算并标注头部区域（红色圆形 + seg掩码交集）
    print("\n[5/6] 加载关键点检测模型...")
    from core.keypoint_detector import get_keypoint_detector
    
    x1, y1, x2, y2 = best_box
    
    # 裁剪鸟区域用于关键点检测
    bird_crop = img[y1:y2, x1:x2]
    bird_crop_rgb = cv2.cvtColor(bird_crop, cv2.COLOR_BGR2RGB)
    
    # 准备 seg 掩码（裁剪到鸟区域）
    crop_seg_mask = None
    if results[0].masks is not None:
        mask = results[0].masks.data[best_idx].cpu().numpy()
        mask_resized = cv2.resize(mask, (w, h))
        bird_mask_full = (mask_resized > 0.5).astype(np.uint8) * 255
        crop_seg_mask = bird_mask_full[y1:y2, x1:x2]
    
    # 检测关键点
    detector = get_keypoint_detector()
    kp_result = detector.detect(bird_crop_rgb, box=(x1, y1, x2-x1, y2-y1), seg_mask=crop_seg_mask)
    
    print("\n[6/6] 生成掩码叠加图...")
    
    # 在原图大小创建掩码
    head_mask_full = np.zeros((h, w), dtype=np.uint8)
    
    if kp_result and kp_result.visible_eye:
        crop_h, crop_w = bird_crop.shape[:2]
        
        # 获取眼睛和喙的位置
        left_eye = kp_result.left_eye
        right_eye = kp_result.right_eye
        beak = kp_result.beak
        
        # 选择可见的眼睛
        if kp_result.visible_eye == 'both':
            # 选更远离喙的眼睛
            left_dist = np.sqrt((left_eye[0]-beak[0])**2 + (left_eye[1]-beak[1])**2)
            right_dist = np.sqrt((right_eye[0]-beak[0])**2 + (right_eye[1]-beak[1])**2)
            eye = left_eye if left_dist >= right_dist else right_eye
        elif kp_result.visible_eye == 'left':
            eye = left_eye
        else:
            eye = right_eye
        
        # 转为像素坐标
        eye_px = (int(eye[0] * crop_w), int(eye[1] * crop_h))
        beak_px = (int(beak[0] * crop_w), int(beak[1] * crop_h))
        
        # 计算半径 (眼喙距离 x 1.2)
        radius = int(np.sqrt((eye_px[0]-beak_px[0])**2 + (eye_px[1]-beak_px[1])**2) * 1.2)
        radius = max(20, min(radius, min(crop_w, crop_h) // 2))
        
        # 在裁剪区域内创建圆形掩码
        circle_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cv2.circle(circle_mask, eye_px, radius, 255, -1)
        
        # 与 seg 掩码取交集
        if crop_seg_mask is not None:
            head_mask_crop = cv2.bitwise_and(circle_mask, crop_seg_mask)
        else:
            head_mask_crop = circle_mask
        
        # 放回原图位置
        head_mask_full[y1:y2, x1:x2] = head_mask_crop
        
        # 眼睛位置（用于标注）
        eye_full = (x1 + eye_px[0], y1 + eye_px[1])
        beak_full = (x1 + beak_px[0], y1 + beak_px[1])
    
    # 创建叠加层
    overlay = img.copy()
    
    # 绿色半透明身体掩码
    if results[0].masks is not None:
        green_overlay = np.zeros_like(img)
        green_overlay[:, :] = [0, 200, 0]  # BGR: 绿色
        overlay = np.where(bird_mask_full[:, :, None] > 0,
                          cv2.addWeighted(img, 0.6, green_overlay, 0.4, 0),
                          overlay)
    
    # 红色半透明头部区域（圆形 + seg交集）
    if np.any(head_mask_full > 0):
        red_layer = np.zeros_like(img)
        red_layer[:, :] = [0, 0, 255]  # BGR: 红色
        overlay = np.where(head_mask_full[:, :, None] > 0,
                          cv2.addWeighted(overlay, 0.5, red_layer, 0.5, 0),
                          overlay)
    
    # 添加边框和标注
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)  # 黄色检测框
    
    if kp_result and kp_result.visible_eye:
        # 标注眼睛和喙
        cv2.circle(overlay, eye_full, 8, (255, 255, 0), -1)  # 青色眼睛
        cv2.circle(overlay, beak_full, 6, (255, 0, 255), -1)  # 紫色喙
    
    # 裁剪区域（扩展15%边距）
    box_w = x2 - x1
    box_h = y2 - y1
    margin = 0.15
    crop_x1 = max(0, int(x1 - box_w * margin))
    crop_y1 = max(0, int(y1 - box_h * margin))
    crop_x2 = min(w, int(x2 + box_w * margin))
    crop_y2 = min(h, int(y2 + box_h * margin))
    
    # 裁剪原图和叠加图
    original_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
    overlay_crop = overlay[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # 调整大小为固定宽度
    target_width = 600
    aspect = original_crop.shape[0] / original_crop.shape[1]
    target_height = int(target_width * aspect)
    
    original_resized = cv2.resize(original_crop, (target_width, target_height))
    overlay_resized = cv2.resize(overlay_crop, (target_width, target_height))
    
    # 拼接左右对比图
    comparison = np.hstack([original_resized, overlay_resized])
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (20, 40), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "AI Detection", (target_width + 20, 40), font, 1, (255, 255, 255), 2)
    
    # 图例
    cv2.putText(comparison, "Green: Body", (target_width + 20, target_height - 60), font, 0.6, (0, 200, 0), 2)
    cv2.putText(comparison, "Red: Head (Sharpness)", (target_width + 20, target_height - 30), font, 0.6, (0, 0, 255), 2)
    
    # 保存
    print("\n[5/5] 保存输出图片...")
    
    comparison_path = os.path.join(output_dir, "comparison.jpg")
    cv2.imwrite(comparison_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"   -> 对比图: {comparison_path}")
    
    # 单独保存叠加图
    overlay_path = os.path.join(output_dir, "detection_overlay.jpg")
    cv2.imwrite(overlay_path, overlay_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"   -> 叠加图: {overlay_path}")
    
    print("\n" + "=" * 60)
    print("✅ 完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
