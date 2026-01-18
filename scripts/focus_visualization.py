#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focus Point Visualization Test - 对焦点可视化测试
在一张图片上同时显示：
  🟢 绿色 - 鸟身体 SEG 掩码边缘
  🔵 蓝色圆圈 - 头部区域（眼睛 + 鸟喙）
  🔴 红色十字 - 对焦点位置
"""

import os
import sys
import cv2
import numpy as np

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from ai_model import load_yolo_model, detect_and_draw_birds
from core.keypoint_detector import get_keypoint_detector
from core.focus_point_detector import get_focus_detector


def visualize_focus_test(nef_path: str, output_path: str = None):
    """
    可视化对焦点测试
    
    Args:
        nef_path: NEF 文件路径
        output_path: 输出 JPG 路径（默认为 focus_visualization.jpg）
    """
    if output_path is None:
        output_path = os.path.join(project_root, "focus_visualization.jpg")
    
    # 先用 sips 转换 NEF 到临时 JPG
    import subprocess
    import tempfile
    
    temp_jpg = tempfile.mktemp(suffix='.jpg')
    print(f"🔄 转换 NEF 到 JPG...")
    result = subprocess.run(['sips', '-s', 'format', 'jpeg', nef_path, '--out', temp_jpg],
                           capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ sips 转换失败: {result.stderr}")
        return
    
    # 读取图片
    img_bgr = cv2.imread(temp_jpg)
    if img_bgr is None:
        print("❌ 无法读取图片")
        return
    
    h_orig, w_orig = img_bgr.shape[:2]
    print(f"📷 图像尺寸: {w_orig}x{h_orig}")
    
    # ============================================
    # 1. YOLO 检测获取 bird_bbox 和 bird_mask
    # ============================================
    print("🤖 运行 YOLO 检测...")
    model = load_yolo_model()
    ui_settings = [50, 400, 5.2, False, 'log_compression']  # 默认设置
    
    result = detect_and_draw_birds(temp_jpg, model, None, project_root, ui_settings, None, skip_nima=True)
    
    if result is None or not result[0]:
        print("❌ 未检测到鸟")
        # os.remove(temp_jpg)
        return
    
    detected, _, confidence, sharpness, _, bird_bbox, img_dims, bird_mask = result
    print(f"✅ 检测到鸟: confidence={confidence:.2f}, bbox={bird_bbox}")
    
    # ============================================
    # 2. 准备裁剪区域和掩码
    # ============================================
    w_resized, h_resized = img_dims
    scale_x = w_orig / w_resized
    scale_y = h_orig / h_resized
    
    # 缩放 bbox 到原图尺寸
    x, y, w, h = bird_bbox
    x_orig = int(x * scale_x)
    y_orig = int(y * scale_y)
    w_orig_box = int(w * scale_x)
    h_orig_box = int(h * scale_y)
    
    # 确保边界有效
    x_orig = max(0, min(x_orig, w_orig - 1))
    y_orig = max(0, min(y_orig, h_orig - 1))
    w_orig_box = min(w_orig_box, w_orig - x_orig)
    h_orig_box = min(h_orig_box, h_orig - y_orig)
    
    # 裁剪鸟的区域
    bird_crop_bgr = img_bgr[y_orig:y_orig+h_orig_box, x_orig:x_orig+w_orig_box]
    
    # 缩放 mask 到原图尺寸
    bird_mask_orig = None
    bird_crop_mask = None
    if bird_mask is not None:
        if bird_mask.shape[:2] != (h_orig, w_orig):
            bird_mask_orig = cv2.resize(bird_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        else:
            bird_mask_orig = bird_mask
        bird_crop_mask = bird_mask_orig[y_orig:y_orig+h_orig_box, x_orig:x_orig+w_orig_box]
    
    # ============================================
    # 3. 关键点检测获取头部区域
    # ============================================
    print("👁️  运行关键点检测...")
    keypoint_detector = get_keypoint_detector()
    keypoint_detector.load_model()
    
    crop_rgb = cv2.cvtColor(bird_crop_bgr, cv2.COLOR_BGR2RGB)
    kp_result = keypoint_detector.detect(
        crop_rgb, 
        box=(x_orig, y_orig, w_orig_box, h_orig_box),
        seg_mask=bird_crop_mask
    )
    
    head_center = None
    head_radius = None
    left_eye_px = None
    right_eye_px = None
    beak_px = None
    
    if kp_result is not None:
        ch, cw = bird_crop_bgr.shape[:2]
        
        # 转换为像素坐标（相对于裁剪区域）
        left_eye_px = (int(kp_result.left_eye[0] * cw), int(kp_result.left_eye[1] * ch))
        right_eye_px = (int(kp_result.right_eye[0] * cw), int(kp_result.right_eye[1] * ch))
        beak_px = (int(kp_result.beak[0] * cw), int(kp_result.beak[1] * ch))
        
        # 选择用于头部圆心的眼睛（用更可见的那只）
        if kp_result.left_eye_vis >= kp_result.right_eye_vis:
            eye_px = left_eye_px
        else:
            eye_px = right_eye_px
        
        # 计算头部半径（眼喙距离 * 1.2）
        if kp_result.beak_vis >= 0.3:
            dist = np.sqrt((eye_px[0] - beak_px[0])**2 + (eye_px[1] - beak_px[1])**2)
            head_radius = int(dist * 1.2)
        else:
            head_radius = int(max(cw, ch) * 0.15)
        
        head_radius = max(20, min(head_radius, min(cw, ch) // 2))
        head_center = eye_px
        
        print(f"✅ 关键点: 左眼vis={kp_result.left_eye_vis:.2f}, 右眼vis={kp_result.right_eye_vis:.2f}, 鸟喙vis={kp_result.beak_vis:.2f}")
    else:
        print("⚠️  关键点检测失败")
    
    # ============================================
    # 4. 对焦点检测
    # ============================================
    print("📍 读取对焦点...")
    focus_detector = get_focus_detector()
    focus_result = focus_detector.detect(nef_path)
    
    focus_px = None
    if focus_result is not None:
        # 对焦点是归一化坐标，转换为原图像素坐标
        focus_px = (int(focus_result.x * w_orig), int(focus_result.y * h_orig))
        print(f"✅ 对焦点: ({focus_result.x:.3f}, {focus_result.y:.3f}) -> ({focus_px[0]}, {focus_px[1]})")
    else:
        print("⚠️  无对焦点数据")
    
    # ============================================
    # 5. 绘制可视化
    # ============================================
    print("🎨 绘制可视化...")
    vis_img = img_bgr.copy()
    
    # --- 🟢 绿色: SEG 掩码边缘 ---
    if bird_mask_orig is not None:
        contours, _ = cv2.findContours(bird_mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 4)  # 绿色轮廓
        print("  🟢 绘制 SEG 掩码边缘（绿色）")
    
    # --- 🔵 蓝色圆圈: 头部区域 ---
    if head_center is not None and head_radius is not None:
        # 转换到原图坐标
        head_center_orig = (head_center[0] + x_orig, head_center[1] + y_orig)
        cv2.circle(vis_img, head_center_orig, head_radius, (255, 100, 0), 4)  # 蓝色圆圈
        
        # 绘制眼睛和鸟喙位置（小圆点）
        if left_eye_px:
            le_orig = (left_eye_px[0] + x_orig, left_eye_px[1] + y_orig)
            cv2.circle(vis_img, le_orig, 8, (255, 255, 0), -1)  # 青色实心 - 左眼
        if right_eye_px:
            re_orig = (right_eye_px[0] + x_orig, right_eye_px[1] + y_orig)
            cv2.circle(vis_img, re_orig, 8, (255, 255, 0), -1)  # 青色实心 - 右眼
        if beak_px:
            bk_orig = (beak_px[0] + x_orig, beak_px[1] + y_orig)
            cv2.circle(vis_img, bk_orig, 8, (0, 255, 255), -1)  # 黄色实心 - 鸟喙
        
        print("  🔵 绘制头部区域（蓝色圆圈）")
    
    # --- 🔴 红色十字: 对焦点 ---
    if focus_px is not None:
        cross_size = 60
        thickness = 4
        cv2.line(vis_img, (focus_px[0] - cross_size, focus_px[1]), 
                 (focus_px[0] + cross_size, focus_px[1]), (0, 0, 255), thickness)
        cv2.line(vis_img, (focus_px[0], focus_px[1] - cross_size), 
                 (focus_px[0], focus_px[1] + cross_size), (0, 0, 255), thickness)
        print("  🔴 绘制对焦点（红色十字）")
    
    # --- 添加图例 ---
    legend_start = 50
    cv2.putText(vis_img, "SEG Mask (Green)", (legend_start, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(vis_img, "Head Region (Blue)", (legend_start, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 0), 3)
    cv2.putText(vis_img, "Focus Point (Red)", (legend_start, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # ============================================
    # 6. 判断碰撞
    # ============================================
    print("\n" + "=" * 50)
    print("📊 碰撞检测结果:")
    
    if focus_px is not None:
        # 检查是否在 SEG 掩码内
        if bird_mask_orig is not None:
            in_seg = bird_mask_orig[focus_px[1], focus_px[0]] > 0
            print(f"  • 对焦点在 SEG 掩码内: {'✅ 是' if in_seg else '❌ 否'}")
        
        # 检查是否在头部区域内
        if head_center is not None and head_radius is not None:
            head_center_orig = (head_center[0] + x_orig, head_center[1] + y_orig)
            dist_to_head = np.sqrt((focus_px[0] - head_center_orig[0])**2 + 
                                   (focus_px[1] - head_center_orig[1])**2)
            in_head = dist_to_head <= head_radius
            print(f"  • 对焦点在头部区域内: {'✅ 是' if in_head else '❌ 否'} (距离: {dist_to_head:.0f}, 半径: {head_radius})")
    
    print("=" * 50)
    
    # ============================================
    # 7. 保存结果
    # ============================================
    # 缩小以便查看
    max_dim = 2400
    if max(w_orig, h_orig) > max_dim:
        scale = max_dim / max(w_orig, h_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        vis_img = cv2.resize(vis_img, (new_w, new_h))
    
    cv2.imwrite(output_path, vis_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\n✅ 已保存到: {output_path}")
    
    # 清理临时文件
    #os.remove(temp_jpg)
    
    return output_path


if __name__ == "__main__":
    # 默认使用测试照片
    test_nef = "/Volumes/990PRO4TB/2026/2026-01-06/3星_优选/_Z9W9956-2.NEF"
    
    if len(sys.argv) > 1:
        test_nef = sys.argv[1]
    
    if not os.path.exists(test_nef):
        print(f"❌ 文件不存在: {test_nef}")
        sys.exit(1)
    
    visualize_focus_test(test_nef)
