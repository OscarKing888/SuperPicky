#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky V3.2 - Post Digital Adjustment Dialog
二次选鸟对话框 - 简洁专业风格
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Set, Optional
from post_adjustment_engine import PostAdjustmentEngine
from exiftool_manager import get_exiftool_manager
from advanced_config import get_advanced_config
from i18n import get_i18n


class PostAdjustmentDialog:
    """二次选鸟对话框"""

    def __init__(self, parent, directory: str, current_sharpness: int = 7500,
                 current_nima: float = 4.8, on_complete_callback=None):
        """
        初始化对话框

        Args:
            parent: 父窗口
            directory: 照片目录
            current_sharpness: 当前UI设置的锐度阈值
            current_nima: 当前UI设置的美学阈值
            on_complete_callback: 完成后的回调函数
        """
        self.window = tk.Toplevel(parent)

        # 初始化配置和国际化
        self.config = get_advanced_config()
        self.i18n = get_i18n(self.config.language)

        self.window.title(self.i18n.t("post_adjustment.title"))
        self.window.geometry("750x800")  # 高度增加100
        self.window.resizable(False, False)

        self.directory = directory
        self.on_complete_callback = on_complete_callback

        # 初始化引擎
        self.engine = PostAdjustmentEngine(directory)

        # 0星阈值变量（从高级配置加载）
        self.min_confidence_var = tk.DoubleVar(value=self.config.min_confidence)
        self.min_sharpness_var = tk.IntVar(value=self.config.min_sharpness)
        self.min_nima_var = tk.DoubleVar(value=self.config.min_nima)
        self.max_brisque_var = tk.IntVar(value=self.config.max_brisque)

        # 2/3星阈值变量（从主界面当前设置加载）
        self.sharpness_threshold_var = tk.IntVar(value=current_sharpness)
        self.nima_threshold_var = tk.DoubleVar(value=current_nima)
        self.picked_percentage_var = tk.IntVar(value=self.config.picked_top_percentage)

        # 数据
        self.original_photos: List[Dict] = []
        self.updated_photos: List[Dict] = []
        self.picked_files: Set[str] = set()

        # 统计数据
        self.current_stats: Optional[Dict] = None
        self.preview_stats: Optional[Dict] = None

        # 防抖定时器
        self._preview_timer = None

        # 创建UI
        self._create_widgets()

        # 加载数据
        self._load_data()

        # 居中窗口
        self._center_window()

    def _create_widgets(self):
        """创建UI组件 - 与主界面风格一致"""

        # ===== 1. 顶部说明 =====
        desc_frame = ttk.Frame(self.window, padding=15)
        desc_frame.pack(fill=tk.X)

        ttk.Label(
            desc_frame,
            text=self.i18n.t("post_adjustment.description"),
            font=("PingFang SC", 16),
            foreground="#666"
        ).pack()

        # ===== 2. 当前统计区域 =====
        stats_frame = ttk.LabelFrame(
            self.window,
            text=self.i18n.t("stats.current_stats"),
            padding=15
        )
        stats_frame.pack(fill=tk.X, padx=15, pady=(15, 10))

        # 配置LabelFrame标题字体
        stats_frame_style = ttk.Style()
        stats_frame_style.configure('Stats.TLabelframe.Label', font=('PingFang SC', 16))
        stats_frame.configure(style='Stats.TLabelframe')

        # 使用Text组件以支持行间距设置
        self.current_stats_label = tk.Text(
            stats_frame,
            height=7,
            font=("Arial", 14),
            spacing1=4,
            spacing2=2,
            spacing3=4,
            relief=tk.FLAT,
            wrap=tk.WORD,
            state='disabled'
        )
        self.current_stats_label.pack(fill=tk.BOTH)

        # ===== 3. 阈值调整区域 =====
        threshold_frame = ttk.LabelFrame(
            self.window,
            text=self.i18n.t("post_adjustment.threshold_group"),
            padding=15
        )
        threshold_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        # 配置LabelFrame标题字体
        threshold_frame_style = ttk.Style()
        threshold_frame_style.configure('Threshold.TLabelframe.Label', font=('PingFang SC', 16))
        threshold_frame.configure(style='Threshold.TLabelframe')

        # 说明
        ttk.Label(
            threshold_frame,
            text=self.i18n.t("post_adjustment.threshold_description"),
            font=("PingFang SC", 11),
            foreground="#666"
        ).pack(pady=(0, 12))

        # 锐度阈值
        self._create_slider(
            threshold_frame,
            self.i18n.t("post_adjustment.sharpness_threshold"),
            self.sharpness_threshold_var,
            from_=6000, to=9000,
            step=100,
            format_func=lambda v: f"{v:.0f}"
        )

        # 美学阈值
        self._create_slider(
            threshold_frame,
            self.i18n.t("post_adjustment.nima_threshold"),
            self.nima_threshold_var,
            from_=4.5, to=5.5,
            step=0.1,
            format_func=lambda v: f"{v:.1f}"
        )

        # 精选百分比
        self._create_slider(
            threshold_frame,
            self.i18n.t("post_adjustment.picked_percentage"),
            self.picked_percentage_var,
            from_=10, to=50,
            step=5,
            format_func=lambda v: f"{v:.0f}%"
        )

        # ===== 4. 预览区域 =====
        preview_frame = ttk.LabelFrame(
            self.window,
            text=self.i18n.t("post_adjustment.preview_title"),
            padding=15
        )
        preview_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        # 配置LabelFrame标题字体
        preview_frame_style = ttk.Style()
        preview_frame_style.configure('Preview.TLabelframe.Label', font=('PingFang SC', 16))
        preview_frame.configure(style='Preview.TLabelframe')

        # 使用Text组件以支持行间距设置
        self.preview_stats_label = tk.Text(
            preview_frame,
            height=7,
            font=("Arial", 14),
            spacing1=4,
            spacing2=2,
            spacing3=4,
            relief=tk.FLAT,
            wrap=tk.WORD,
            state='disabled'
        )
        self.preview_stats_label.pack(fill=tk.BOTH)

        # ===== 5. 进度区域（隐藏）=====
        self.progress_frame = ttk.Frame(self.window, padding=10)
        # 不pack，需要时再显示

        self.progress_label = ttk.Label(
            self.progress_frame,
            text="",
            font=("Arial", 13)
        )
        self.progress_label.pack()

        # ===== 6. 底部按钮 =====
        btn_frame = ttk.Frame(self.window, padding=15)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # 左侧取消按钮
        ttk.Button(
            btn_frame,
            text=self.i18n.t("buttons.cancel"),
            command=self.window.destroy,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        # 右侧应用按钮
        self.apply_btn = ttk.Button(
            btn_frame,
            text=self.i18n.t("buttons.apply"),
            command=self._apply_new_ratings,
            width=15,
            state='disabled'
        )
        self.apply_btn.pack(side=tk.RIGHT, padx=5)

    def _create_slider(self, parent, label_text, variable, from_, to, step, format_func):
        """创建滑块组件，支持步进"""
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, pady=6)

        # 标签（左侧）
        label = ttk.Label(
            container,
            text=label_text,
            width=18,
            font=("Arial", 13)
        )
        label.pack(side=tk.LEFT)

        # 滑块（中间）
        slider = ttk.Scale(
            container,
            from_=from_,
            to=to,
            variable=variable,
            orient=tk.HORIZONTAL,
            command=lambda v: self._on_slider_change(variable, float(v), step, value_label, format_func)
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 数值显示（右侧）
        value_label = ttk.Label(
            container,
            text=format_func(variable.get()),
            width=8,
            font=("Arial", 13)
        )
        value_label.pack(side=tk.LEFT)

        # 初始化标签
        self._snap_to_step(variable, variable.get(), step)
        value_label.config(text=format_func(variable.get()))

    def _snap_to_step(self, variable, value, step):
        """将值对齐到步进"""
        snapped = round(value / step) * step
        variable.set(snapped)

    def _on_slider_change(self, variable, value, step, value_label, format_func):
        """滑块变化回调：步进+更新标签+触发预览"""
        # 步进对齐
        self._snap_to_step(variable, value, step)

        # 更新数值标签
        value_label.config(text=format_func(variable.get()))

        # 触发预览更新
        self._on_threshold_changed()

    def _load_data(self):
        """加载CSV数据"""
        success, message = self.engine.load_report()

        if not success:
            messagebox.showerror(self.i18n.t("errors.error_title"), message)
            self.window.destroy()
            return

        self.original_photos = self.engine.photos_data.copy()
        self.current_stats = self._get_original_statistics()
        self._update_current_stats_display()

        self.apply_btn.config(state='normal')
        self._on_threshold_changed()

    def _get_original_statistics(self) -> Dict[str, int]:
        """获取原始统计（包括重新计算picked）"""
        stats = {
            'star_0': 0,
            'star_1': 0,
            'star_2': 0,
            'star_3': 0,
            'picked': 0,
            'total': len(self.original_photos)
        }

        star_3_photos = []

        for photo in self.original_photos:
            rating_str = photo.get('rating', 0)
            rating = int(rating_str) if isinstance(rating_str, (int, str)) and str(rating_str).isdigit() else 0

            if rating == 0:
                stats['star_0'] += 1
            elif rating == 1:
                stats['star_1'] += 1
            elif rating == 2:
                stats['star_2'] += 1
            elif rating == 3:
                stats['star_3'] += 1
                star_3_photos.append(photo)

        # 基于当前配置重新计算精选旗标数量
        picked_files = self.engine.recalculate_picked(
            star_3_photos,
            self.picked_percentage_var.get()
        )
        stats['picked'] = len(picked_files)

        return stats

    def _update_current_stats_display(self):
        """更新当前统计显示 - 大字体"""
        if not self.current_stats:
            return

        stats = self.current_stats
        total = stats['total']

        text = self.i18n.t("stats.total_bird_photos", total=total) + "\n\n"

        if stats.get('picked', 0) > 0:
            text += self.i18n.t("stats.picked_count", count=stats['picked']) + "\n"

        text += self.i18n.t("stats.star_3_count", count=stats['star_3'], percent=stats['star_3']/total*100) + "\n"
        text += self.i18n.t("stats.star_2_count", count=stats['star_2'], percent=stats['star_2']/total*100) + "\n"
        text += self.i18n.t("stats.star_1_count", count=stats['star_1'], percent=stats['star_1']/total*100) + "\n"
        text += self.i18n.t("stats.star_0_count", count=stats['star_0'], percent=stats['star_0']/total*100)

        self.current_stats_label.config(state=tk.NORMAL)
        self.current_stats_label.delete("1.0", tk.END)
        self.current_stats_label.insert("1.0", text)
        self.current_stats_label.config(state=tk.DISABLED)

    def _on_threshold_changed(self):
        """阈值改变回调（防抖）"""
        if self._preview_timer:
            self.window.after_cancel(self._preview_timer)

        self._preview_timer = self.window.after(300, self._update_preview)

    def _update_preview(self):
        """更新预览统计"""
        # 获取当前阈值
        min_confidence = self.min_confidence_var.get()
        min_sharpness = self.min_sharpness_var.get()
        min_nima = self.min_nima_var.get()
        max_brisque = self.max_brisque_var.get()
        sharpness_threshold = self.sharpness_threshold_var.get()
        nima_threshold = self.nima_threshold_var.get()
        picked_percentage = self.picked_percentage_var.get()

        # 重新计算
        self.updated_photos = self.engine.recalculate_ratings(
            self.original_photos,
            min_confidence=min_confidence,
            min_sharpness=min_sharpness,
            min_nima=min_nima,
            max_brisque=max_brisque,
            sharpness_threshold=sharpness_threshold,
            nima_threshold=nima_threshold
        )

        star_3_photos = [p for p in self.updated_photos if p.get('新星级') == 3]
        self.picked_files = self.engine.recalculate_picked(star_3_photos, picked_percentage)

        self.preview_stats = self.engine.get_statistics(self.updated_photos)
        self.preview_stats['picked'] = len(self.picked_files)

        self._update_preview_display()

    def _update_preview_display(self):
        """更新预览显示 - 大字体"""
        if not self.preview_stats or not self.current_stats:
            return

        old = self.current_stats
        new = self.preview_stats

        def format_diff(old_val, new_val, total):
            diff = new_val - old_val
            pct = new_val / total * 100 if total > 0 else 0

            if diff > 0:
                return self.i18n.t("stats.count_with_percent_increase", count=new_val, percent=pct, diff=diff)
            elif diff < 0:
                return self.i18n.t("stats.count_with_percent_decrease", count=new_val, percent=pct, diff=diff)
            else:
                return self.i18n.t("stats.count_with_percent_nochange", count=new_val, percent=pct)

        total = new['total']
        text = ""

        # 精选旗标
        picked_count = new['picked']
        star_3_count = new['star_3']
        if star_3_count > 0:
            picked_pct = picked_count / star_3_count * 100
            old_picked = old.get('picked', 0)
            picked_diff = picked_count - old_picked
            if picked_diff > 0:
                text += self.i18n.t("stats.picked_diff_increase", count=picked_count, pct=picked_pct, diff=picked_diff) + "\n\n"
            elif picked_diff < 0:
                text += self.i18n.t("stats.picked_diff_decrease", count=picked_count, pct=picked_pct, diff=picked_diff) + "\n\n"
            else:
                text += self.i18n.t("stats.picked_diff_nochange", count=picked_count, pct=picked_pct) + "\n\n"
        else:
            text += self.i18n.t("stats.picked_no_three_star") + "\n\n"

        text += "⭐⭐⭐ 3星: " + format_diff(old['star_3'], new['star_3'], total) + "\n"
        text += "⭐⭐ 2星: " + format_diff(old['star_2'], new['star_2'], total) + "\n"
        text += "⭐ 1星: " + format_diff(old['star_1'], new['star_1'], total) + "\n"
        text += "0星: " + format_diff(old['star_0'], new['star_0'], total)

        self.preview_stats_label.config(state=tk.NORMAL)
        self.preview_stats_label.delete("1.0", tk.END)
        self.preview_stats_label.insert("1.0", text)
        self.preview_stats_label.config(state=tk.DISABLED)

    def _apply_new_ratings(self):
        """应用新评分"""
        if not self.updated_photos:
            messagebox.showwarning(
                self.i18n.t("messages.hint"),
                self.i18n.t("post_adjustment.no_data_warning")
            )
            return

        msg = self.i18n.t("post_adjustment.apply_confirm_msg", count=len(self.updated_photos))

        if not messagebox.askyesno(self.i18n.t("post_adjustment.apply_confirm_title"), msg):
            return

        self.apply_btn.config(state='disabled')
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)

        self.progress_frame.pack(fill=tk.X, padx=15, pady=10)
        self.progress_label.config(text=self.i18n.t("post_adjustment.preparing_data", count=len(self.updated_photos)))
        self.window.update()

        batch_data = []
        not_found_count = 0

        for photo in self.updated_photos:
            filename = photo['filename']
            file_path = self.engine.find_image_file(filename)

            if not file_path:
                not_found_count += 1
                continue

            rating = photo.get('新星级', 0)
            pick = 1 if filename in self.picked_files else 0

            batch_data.append({
                'file': file_path,
                'rating': rating,
                'pick': pick
            })

        if not_found_count > 0:
            self.progress_label.config(text=self.i18n.t("post_adjustment.files_not_found", count=not_found_count))
            self.window.update()

        try:
            self.progress_label.config(
                text=self.i18n.t("post_adjustment.writing_exif", count=len(batch_data))
            )
            self.window.update()

            exiftool_mgr = get_exiftool_manager()
            stats = exiftool_mgr.batch_set_metadata(batch_data)

            self.progress_frame.pack_forget()

            if not_found_count > 0:
                result_msg = self.i18n.t("post_adjustment.apply_success_with_skip",
                                        success=stats['success'],
                                        failed=stats['failed'],
                                        skipped=not_found_count)
            else:
                result_msg = self.i18n.t("post_adjustment.apply_success_msg",
                                        success=stats['success'],
                                        failed=stats['failed'])

            messagebox.showinfo(self.i18n.t("post_adjustment.apply_success_title"), result_msg)

            if self.on_complete_callback:
                self.on_complete_callback()

            self.window.destroy()

        except Exception as e:
            self.progress_frame.pack_forget()
            self.apply_btn.config(state='normal')
            messagebox.showerror(
                self.i18n.t("post_adjustment.apply_error_title"),
                self.i18n.t("post_adjustment.apply_error_msg", error=str(e))
            )

    def _center_window(self):
        """居中窗口"""
        try:
            # 确保窗口已经完全创建
            self.window.update_idletasks()

            # 使用指定的宽高（750x800）
            width = 750
            height = 800

            # 计算居中位置
            screen_width = self.window.winfo_screenwidth()
            screen_height = self.window.winfo_screenheight()
            x = (screen_width // 2) - (width // 2)
            y = (screen_height // 2) - (height // 2)

            # 设置窗口位置
            self.window.geometry(f'{width}x{height}+{x}+{y}')
        except Exception as e:
            # 如果居中失败，不影响窗口显示
            print(f"警告: 窗口居中失败: {e}")
