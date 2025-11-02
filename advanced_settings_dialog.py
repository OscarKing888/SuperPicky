#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky V3.1 - 高级设置对话框
"""

import tkinter as tk
from tkinter import ttk, messagebox
from advanced_config import get_advanced_config
from i18n import get_i18n


class AdvancedSettingsDialog:
    """高级设置对话框"""

    def __init__(self, parent):
        self.parent = parent
        self.config = get_advanced_config()
        self.i18n = get_i18n(self.config.language)
        self.dialog = None
        self.vars = {}  # 存储所有变量

    def show(self):
        """显示对话框"""
        # 创建顶层窗口
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(self.i18n.t("advanced_settings.title"))
        self.dialog.geometry("550x650")  # 增加高度以容纳语言设置和保存按钮
        self.dialog.minsize(550, 600)  # 设置最小尺寸
        self.dialog.resizable(True, True)  # 允许调整大小

        # 居中显示
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # 创建Notebook（选项卡）
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: 评分阈值
        rating_frame = ttk.Frame(notebook, padding=15)
        notebook.add(rating_frame, text=self.i18n.t("advanced_settings.zero_star_thresholds"))
        self._create_rating_tab(rating_frame)

        # Tab 2: 输出设置
        output_frame = ttk.Frame(notebook, padding=15)
        notebook.add(output_frame, text=self.i18n.t("advanced_settings.output_settings"))
        self._create_output_tab(output_frame)

        # 底部按钮
        self._create_buttons()

        # 加载当前配置
        self._load_current_config()

    def _create_rating_tab(self, parent):
        """创建评分阈值选项卡"""
        # 说明文字
        desc = ttk.Label(parent, text=self.i18n.t("advanced_settings.rating_tab_description"),
                        font=("Arial", 10), foreground="#666")
        desc.pack(pady=(0, 15))

        # AI置信度阈值
        self._create_slider_setting(
            parent,
            key="min_confidence",
            label=self.i18n.t("advanced_settings.min_confidence_label"),
            description=self.i18n.t("advanced_settings.min_confidence_description"),
            from_=0.3, to=0.7, resolution=0.05,
            default=0.5,
            format_func=lambda v: f"{v:.2f} ({int(v*100)}%)"
        )

        # 锐度最低阈值
        self._create_slider_setting(
            parent,
            key="min_sharpness",
            label=self.i18n.t("advanced_settings.min_sharpness_label"),
            description=self.i18n.t("advanced_settings.min_sharpness_description"),
            from_=2000, to=6000, resolution=100,
            default=4000,
            format_func=lambda v: f"{int(v)}"
        )

        # 美学最低阈值
        self._create_slider_setting(
            parent,
            key="min_nima",
            label=self.i18n.t("advanced_settings.min_nima_label"),
            description=self.i18n.t("advanced_settings.min_nima_description"),
            from_=3.0, to=5.0, resolution=0.1,
            default=4.0,
            format_func=lambda v: f"{v:.1f}"
        )

        # 噪点最高阈值
        self._create_slider_setting(
            parent,
            key="max_brisque",
            label=self.i18n.t("advanced_settings.max_brisque_label"),
            description=self.i18n.t("advanced_settings.max_brisque_description"),
            from_=20, to=50, resolution=1,
            default=30,
            format_func=lambda v: f"{int(v)}"
        )

    def _create_output_tab(self, parent):
        """创建输出设置选项卡"""
        # 说明文字
        desc = ttk.Label(parent, text=self.i18n.t("advanced_settings.output_tab_description"),
                        font=("Arial", 10), foreground="#666")
        desc.pack(pady=(0, 15))

        # 精选旗标Top百分比
        self._create_slider_setting(
            parent,
            key="picked_top_percentage",
            label=self.i18n.t("advanced_settings.picked_percentage_label"),
            description=self.i18n.t("advanced_settings.picked_percentage_description"),
            from_=10, to=50, resolution=5,
            default=25,
            format_func=lambda v: f"{int(v)}%"
        )

        # CSV报告 - 隐藏（不再显示，因为用户不需要了解这个技术细节）
        # CSV保存强制启用,因为二次选鸟功能需要
        self.vars["save_csv"] = tk.BooleanVar(value=True)

        # 日志详细程度
        log_frame = ttk.LabelFrame(parent, text=self.i18n.t("advanced_settings.log_level_label"), padding=10)
        log_frame.pack(fill=tk.X, pady=5)

        self.vars["log_level"] = tk.StringVar(value="detailed")
        ttk.Radiobutton(log_frame, text=self.i18n.t("advanced_settings.log_detailed_label"),
                       variable=self.vars["log_level"], value="detailed").pack(anchor=tk.W)
        ttk.Radiobutton(log_frame, text=self.i18n.t("advanced_settings.log_simple_label"),
                       variable=self.vars["log_level"], value="simple").pack(anchor=tk.W)

        # 语言设置
        lang_frame = ttk.LabelFrame(parent, text=self.i18n.t("advanced_settings.language_settings"), padding=10)
        lang_frame.pack(fill=tk.X, pady=5)

        # 获取可用语言列表
        i18n = get_i18n()
        available_languages = i18n.get_available_languages()  # {'zh_CN': '简体中文', 'en_US': 'English'}

        # 语言下拉菜单
        lang_select_frame = ttk.Frame(lang_frame)
        lang_select_frame.pack(fill=tk.X, pady=5)

        ttk.Label(lang_select_frame, text=self.i18n.t("advanced_settings.language_label"),
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 10))

        # 保存语言代码（内部使用）
        self.vars["language"] = tk.StringVar(value=self.config.language)

        # 创建语言名称到代码的映射
        self.lang_name_to_code = {}  # {'简体中文': 'zh_CN', 'English': 'en_US'}
        self.lang_code_to_name = {}  # {'zh_CN': '简体中文', 'en_US': 'English'}
        for code, name in available_languages.items():
            self.lang_name_to_code[name] = code
            self.lang_code_to_name[code] = name

        # Combobox显示语言名称
        language_combo = ttk.Combobox(
            lang_select_frame,
            values=list(available_languages.values()),  # 显示语言名称
            state="readonly",
            width=20
        )
        language_combo.pack(side=tk.LEFT)

        # 设置当前语言的显示值（显示名称）
        if self.config.language in self.lang_code_to_name:
            language_combo.set(self.lang_code_to_name[self.config.language])

        # 当选择改变时，更新内部的语言代码
        def on_language_change(event):
            selected_name = language_combo.get()
            if selected_name in self.lang_name_to_code:
                self.vars["language"].set(self.lang_name_to_code[selected_name])

        language_combo.bind('<<ComboboxSelected>>', on_language_change)

        # 提示文字
        ttk.Label(lang_frame, text=self.i18n.t("advanced_settings.language_note"),
                 font=("Arial", 9), foreground="#FF6B6B").pack(anchor=tk.W, pady=(5, 0))

    def _create_slider_setting(self, parent, key, label, description, from_, to, resolution, default, format_func):
        """创建滑块设置项"""
        frame = ttk.LabelFrame(parent, text=label, padding=10)
        frame.pack(fill=tk.X, pady=5)

        # 描述文字
        ttk.Label(frame, text=description, font=("Arial", 9),
                 foreground="#888").pack(anchor=tk.W)

        # 滑块和值显示
        slider_frame = ttk.Frame(frame)
        slider_frame.pack(fill=tk.X, pady=(5, 0))

        self.vars[key] = tk.DoubleVar(value=default)

        slider = ttk.Scale(slider_frame, from_=from_, to=to,
                          variable=self.vars[key], orient=tk.HORIZONTAL)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        value_label = ttk.Label(slider_frame, text=format_func(default),
                               width=12, font=("Arial", 10, "bold"))
        value_label.pack(side=tk.LEFT)

        # 更新标签的回调
        def update_label(*args):
            value_label.configure(text=format_func(self.vars[key].get()))

        self.vars[key].trace_add('write', update_label)

    def _create_buttons(self):
        """创建底部按钮"""
        btn_frame = ttk.Frame(self.dialog, padding=10)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # 恢复默认值
        ttk.Button(btn_frame, text=self.i18n.t("advanced_settings.reset_to_default"),
                  command=self._reset_to_default).pack(side=tk.LEFT, padx=5)

        # 右侧按钮
        right_buttons = ttk.Frame(btn_frame)
        right_buttons.pack(side=tk.RIGHT)

        ttk.Button(right_buttons, text=self.i18n.t("buttons.cancel"),
                  command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(right_buttons, text=self.i18n.t("buttons.save"),
                  command=self._save_settings).pack(side=tk.LEFT, padx=5)

    def _load_current_config(self):
        """加载当前配置到界面"""
        self.vars["min_confidence"].set(self.config.min_confidence)
        self.vars["min_sharpness"].set(self.config.min_sharpness)
        self.vars["min_nima"].set(self.config.min_nima)
        self.vars["max_brisque"].set(self.config.max_brisque)
        self.vars["picked_top_percentage"].set(self.config.picked_top_percentage)
        self.vars["save_csv"].set(self.config.save_csv)
        self.vars["log_level"].set(self.config.log_level)
        if "language" in self.vars:
            self.vars["language"].set(self.config.language)

    def _reset_to_default(self):
        """恢复默认设置"""
        if messagebox.askyesno(self.i18n.t("messages.reset_confirm_title"),
                             self.i18n.t("advanced_settings.settings_reset")):
            self.config.reset_to_default()
            self._load_current_config()
            messagebox.showinfo(self.i18n.t("advanced_settings.settings_reset_title"),
                              self.i18n.t("advanced_settings.settings_reset"))

    def _save_settings(self):
        """保存设置"""
        # 更新配置
        self.config.set_min_confidence(self.vars["min_confidence"].get())
        self.config.set_min_sharpness(self.vars["min_sharpness"].get())
        self.config.set_min_nima(self.vars["min_nima"].get())
        self.config.set_max_brisque(self.vars["max_brisque"].get())
        self.config.set_picked_top_percentage(self.vars["picked_top_percentage"].get())
        # CSV保存强制为True,因为二次选鸟功能需要
        self.config.set_save_csv(True)
        self.config.set_log_level(self.vars["log_level"].get())

        # 保存语言设置
        if "language" in self.vars:
            self.config.set_language(self.vars["language"].get())

        # 保存到文件
        if self.config.save():
            message = self.i18n.t("advanced_settings.settings_saved") + "\n" + \
                     self.i18n.t("advanced_settings.language_note")
            messagebox.showinfo(self.i18n.t("advanced_settings.settings_saved_title"), message)
            self.dialog.destroy()
        else:
            messagebox.showerror(self.i18n.t("errors.error_title"),
                               self.i18n.t("advanced_settings.settings_save_failed", error=""))
