# -*- coding: utf-8 -*-
"""
SuperPicky - 参数设置对话框
顶部标签页布局
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton,
    QWidget, QFrame, QRadioButton,
    QButtonGroup, QTabWidget, QCheckBox, QComboBox
)
from PySide6.QtCore import Qt, Slot

from advanced_config import get_advanced_config
from tools.i18n import get_i18n
from ui.styles import COLORS, FONTS
from ui.custom_dialogs import StyledMessageBox


class AdvancedSettingsDialog(QDialog):
    """参数设置对话框 - 顶部标签页布局"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = get_advanced_config()
        self.i18n = get_i18n(self.config.language)

        self.vars = {}

        self._setup_ui()
        self._load_current_config()

    def _setup_ui(self):
        """设置 UI"""
        self.setWindowTitle(self.i18n.t("advanced_settings.window_title"))
        self.setMinimumSize(480, 480)
        self.resize(520, 520)
        self.setModal(True)

        # 应用样式
        self._apply_styles()

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 8, 0, 0)
        main_layout.setSpacing(0)

        # 标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background-color: {COLORS['bg_primary']};
            }}
            QTabBar::tab {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_secondary']};
                padding: 10px 24px;
                border: none;
                border-bottom: 2px solid transparent;
                font-size: 13px;
            }}
            QTabBar::tab:selected {{
                color: {COLORS['text_primary']};
                border-bottom: 2px solid {COLORS['accent']};
            }}
            QTabBar::tab:hover:!selected {{
                color: {COLORS['text_primary']};
                background-color: {COLORS['bg_elevated']};
            }}
        """)

        # 添加标签页
        self.tab_widget.addTab(
            self._create_culling_page(),
            self.i18n.t("advanced_settings.section_selection")
        )
        self.tab_widget.addTab(
            self._create_output_page(),
            self.i18n.t("advanced_settings.section_output")
        )

        main_layout.addWidget(self.tab_widget, 1)

        # 底部按钮区域
        self._create_buttons(main_layout)

    def _apply_styles(self):
        """应用全局样式"""
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_primary']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QPushButton {{
                background-color: {COLORS['accent']};
                color: {COLORS['bg_void']};
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #00e6b8;
            }}
            QPushButton#secondary {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
            QPushButton#secondary:hover {{
                border-color: {COLORS['text_muted']};
                color: {COLORS['text_primary']};
            }}
            QPushButton#tertiary {{
                background-color: transparent;
                color: {COLORS['text_tertiary']};
            }}
            QPushButton#tertiary:hover {{
                color: {COLORS['text_secondary']};
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {COLORS['bg_input']};
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS['accent']};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 16px;
                height: 16px;
                margin: -6px 0;
                background: {COLORS['text_primary']};
                border-radius: 8px;
            }}
            QRadioButton {{
                color: {COLORS['text_secondary']};
                font-size: 13px;
                color: {COLORS['text_secondary']};
                font-size: 13px;
                spacing: 8px;
                padding: 2px;
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
            }}
            QRadioButton::indicator:unchecked {{
                border: 2px solid {COLORS['text_tertiary']};
                border-radius: 9px;
                background: transparent;
            }}
            QRadioButton::indicator:checked {{
                border: 2px solid {COLORS['accent']};
                border-radius: 9px;
                background: {COLORS['accent']};
            }}
        """)

    def _create_culling_page(self):
        """创建选片设置页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        # 检测敏感度
        self.vars["min_confidence"] = self._create_slider_setting(
            layout,
            self.i18n.t("advanced_settings.detection_sensitivity"),
            self.i18n.t("advanced_settings.detection_sensitivity_hint"),
            min_val=30, max_val=70, default=50,
            format_func=lambda v: f"{v}%"
        )

        # 清晰度要求
        self.vars["min_sharpness"] = self._create_slider_setting(
            layout,
            self.i18n.t("advanced_settings.sharpness_requirement"),
            self.i18n.t("advanced_settings.sharpness_requirement_hint"),
            min_val=100, max_val=500, default=100,
            step=50
        )

        # 画面美感要求
        self.vars["min_nima"] = self._create_slider_setting(
            layout,
            self.i18n.t("advanced_settings.aesthetics_requirement"),
            self.i18n.t("advanced_settings.aesthetics_requirement_hint"),
            min_val=30, max_val=50, default=40,
            format_func=lambda v: f"{v/10:.1f}",
            scale=10
        )

        # 分隔线
        self._add_divider(layout)

        # 识别确信度
        self.vars["birdid_confidence"] = self._create_slider_setting(
            layout,
            self.i18n.t("advanced_settings.birdid_confidence"),
            self.i18n.t("advanced_settings.birdid_confidence_hint"),
            min_val=50, max_val=95, default=70,
            step=5,
            format_func=lambda v: f"{v}%"
        )

        # 连拍速度
        self.vars["burst_fps"] = self._create_slider_setting(
            layout,
            self.i18n.t("advanced_settings.burst_fps"),
            self.i18n.t("advanced_settings.burst_fps_hint"),
            min_val=4, max_val=20, default=10,
            step=1,
            format_func=lambda v: f"{v} fps"
        )

        layout.addStretch()
        return page

    def _create_output_page(self):
        """创建输出设置页面 - XMP 设置"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # 页面标题
        title = QLabel(self.i18n.t("advanced_settings.xmp_write_mode"))
        title.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 4px;
        """)
        layout.addWidget(title)

        # XMP 写入方式 - 使用单选按钮组
        xmp_group_widget = QWidget()
        xmp_group_widget.setObjectName("xmpGroup")
        xmp_group_widget.setStyleSheet(f"""
            #xmpGroup {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 8px;
            }}
        """)
        xmp_layout = QVBoxLayout(xmp_group_widget)
        xmp_layout.setContentsMargins(16, 12, 16, 12)
        xmp_layout.setSpacing(0)

        self.xmp_button_group = QButtonGroup(self)

        # 选项1: 嵌入 RAW
        embedded_container = QWidget()
        embedded_layout = QVBoxLayout(embedded_container)
        embedded_layout.setContentsMargins(0, 8, 0, 8)
        embedded_layout.setSpacing(4)

        embedded_option = QRadioButton(self.i18n.t("advanced_settings.xmp_mode_embedded"))
        self.vars["xmp_embedded"] = embedded_option
        self.xmp_button_group.addButton(embedded_option, 0)
        embedded_layout.addWidget(embedded_option)

        embedded_hint = QLabel(self.i18n.t("advanced_settings.xmp_mode_embedded_hint"))
        embedded_hint.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 11px;
            margin-left: 24px;
        """)
        embedded_layout.addWidget(embedded_hint)

        xmp_layout.addWidget(embedded_container)

        # 分隔线
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        xmp_layout.addWidget(sep)

        # 选项2: XMP 侧车文件（推荐）
        sidecar_container = QWidget()
        sidecar_layout = QVBoxLayout(sidecar_container)
        sidecar_layout.setContentsMargins(0, 8, 0, 8)
        sidecar_layout.setSpacing(4)

        sidecar_option = QRadioButton(self.i18n.t("advanced_settings.xmp_mode_sidecar"))
        # 使用全局样式，移除由于覆盖样式导致的 indicator 丢失问题
        self.vars["xmp_sidecar"] = sidecar_option
        self.xmp_button_group.addButton(sidecar_option, 1)
        sidecar_layout.addWidget(sidecar_option)

        sidecar_hint = QLabel(self.i18n.t("advanced_settings.xmp_mode_sidecar_hint"))
        sidecar_hint.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 11px;
            margin-left: 24px;
        """)
        sidecar_layout.addWidget(sidecar_hint)

        # 推荐标签
        recommend_label = QLabel(self.i18n.t("advanced_settings.xmp_mode_sidecar_recommend"))
        recommend_label.setStyleSheet(f"""
            color: {COLORS['accent']};
            font-size: 11px;
            margin-left: 24px;
        """)
        sidecar_layout.addWidget(recommend_label)

        xmp_layout.addWidget(sidecar_container)

        layout.addWidget(xmp_group_widget)

        # 预览图管理
        preview_group_widget = QWidget()
        preview_group_widget.setObjectName("previewGroup")
        preview_group_widget.setStyleSheet(f"""
            #previewGroup {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 8px;
            }}
        """)
        preview_layout = QVBoxLayout(preview_group_widget)
        preview_layout.setContentsMargins(16, 12, 16, 12)
        preview_layout.setSpacing(12)

        # 标题
        preview_title = QLabel(self.i18n.t("advanced_settings.preview_management"))
        preview_title.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 13px;
            font-weight: 500;
        """)
        preview_layout.addWidget(preview_title)

        # 1. Checkbox: 保留预览图片
        keep_preview_check = QCheckBox(self.i18n.t("advanced_settings.keep_preview"))
        # 使用全局样式，移除由于覆盖样式导致的 indicator 丢失问题
        self.vars["keep_temp_files"] = keep_preview_check
        preview_layout.addWidget(keep_preview_check)

        keep_preview_hint = QLabel(self.i18n.t("advanced_settings.keep_preview_hint"))
        keep_preview_hint.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 11px;
            margin-left: 24px;
        """)
        preview_layout.addWidget(keep_preview_hint)

        # 2. Combo: 自动清理周期
        cleanup_container = QWidget()
        cleanup_layout = QHBoxLayout(cleanup_container)
        cleanup_layout.setContentsMargins(0, 4, 0, 0)
        cleanup_layout.setSpacing(10)

        cleanup_label = QLabel(self.i18n.t("advanced_settings.auto_cleanup"))
        cleanup_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 13px;
            margin-left: 24px;
        """)
        cleanup_layout.addWidget(cleanup_label)

        cleanup_combo = QComboBox()
        cleanup_combo.addItem(self.i18n.t("advanced_settings.cleanup_3_days"), 3)
        cleanup_combo.addItem(self.i18n.t("advanced_settings.cleanup_7_days"), 7)
        cleanup_combo.addItem(self.i18n.t("advanced_settings.cleanup_30_days"), 30)
        cleanup_combo.addItem(self.i18n.t("advanced_settings.cleanup_forever"), 0)
        
        self.vars["auto_cleanup_days"] = cleanup_combo
        cleanup_layout.addWidget(cleanup_combo)
        cleanup_layout.addStretch()

        preview_layout.addWidget(cleanup_container)
        
        # 联动逻辑：不保留预览图时，禁用清理周期设置
        keep_preview_check.toggled.connect(cleanup_container.setEnabled)

        layout.addWidget(preview_group_widget)

        layout.addStretch()
        return page

    def _add_divider(self, layout):
        """添加分隔线"""
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {COLORS['border_subtle']}; margin: 4px 0;")
        layout.addWidget(divider)

    def _create_slider_setting(self, layout, label_text, hint_text,
                               min_val, max_val, default, step=1,
                               format_func=None, scale=1):
        """创建滑块设置项"""
        container = QHBoxLayout()
        container.setSpacing(16)

        label = QLabel(label_text)
        label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; min-width: 80px;")
        container.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setSingleStep(step)
        container.addWidget(slider, 1)

        if format_func is None:
            format_func = lambda v: str(v)

        value_label = QLabel(format_func(default))
        value_label.setStyleSheet(f"""
            color: {COLORS['accent']};
            font-size: 14px;
            font-family: {FONTS['mono']};
            font-weight: 500;
            min-width: 50px;
        """)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        container.addWidget(value_label)

        slider.valueChanged.connect(lambda v: value_label.setText(format_func(v)))

        layout.addLayout(container)

        # 添加小字提示
        hint_label = QLabel(hint_text)
        hint_label.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 11px;
            margin-left: 96px;
            margin-bottom: 8px;
        """)
        layout.addWidget(hint_label)

        slider.scale = scale
        return slider

    def _create_buttons(self, layout):
        """创建底部按钮"""
        btn_container = QWidget()
        btn_container.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_card']};
                border-top: 1px solid {COLORS['border_subtle']};
            }}
        """)

        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(24, 20, 24, 20)

        # 恢复默认
        reset_btn = QPushButton(self.i18n.t("advanced_settings.reset_defaults"))
        reset_btn.setObjectName("tertiary")
        reset_btn.clicked.connect(self._reset_to_default)
        btn_layout.addWidget(reset_btn)

        btn_layout.addStretch()

        # 取消
        cancel_btn = QPushButton(self.i18n.t("advanced_settings.cancel"))
        cancel_btn.setObjectName("secondary")
        cancel_btn.setMinimumWidth(80)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        # 保存 - 使用绿色主色调
        save_btn = QPushButton(self.i18n.t("advanced_settings.save"))
        save_btn.setMinimumWidth(80)
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: {COLORS['bg_void']};
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #00e6b8;
            }}
        """)
        save_btn.clicked.connect(self._save_settings)
        btn_layout.addWidget(save_btn)

        layout.addWidget(btn_container)

    def _load_current_config(self):
        """加载当前配置"""
        self.vars["min_confidence"].setValue(int(self.config.min_confidence * 100))
        self.vars["min_sharpness"].setValue(int(self.config.min_sharpness))
        self.vars["min_nima"].setValue(int(self.config.min_nima * 10))
        self.vars["burst_fps"].setValue(int(self.config.burst_fps))
        self.vars["birdid_confidence"].setValue(int(self.config.birdid_confidence))

        # 加载 XMP 设置
        try:
            arw_mode = self.config.arw_write_mode
            if arw_mode in ("sidecar", "auto"):
                self.vars["xmp_sidecar"].setChecked(True)
            else:
                self.vars["xmp_embedded"].setChecked(True)
        except Exception:
            self.vars["xmp_embedded"].setChecked(True)

        # 加载预览图设置
        keep_temp = self.config.keep_temp_files
        self.vars["keep_temp_files"].setChecked(keep_temp)
        
        cleanup_days = self.config.auto_cleanup_days
        combo = self.vars["auto_cleanup_days"]
        index = combo.findData(cleanup_days)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setCurrentIndex(2) # Default to 30 days if not found

    @Slot()
    def _reset_to_default(self):
        """恢复默认设置"""
        reply = StyledMessageBox.question(
            self,
            self.i18n.t("advanced_settings.confirm_reset_title"),
            self.i18n.t("advanced_settings.confirm_reset_msg"),
            yes_text=self.i18n.t("advanced_settings.yes"),
            no_text=self.i18n.t("advanced_settings.cancel")
        )

        if reply == StyledMessageBox.Yes:
            self.config.reset_to_default()
            self._load_current_config()
            StyledMessageBox.information(
                self,
                self.i18n.t("advanced_settings.reset_done_title"),
                self.i18n.t("advanced_settings.reset_done_msg"),
                ok_text=self.i18n.t("buttons.close")
            )

    @Slot()
    def _save_settings(self):
        """保存设置"""
        min_confidence = self.vars["min_confidence"].value() / 100.0
        min_sharpness = self.vars["min_sharpness"].value()
        min_nima = self.vars["min_nima"].value() / 10.0
        burst_fps = self.vars["burst_fps"].value()
        birdid_confidence = self.vars["birdid_confidence"].value()

        self.config.set_min_confidence(min_confidence)
        self.config.set_min_sharpness(min_sharpness)
        self.config.set_min_nima(min_nima)
        self.config.set_burst_fps(burst_fps)
        self.config.set_birdid_confidence(birdid_confidence)

        # 保存 XMP 设置
        xmp_mode = "sidecar" if self.vars["xmp_sidecar"].isChecked() else "embedded"
        self.config.set_arw_write_mode(xmp_mode)
        self.config.set_arw_write_mode(xmp_mode)
        self.config.set_save_csv(True)

        # 保存预览图设置
        self.config.set_keep_temp_files(self.vars["keep_temp_files"].isChecked())
        self.config.set_auto_cleanup_days(self.vars["auto_cleanup_days"].currentData())

        if self.config.save():
            StyledMessageBox.information(
                self,
                self.i18n.t("advanced_settings.save_success_title"),
                self.i18n.t("advanced_settings.save_success_msg"),
                ok_text=self.i18n.t("buttons.close")
            )
            self.accept()
        else:
            StyledMessageBox.critical(
                self,
                self.i18n.t("advanced_settings.save_error_title"),
                self.i18n.t("advanced_settings.save_error_msg"),
                ok_text=self.i18n.t("buttons.close")
            )

    @Slot(str, int, float)
    def _on_skill_level_changed(self, level_key: str, sharpness: int, aesthetics: float):
        """处理水平变化"""
        self.config.set_skill_level(level_key)
        self.config.save()

        if self.parent() and hasattr(self.parent(), '_apply_skill_level_thresholds'):
            self.parent()._apply_skill_level_thresholds(level_key)

        print(f"✅ 已切换摄影水平: {level_key} (锐度={sharpness}, 美学={aesthetics})")
