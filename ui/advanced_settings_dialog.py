# -*- coding: utf-8 -*-
"""
SuperPicky - 参数设置对话框
简化版 - 移除技术术语，使普通用户更易理解
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QComboBox,
    QWidget, QFrame, QCheckBox
)
from PySide6.QtCore import Qt, Slot

from advanced_config import get_advanced_config
from tools.i18n import get_i18n
from ui.styles import COLORS, FONTS
from ui.custom_dialogs import StyledMessageBox


class AdvancedSettingsDialog(QDialog):
    """参数设置对话框 - 简化版"""

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
        self.setMinimumSize(500, 680)
        self.resize(520, 800)
        self.setModal(True)

        # 应用样式
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
            QComboBox {{
                background-color: {COLORS['bg_input']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px 16px;
                color: {COLORS['text_primary']};
                font-size: 13px;
                min-width: 150px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {COLORS['text_tertiary']};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['bg_elevated']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 4px;
                selection-background-color: {COLORS['accent']};
                selection-color: {COLORS['bg_void']};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # 标题
        title = QLabel(self.i18n.t("advanced_settings.title"))
        title.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 18px;
            font-weight: 600;
        """)
        layout.addWidget(title)

        # === 选片标准 ===
        self._create_section_title(layout, self.i18n.t("advanced_settings.section_selection"))
        
        # 检测敏感度（原 AI 置信度）
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
        divider1 = QFrame()
        divider1.setFixedHeight(1)
        divider1.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        layout.addWidget(divider1)

        # === 连拍设置 V4.0.4 ===
        self._create_section_title(layout, self.i18n.t("advanced_settings.section_burst"))
        
        # 连拍速度
        self.vars["burst_fps"] = self._create_slider_setting(
            layout,
            self.i18n.t("advanced_settings.burst_fps"),
            self.i18n.t("advanced_settings.burst_fps_hint"),
            min_val=4, max_val=20, default=10,
            step=1,
            format_func=lambda v: f"{v} 张/秒"
        )

        # 分隔线
        divider2 = QFrame()
        divider2.setFixedHeight(1)
        divider2.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        layout.addWidget(divider2)

        # === 识鸟设置 ===
        self._create_section_title(layout, self.i18n.t("advanced_settings.section_birdid"))
        
        # 识别确信度
        self.vars["birdid_confidence"] = self._create_slider_setting(
            layout,
            self.i18n.t("advanced_settings.birdid_confidence"),
            self.i18n.t("advanced_settings.birdid_confidence_hint"),
            min_val=50, max_val=95, default=70,
            step=5,
            format_func=lambda v: f"{v}%"
        )

        # 分隔线
        divider3 = QFrame()
        divider3.setFixedHeight(1)
        divider3.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        layout.addWidget(divider3)

        # === XMP 写入 ===
        self._create_section_title(layout, self.i18n.t("advanced_settings.section_xmp"))
        xmp_row = QHBoxLayout()
        xmp_row.setSpacing(16)
        self.vars["xmp_write"] = QCheckBox()
        self.vars["xmp_write"].setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        xmp_label = QLabel(self.i18n.t("advanced_settings.xmp_write"))
        xmp_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; min-width: 80px;")
        xmp_row.addWidget(xmp_label)
        xmp_row.addWidget(self.vars["xmp_write"])
        layout.addLayout(xmp_row)
        xmp_hint = QLabel(self.i18n.t("advanced_settings.xmp_write_hint"))
        xmp_hint.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 11px;
            margin-left: 96px;
            margin-bottom: 8px;
        """)
        layout.addWidget(xmp_hint)

        layout.addStretch()

        # 底部按钮
        self._create_buttons(layout)

    def _create_section_title(self, layout, text):
        """创建区域标题"""
        label = QLabel(text)
        label.setStyleSheet(f"""
            color: {COLORS['text_tertiary']};
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 1px;
            text-transform: uppercase;
        """)
        layout.addWidget(label)

    def _create_slider_setting(self, layout, label_text, hint_text,
                               min_val, max_val, default, step=1,
                               format_func=None, scale=1):
        """创建滑块设置项"""
        # 主行：标签 + 滑块 + 值
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
            min-width: 40px;
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
        btn_layout = QHBoxLayout()

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

        # 保存
        save_btn = QPushButton(self.i18n.t("advanced_settings.save"))
        save_btn.setMinimumWidth(80)
        save_btn.clicked.connect(self._save_settings)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _load_current_config(self):
        """加载当前配置"""
        self.vars["min_confidence"].setValue(int(self.config.min_confidence * 100))
        self.vars["min_sharpness"].setValue(int(self.config.min_sharpness))
        self.vars["min_nima"].setValue(int(self.config.min_nima * 10))
        self.vars["burst_fps"].setValue(int(self.config.burst_fps))
        self.vars["birdid_confidence"].setValue(int(self.config.birdid_confidence))
        try:
            arw_mode = self.config.arw_write_mode
            self.vars["xmp_write"].setChecked(arw_mode in ("sidecar", "auto"))
        except Exception:
            self.vars["xmp_write"].setChecked(False)

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
        self.config.set_arw_write_mode("sidecar" if self.vars["xmp_write"].isChecked() else "embedded")
        self.config.set_save_csv(True)

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
        # 保存到配置
        self.config.set_skill_level(level_key)
        self.config.save()
        
        # 如果有父窗口（主窗口），更新其滑块和标签
        if self.parent() and hasattr(self.parent(), '_apply_skill_level_thresholds'):
            self.parent()._apply_skill_level_thresholds(level_key)
        
        print(f"✅ 已切换摄影水平: {level_key} (锐度={sharpness}, 美学={aesthetics})")
