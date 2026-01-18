#!/usr/bin/env python3
"""
鸟类识别停靠面板
可停靠在主窗口边缘的识鸟功能面板
风格与 SuperPicky 主窗口统一
"""

import os
import sys

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QFileDialog,
    QProgressBar, QSizePolicy, QComboBox, QCheckBox
)
import json
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QFont

from ui.styles import COLORS, FONTS


def get_birdid_data_path(relative_path: str) -> str:
    """获取 birdid/data 目录下的资源路径"""
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'birdid', 'data', relative_path)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'birdid', 'data', relative_path)


def get_settings_path() -> str:
    """获取设置文件路径"""
    if sys.platform == 'darwin':
        settings_dir = os.path.expanduser('~/Documents/SuperPicky_Data')
    else:
        settings_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'SuperPicky_Data')
    os.makedirs(settings_dir, exist_ok=True)
    return os.path.join(settings_dir, 'birdid_dock_settings.json')


class IdentifyWorker(QThread):
    """后台识别线程"""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, image_path: str, top_k: int = 5,
                 use_gps: bool = True, use_ebird: bool = True,
                 country_code: str = None, region_code: str = None):
        super().__init__()
        self.image_path = image_path
        self.top_k = top_k
        self.use_gps = use_gps
        self.use_ebird = use_ebird
        self.country_code = country_code
        self.region_code = region_code

    def run(self):
        try:
            from birdid.bird_identifier import identify_bird
            result = identify_bird(
                self.image_path,
                top_k=self.top_k,
                use_gps=self.use_gps,
                use_ebird=self.use_ebird,
                country_code=self.country_code,
                region_code=self.region_code
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class DropArea(QFrame):
    """拖放区域 - 深色主题"""
    fileDropped = Signal(str)
    imageDropped = Signal(object)  # 直接传递 QImage 对象

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumSize(250, 160)
        self.setStyleSheet(f"""
            DropArea {{
                border: 2px dashed {COLORS['border']};
                border-radius: 10px;
                background-color: {COLORS['bg_elevated']};
            }}
            DropArea:hover {{
                border-color: {COLORS['accent']};
                background-color: {COLORS['bg_card']};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(8)

        # 图标 - 使用 + 号
        icon_label = QLabel("+")
        icon_label.setStyleSheet(f"""
            font-size: 48px;
            font-weight: 300;
            color: {COLORS['text_tertiary']};
            background: transparent;
        """)
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        # 提示文字
        hint_label = QLabel("拖放/粘贴图片\n或点击选择")
        hint_label.setAlignment(Qt.AlignCenter)
        hint_label.setStyleSheet(f"""
            color: {COLORS['text_tertiary']};
            font-size: 13px;
            background: transparent;
        """)
        layout.addWidget(hint_label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() or event.mimeData().hasImage():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        mime = event.mimeData()
        if mime.hasUrls():
            urls = mime.urls()
            if urls:
                file_path = urls[0].toLocalFile()
                self.fileDropped.emit(file_path)
        elif mime.hasImage():
            image = mime.imageData()
            if image:
                self.imageDropped.emit(image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selectFile()

    def selectFile(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择鸟类图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.nef *.cr2 *.cr3 *.arw *.raf *.orf *.rw2 *.dng);;所有文件 (*)"
        )
        if file_path:
            self.fileDropped.emit(file_path)


class DropPreviewLabel(QLabel):
    """支持拖放的图片预览标签"""
    fileDropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.fileDropped.emit(file_path)


class ResultCard(QFrame):
    """识别结果卡片 - 深色主题，可点击选中"""
    
    clicked = Signal(int)  # 发送排名信号

    def __init__(self, rank: int, cn_name: str, en_name: str, confidence: float):
        super().__init__()
        self.rank = rank
        self.cn_name = cn_name
        self.en_name = en_name
        self.confidence = confidence
        self._selected = False
        
        self.setCursor(Qt.PointingHandCursor)
        self._update_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        # 排名
        self.rank_label = QLabel(f"#{rank}")
        self.rank_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {COLORS['accent']};
            min-width: 28px;
            background: transparent;
        """)
        layout.addWidget(self.rank_label)

        # 名称
        name_layout = QVBoxLayout()
        name_layout.setSpacing(2)

        # 中文名（主要显示）
        self.cn_label = QLabel(cn_name)
        self.cn_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {COLORS['text_primary']};
            background: transparent;
        """)
        name_layout.addWidget(self.cn_label)

        # 英文名（次要显示）
        self.en_label = QLabel(en_name)
        self.en_label.setStyleSheet(f"""
            font-size: 11px;
            color: {COLORS['text_tertiary']};
            background: transparent;
        """)
        name_layout.addWidget(self.en_label)

        layout.addLayout(name_layout, 1)

        # 置信度
        if confidence >= 70:
            conf_color = COLORS['success']
        elif confidence >= 40:
            conf_color = COLORS['warning']
        else:
            conf_color = COLORS['error']

        self.conf_label = QLabel(f"{confidence:.0f}%")
        self.conf_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {conf_color};
            font-family: {FONTS['mono']};
            background: transparent;
        """)
        layout.addWidget(self.conf_label)
    
    def _update_style(self):
        """更新选中/未选中样式"""
        if self._selected:
            self.setStyleSheet(f"""
                ResultCard {{
                    background-color: {COLORS['bg_card']};
                    border: 2px solid {COLORS['accent']};
                    border-radius: 8px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                ResultCard {{
                    background-color: {COLORS['bg_card']};
                    border: 1px solid {COLORS['border_subtle']};
                    border-radius: 8px;
                }}
                ResultCard:hover {{
                    border: 1px solid {COLORS['text_muted']};
                }}
            """)
    
    def set_selected(self, selected: bool):
        """设置选中状态"""
        self._selected = selected
        self._update_style()
    
    def is_selected(self):
        return self._selected
    
    def mousePressEvent(self, event):
        """点击事件"""
        self.clicked.emit(self.rank)
        super().mousePressEvent(event)


class BirdIDDockWidget(QDockWidget):
    """鸟类识别停靠面板 - 深色主题"""

    def __init__(self, parent=None):
        super().__init__("鸟类识别", parent)
        self.setObjectName("BirdIDDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(280)

        # 设置 Dock 标题栏样式
        self.setStyleSheet(f"""
            QDockWidget {{
                color: {COLORS['text_primary']};
                font-size: 13px;
                font-weight: 500;
            }}
            QDockWidget::title {{
                background-color: {COLORS['bg_elevated']};
                padding: 8px;
                text-align: left;
            }}
        """)

        self.worker = None
        self.current_image_path = None
        self.identify_results = None
        
        # 加载区域数据和设置
        self.regions_data = self._load_regions_data()
        self.country_list = self._build_country_list()
        self.settings = self._load_settings()

        self._setup_ui()
        self._apply_settings()
    
    def _load_regions_data(self) -> dict:
        """加载 eBird 区域数据"""
        regions_path = get_birdid_data_path('ebird_regions.json')
        if os.path.exists(regions_path):
            try:
                with open(regions_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载区域数据失败: {e}")
        return {'countries': []}
    
    def _build_country_list(self) -> dict:
        """构建国家列表 {显示名称: 代码}
        
        只显示有离线数据的优先国家，其他国家归入"更多国家..."选项
        """
        # 加载离线数据索引，获取有离线数据的国家代码
        offline_index_path = get_birdid_data_path('offline_ebird_data/offline_index.json')
        offline_countries = set()
        if os.path.exists(offline_index_path):
            try:
                with open(offline_index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                offline_countries = set(index_data.get('countries', {}).keys())
            except:
                pass
        
        # 特殊选项
        country_list = {"自动检测 (GPS)": None, "全球模式": "GLOBAL"}
        
        # 优先显示的国家代码（按顺序）- 只显示这些
        priority_codes = ['AU', 'US', 'GB', 'CN', 'HK', 'TW', 'JP']
        
        # 国家代码到中文名的映射
        code_to_cn = {
            'AU': '澳大利亚', 'US': '美国', 'GB': '英国', 'CN': '中国',
            'HK': '香港', 'TW': '台湾', 'JP': '日本'
        }
        
        # 添加优先国家（只添加有离线数据或在 regions_data 中的）
        for code in priority_codes:
            cn_name = code_to_cn.get(code, code)
            # 检查是否存在该国家数据（离线或 regions_data 中）
            if code in offline_countries:
                country_list[cn_name] = code
            else:
                # 从 regions_data 查找
                for country in self.regions_data.get('countries', []):
                    if country.get('code') == code:
                        country_list[cn_name] = code
                        break
        
        # 添加"更多国家..."选项
        country_list["── 更多国家 ──"] = "MORE"
        
        return country_list
    
    def _load_settings(self) -> dict:
        """加载设置"""
        settings_path = get_settings_path()
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            'use_ebird': True,
            'selected_country': '自动检测 (GPS)',
            'selected_region': '整个国家'
        }
    
    def _save_settings(self):
        """保存设置"""
        self.settings = {
            'use_ebird': self.ebird_checkbox.isChecked(),
            'selected_country': self.country_combo.currentText(),
            'selected_region': self.region_combo.currentText()
        }
        try:
            settings_path = get_settings_path()
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存设置失败: {e}")
    
    def _apply_settings(self):
        """应用保存的设置"""
        self.ebird_checkbox.setChecked(self.settings.get('use_ebird', True))
        
        saved_country = self.settings.get('selected_country', '自动检测 (GPS)')
        idx = self.country_combo.findText(saved_country)
        if idx >= 0:
            self.country_combo.setCurrentIndex(idx)
        
        saved_region = self.settings.get('selected_region', '整个国家')
        idx = self.region_combo.findText(saved_region)
        if idx >= 0:
            self.region_combo.setCurrentIndex(idx)
    
    def _on_country_changed(self, country_display: str):
        """国家选择变化时更新区域列表"""
        country_code = self.country_list.get(country_display)
        
        # 处理"更多国家"选项
        if country_code == "MORE":
            self._show_more_countries_dialog()
            return
        
        # 设置标志，防止在填充区域列表时触发 _on_region_changed
        self._updating_regions = True
        
        self.region_combo.clear()
        self.region_combo.addItem("整个国家")
        
        if country_code and country_code != "GLOBAL":
            # 查找该国家的区域列表
            for country in self.regions_data.get('countries', []):
                if country.get('code') == country_code:
                    if country.get('has_regions') and country.get('regions'):
                        for region in country['regions']:
                            region_name = region.get('name', '')
                            region_code = region.get('code', '')
                            self.region_combo.addItem(f"{region_name} ({region_code})")
                    break
        
        self._updating_regions = False
        self._save_settings()
        
        # 如果已有图片，重新识别（应用新的国家/地区过滤）
        self._reidentify_if_needed()

    def _on_region_changed(self, region_display: str):
        """区域选择变化时保存设置并重新识别"""
        # 如果正在更新区域列表，不触发重新识别
        if getattr(self, '_updating_regions', False):
            return
        
        self._save_settings()
        
        # 如果已有图片，重新识别
        self._reidentify_if_needed()

    def _show_more_countries_dialog(self):
        """显示更多国家选择对话框"""
        from PySide6.QtWidgets import QDialog, QListWidget, QDialogButtonBox, QListWidgetItem
        
        dialog = QDialog(self)
        dialog.setWindowTitle("选择国家")
        dialog.setMinimumSize(300, 400)
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_primary']};
            }}
            QListWidget {{
                background-color: {COLORS['bg_elevated']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 6px;
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QListWidget::item {{
                padding: 8px;
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['accent']};
                color: {COLORS['bg_void']};
            }}
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        
        list_widget = QListWidget()
        
        # 添加所有国家（按英文名排序）
        all_countries = []
        for country in self.regions_data.get('countries', []):
            code = country.get('code', '')
            name = country.get('name', '')
            name_cn = country.get('name_cn', '')
            if name_cn:
                display = f"{name_cn} ({name})"
            else:
                display = name
            all_countries.append((display, code, name))
        
        all_countries.sort(key=lambda x: x[2].lower())
        
        for display, code, _ in all_countries:
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, code)
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec() == QDialog.Accepted:
            selected = list_widget.currentItem()
            if selected:
                code = selected.data(Qt.UserRole)
                display = selected.text()
                # 添加到列表并选中
                existing = [self.country_combo.itemText(i) for i in range(self.country_combo.count())]
                if display not in existing:
                    # 在"更多国家"之前插入
                    idx = self.country_combo.findText("── 更多国家 ──")
                    if idx >= 0:
                        self.country_combo.insertItem(idx, display)
                        self.country_list[display] = code
                self.country_combo.setCurrentText(display)
        else:
            # 用户取消，恢复到之前的选择
            saved = self.settings.get('selected_country', '自动检测 (GPS)')
            self.country_combo.setCurrentText(saved)

    def _setup_ui(self):
        """设置界面"""
        container = QWidget()
        container.setStyleSheet(f"background-color: {COLORS['bg_primary']};")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # 拖放区域
        self.drop_area = DropArea()
        self.drop_area.fileDropped.connect(self.on_file_dropped)
        self.drop_area.imageDropped.connect(self.on_image_pasted)
        layout.addWidget(self.drop_area)
        
        # ===== 国家/区域过滤 =====
        filter_frame = QFrame()
        filter_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_elevated']};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        filter_layout = QVBoxLayout(filter_frame)
        filter_layout.setContentsMargins(8, 8, 8, 8)
        filter_layout.setSpacing(6)
        
        # 国家选择行
        country_row = QHBoxLayout()
        country_label = QLabel("国家:")
        country_label.setStyleSheet(f"""
            color: {COLORS['text_tertiary']};
            font-size: 11px;
        """)
        country_row.addWidget(country_label)
        
        self.country_combo = QComboBox()
        self.country_combo.addItems(list(self.country_list.keys()))
        self.country_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['bg_input']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
                padding: 4px 8px;
                color: {COLORS['text_secondary']};
                font-size: 11px;
            }}
            QComboBox:hover {{
                border-color: {COLORS['accent']};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
        """)
        self.country_combo.currentTextChanged.connect(self._on_country_changed)
        country_row.addWidget(self.country_combo, 1)
        filter_layout.addLayout(country_row)
        
        # 区域选择行
        region_row = QHBoxLayout()
        region_label = QLabel("区域:")
        region_label.setStyleSheet(f"""
            color: {COLORS['text_tertiary']};
            font-size: 11px;
        """)
        region_row.addWidget(region_label)
        
        self.region_combo = QComboBox()
        self.region_combo.addItem("整个国家")
        self.region_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['bg_input']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
                padding: 4px 8px;
                color: {COLORS['text_secondary']};
                font-size: 11px;
            }}
            QComboBox:hover {{
                border-color: {COLORS['accent']};
            }}
        """)
        self.region_combo.currentTextChanged.connect(self._on_region_changed)
        region_row.addWidget(self.region_combo, 1)
        filter_layout.addLayout(region_row)
        
        # eBird 过滤开关
        self.ebird_checkbox = QCheckBox("启用 eBird 过滤")
        self.ebird_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {COLORS['text_tertiary']};
                font-size: 11px;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['accent']};
                border-radius: 3px;
            }}
        """)
        self.ebird_checkbox.stateChanged.connect(self._save_settings)
        filter_layout.addWidget(self.ebird_checkbox)
        
        layout.addWidget(filter_frame)

        # 图片预览（初始隐藏，支持拖放替换）
        self.preview_label = DropPreviewLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(100)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.preview_label.setStyleSheet(f"""
            background-color: {COLORS['bg_elevated']};
            border-radius: 10px;
            padding: 8px;
        """)
        self.preview_label.fileDropped.connect(self.on_file_dropped)
        self.preview_label.hide()
        self._current_pixmap = None  # 保存原始 pixmap 用于自适应缩放
        layout.addWidget(self.preview_label)

        # 文件名显示
        self.filename_label = QLabel()
        self.filename_label.setStyleSheet(f"""
            font-size: 11px;
            color: {COLORS['text_tertiary']};
            font-family: {FONTS['mono']};
        """)
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setWordWrap(True)
        self.filename_label.hide()
        layout.addWidget(self.filename_label)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(3)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['bg_input']};
                border-radius: 2px;
                max-height: 3px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['accent']}, stop:1 #00ffcc);
                border-radius: 2px;
            }}
        """)
        self.progress.hide()
        layout.addWidget(self.progress)

        # 结果区域
        self.results_frame = QFrame()
        self.results_frame.setStyleSheet(f"""
            QFrame {{
                background-color: transparent;
            }}
        """)
        results_layout = QVBoxLayout(self.results_frame)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(6)

        self.results_title = QLabel("识别结果")
        self.results_title.setStyleSheet(f"""
            font-size: 11px;
            font-weight: 500;
            color: {COLORS['text_tertiary']};
            text-transform: uppercase;
            letter-spacing: 1px;
        """)
        results_layout.addWidget(self.results_title)

        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setMaximumHeight(350)  # 足够显示3-4个候选
        self.results_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: transparent;
            }}
        """)

        self.results_widget = QWidget()
        self.results_widget.setStyleSheet("background: transparent;")
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(6)
        self.results_scroll.setWidget(self.results_widget)

        results_layout.addWidget(self.results_scroll)
        self.results_frame.hide()
        layout.addWidget(self.results_frame)

        # 操作按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        # 选择图片按钮 - 次级样式
        self.btn_new = QPushButton("选择图片")
        self.btn_new.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                color: {COLORS['text_secondary']};
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                border-color: {COLORS['text_muted']};
                color: {COLORS['text_primary']};
            }}
        """)
        self.btn_new.clicked.connect(self.drop_area.selectFile)
        btn_layout.addWidget(self.btn_new)

        # 写入 EXIF 按钮 - 主按钮样式（青绿色）
        self.btn_write_exif = QPushButton("写入 EXIF")
        self.btn_write_exif.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                border: none;
                color: {COLORS['bg_void']};
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #00e6b8;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_muted']};
            }}
        """)
        self.btn_write_exif.clicked.connect(self.write_exif)
        self.btn_write_exif.setEnabled(False)
        btn_layout.addWidget(self.btn_write_exif)

        layout.addLayout(btn_layout)

        # 状态标签
        self.status_label = QLabel("准备就绪")
        self.status_label.setStyleSheet(f"""
            font-size: 11px;
            color: {COLORS['text_muted']};
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setWidget(container)

    def on_image_pasted(self, image):
        """处理剪贴板粘贴的图片"""
        from PySide6.QtGui import QImage
        from PIL import Image
        import tempfile
        import time
        
        print(f"[调试] on_image_pasted 被调用, image类型: {type(image)}")
        
        if isinstance(image, QImage) and not image.isNull():
            print(f"[调试] 图片尺寸: {image.width()}x{image.height()}, 格式: {image.format()}")
            
            try:
                # 使用 PIL 保存（避免 Qt 6.10 在 macOS 上的崩溃 bug）
                # 先将 QImage 转换为 bytes
                width = image.width()
                height = image.height()
                
                # 转换为 RGBA 格式
                if image.format() != QImage.Format.Format_RGBA8888:
                    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
                
                # 获取原始数据
                ptr = image.bits()
                if ptr is None:
                    print("[调试] 无法获取图片数据")
                    self.status_label.setText("无法读取图片数据")
                    self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['error']};")
                    return
                
                # 创建 PIL Image
                pil_image = Image.frombytes('RGBA', (width, height), bytes(ptr))
                
                # 保存为临时文件
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"superpicky_paste_{int(time.time())}.png")
                
                print(f"[调试] 尝试用 PIL 保存到: {temp_path}")
                pil_image.save(temp_path, "PNG")
                print(f"[调试] PIL 保存成功")
                
                self.current_image_path = temp_path
                self.status_label.setText("正在识别...")
                self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['accent']};")
                
                # 显示文件名
                self.filename_label.setText("剪贴板图片")
                self.filename_label.show()
                
                # 显示预览（从保存的文件加载，避免 QImage 问题）
                self.show_preview(temp_path)
                
                # 清空之前的结果
                self.clear_results()
                
                # 显示进度
                self.progress.show()
                self.results_frame.hide()
                self.btn_write_exif.setEnabled(False)
                
                # 获取过滤设置并启动识别
                self._start_identify(temp_path)
                
            except Exception as e:
                print(f"[调试] 保存失败: {e}")
                import traceback
                traceback.print_exc()
                self.status_label.setText(f"保存图片失败: {e}")
                self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['error']};")
        else:
            print(f"[调试] 无效图片或为空")
            self.status_label.setText("无效的图片")
            self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['error']};")

    def _show_qimage_preview(self, qimage):
        """显示 QImage 预览"""
        from PySide6.QtGui import QImage
        
        pixmap = QPixmap.fromImage(qimage)
        if not pixmap.isNull():
            self._current_pixmap = pixmap
            self.drop_area.hide()
            self.preview_label.show()
            QTimer.singleShot(50, self._scale_preview)

    def on_file_dropped(self, file_path: str):
        """处理文件拖放"""
        if not os.path.exists(file_path):
            self.status_label.setText("文件不存在")
            self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['error']};")
            return

        self.current_image_path = file_path
        self.status_label.setText("正在识别...")
        self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['accent']};")

        # 显示文件名
        filename = os.path.basename(file_path)
        self.filename_label.setText(filename)
        self.filename_label.show()

        # 显示预览
        self.show_preview(file_path)

        # 清空之前的结果
        self.clear_results()

        # 显示进度
        self.progress.show()
        self.results_frame.hide()
        self.btn_write_exif.setEnabled(False)
        
        # 启动识别
        self._start_identify(file_path)

    def _reidentify_if_needed(self):
        """当国家/地区改变时，如果有当前图片，重新识别"""
        if hasattr(self, 'current_image_path') and self.current_image_path:
            if os.path.exists(self.current_image_path):
                print(f"[调试] 国家/地区已改变，重新识别: {self.current_image_path}")
                self.status_label.setText("正在重新识别...")
                self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['accent']};")
                
                # 清空之前的结果
                self.clear_results()
                
                # 显示进度
                self.progress.show()
                self.results_frame.hide()
                self.btn_write_exif.setEnabled(False)
                
                # 重新启动识别
                self._start_identify(self.current_image_path)

    def _start_identify(self, file_path: str):
        """启动识别（供文件拖放和粘贴共用）"""
        # 如果有正在运行的识别任务，先等待它完成或断开连接
        if hasattr(self, 'worker') and self.worker is not None:
            try:
                self.worker.finished.disconnect()
                self.worker.error.disconnect()
            except:
                pass
            if self.worker.isRunning():
                self.worker.wait(1000)  # 最多等待1秒
            self.worker = None
        
        # 获取过滤设置
        use_ebird = self.ebird_checkbox.isChecked()
        use_gps = True  # GPS 自动检测始终启用
        
        country_code = None
        region_code = None
        
        country_display = self.country_combo.currentText()
        country_code_raw = self.country_list.get(country_display)
        
        if country_code_raw and country_code_raw not in ("GLOBAL", "MORE"):
            country_code = country_code_raw
            
            # 检查是否选择了具体区域
            region_display = self.region_combo.currentText()
            if region_display != "整个国家":
                # 从 "South Australia (AU-SA)" 提取 AU-SA
                import re
                match = re.search(r'\(([A-Z]{2}-[A-Z0-9]+)\)', region_display)
                if match:
                    region_code = match.group(1)

        # 启动识别
        self.worker = IdentifyWorker(
            file_path,
            top_k=5,
            use_gps=use_gps,
            use_ebird=use_ebird,
            country_code=country_code,
            region_code=region_code
        )
        self.worker.finished.connect(self.on_identify_finished)
        self.worker.error.connect(self.on_identify_error)
        self.worker.start()

    def show_preview(self, file_path: str):
        """显示图片预览"""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            raw_extensions = ['.nef', '.cr2', '.cr3', '.arw', '.raf', '.orf', '.rw2', '.dng']

            if ext in raw_extensions:
                from birdid.bird_identifier import load_image
                pil_image = load_image(file_path)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    pil_image.save(tmp.name, 'JPEG', quality=85)
                    pixmap = QPixmap(tmp.name)
                    os.unlink(tmp.name)
            else:
                pixmap = QPixmap(file_path)

            if not pixmap.isNull():
                self._current_pixmap = pixmap
                self.drop_area.hide()
                self.preview_label.show()
                # 延迟缩放，确保布局完成
                QTimer.singleShot(50, self._scale_preview)
        except Exception as e:
            print(f"预览加载失败: {e}")

    def _scale_preview(self):
        """根据面板宽度缩放预览图"""
        if self._current_pixmap is None:
            return
        # 获取容器宽度（减去边距和 padding）
        container = self.widget()
        if container:
            available_width = container.width() - 24 - 16  # 边距 + padding
        else:
            available_width = self.width() - 40
        if available_width < 100:
            available_width = 256
        # 限制最大高度
        max_height = 280
        scaled = self._current_pixmap.scaled(
            available_width, max_height,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event):
        """面板大小变化时重新缩放预览图"""
        super().resizeEvent(event)
        if self._current_pixmap is not None and self.preview_label.isVisible():
            self._scale_preview()

    def clear_results(self):
        """清空结果区域"""
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def on_identify_finished(self, result: dict):
        """识别完成"""
        self.progress.hide()

        if not result.get('success'):
            self.status_label.setText("识别失败")
            self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['error']};")
            return

        results = result.get('results', [])
        if not results:
            self.status_label.setText("未能识别")
            self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['warning']};")
            return

        # 显示结果
        self.results_frame.show()
        self.result_cards = []  # 保存卡片引用
        self.selected_index = 0  # 默认选中第一个

        for i, r in enumerate(results, 1):
            card = ResultCard(
                rank=i,
                cn_name=r.get('cn_name', '未知'),
                en_name=r.get('en_name', 'Unknown'),
                confidence=r.get('confidence', 0)
            )
            # 连接点击信号
            card.clicked.connect(self.on_result_card_clicked)
            # 默认选中第一个
            if i == 1:
                card.set_selected(True)
            self.result_cards.append(card)
            self.results_layout.addWidget(card)

        self.results_layout.addStretch()

        # 保存结果
        self.identify_results = results
        self.btn_write_exif.setEnabled(True)

        # 状态显示选中的候选
        self._update_status_label()

    def on_identify_error(self, error_msg: str):
        """识别出错"""
        self.progress.hide()
        self.status_label.setText(f"错误: {error_msg[:30]}")
        self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['error']};")
    
    def on_result_card_clicked(self, rank: int):
        """点击结果卡片，切换选中状态"""
        # rank 从 1 开始，转为 0-based index
        index = rank - 1
        if index < 0 or index >= len(self.result_cards):
            return
        
        # 取消之前选中的
        if hasattr(self, 'result_cards'):
            for card in self.result_cards:
                card.set_selected(False)
        
        # 选中当前点击的
        self.result_cards[index].set_selected(True)
        self.selected_index = index
        
        # 更新状态标签
        self._update_status_label()
    
    def _update_status_label(self):
        """更新状态标签，显示当前选中的候选"""
        if hasattr(self, 'selected_index') and hasattr(self, 'identify_results'):
            if 0 <= self.selected_index < len(self.identify_results):
                selected = self.identify_results[self.selected_index]
                self.status_label.setText(f"✓ {selected['cn_name']} ({selected['confidence']:.0f}%)")
                self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['success']};")

    def write_exif(self):
        """写入 EXIF - 使用选中的候选"""
        if not self.current_image_path or not self.identify_results:
            return

        # 使用选中的候选（默认是第一个）
        selected_index = getattr(self, 'selected_index', 0)
        if selected_index >= len(self.identify_results):
            selected_index = 0
        
        selected = self.identify_results[selected_index]
        # 使用中文名为主，英文名为辅
        bird_name = f"{selected['cn_name']} ({selected['en_name']})"

        try:
            from exiftool_manager import get_exiftool_manager
            exiftool_mgr = get_exiftool_manager()
            success = exiftool_mgr.set_metadata(self.current_image_path, {'Title': bird_name})

            if success:
                self.status_label.setText(f"已写入: {selected['cn_name']}")
                self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['success']};")
            else:
                self.status_label.setText("EXIF 写入失败")
                self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['error']};")
        except Exception as e:
            self.status_label.setText(f"错误: {str(e)[:20]}")
            self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['error']};")

    def reset_view(self):
        """重置视图"""
        self.drop_area.show()
        self.preview_label.hide()
        self.filename_label.hide()
        self.results_frame.hide()
        self.btn_write_exif.setEnabled(False)
        self.status_label.setText("准备就绪")
        self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text_muted']};")
        self.current_image_path = None
        self.identify_results = None
        self._current_pixmap = None
        self.clear_results()
