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
    QProgressBar, QSizePolicy, QComboBox, QCheckBox, QSlider
)
import json
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QFont

from ui.styles import COLORS, FONTS


from tools.i18n import get_i18n

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
        self.i18n = get_i18n()
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
        # 提示文字
        hint_label = QLabel(self.i18n.t("birdid.drag_hint"))
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
            self.i18n.t("birdid.select_image"),
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
        self.i18n = get_i18n()
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

        # 名称 - 只显示当前语言
        is_en = self.i18n.current_lang.startswith('en')
        display_name = en_name if is_en else cn_name

        self.name_label = QLabel(display_name)
        self.name_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {COLORS['text_primary']};
            background: transparent;
        """)

        layout.addWidget(self.name_label, 1)

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
        self.i18n = get_i18n()
        super().__init__(self.i18n.t("birdid.title"), parent)
        self.setObjectName("BirdIDDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(280)

        # 使用自定义标题栏以控制按钮位置
        self._setup_title_bar()

        self.worker = None
        self.current_image_path = None
        self.identify_results = None
        
        # 加载区域数据和设置
        self.regions_data = self._load_regions_data()
        self.country_list = self._build_country_list()
        self.settings = self._load_settings()

        self._setup_ui()
        self._apply_settings()
    
    def _setup_title_bar(self):
        """创建自定义标题栏 - 标题靠左，按钮靠右"""
        title_bar = QWidget()
        title_bar.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_elevated']};
            }}
        """)
        
        layout = QHBoxLayout(title_bar)
        layout.setContentsMargins(12, 6, 8, 6)
        layout.setSpacing(8)
        
        # 标题文字（靠左）
        # 标题文字（靠左）
        title_label = QLabel(self.i18n.t("birdid.title"))
        title_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 13px;
            font-weight: 500;
            background: transparent;
        """)
        layout.addWidget(title_label)
        
        layout.addStretch()
        
        # 浮动按钮（靠右）
        float_btn = QPushButton("⛶")
        float_btn.setFixedSize(24, 24)
        float_btn.setToolTip(self.i18n.t("birdid.toggle_dock"))
        float_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {COLORS['text_tertiary']};
                font-size: 14px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_secondary']};
            }}
        """)
        float_btn.clicked.connect(self._toggle_floating)
        layout.addWidget(float_btn)
        
        # 关闭按钮（最右）
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setToolTip(self.i18n.t("birdid.close_panel"))
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {COLORS['text_tertiary']};
                font-size: 12px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['error']};
                color: {COLORS['text_primary']};
            }}
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setTitleBarWidget(title_bar)
    
    def _toggle_floating(self):
        """切换浮动/停靠状态"""
        self.setFloating(not self.isFloating())
    
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
        t = self.i18n.t
        country_list = {t("birdid.country_auto_gps"): None, t("birdid.country_global"): "GLOBAL"}
        
        # 优先显示的国家代码（按地区排序，每个地区内按英文名排序）
        # 大洋洲 → 美洲 → 亚洲 → 欧洲
        priority_codes = [
            # 大洋洲 (Oceania)
            'AU', 'NZ',
            # 美洲 (Americas)
            'BR', 'CA', 'CR', 'US',
            # 亚洲 (Asia)
            'CN', 'HK', 'IN', 'ID', 'JP', 'MY', 'KR', 'TW',
            # 欧洲 (Europe)
            'FR', 'DE', 'IT', 'NL', 'ES', 'GB'
        ]
        
        # 国家代码到 i18n 键的映射
        code_to_key = {
            # 大洋洲
            'AU': 'birdid.country_au', 'NZ': 'birdid.country_nz',
            # 美洲
            'BR': 'birdid.country_br', 'CA': 'birdid.country_ca', 'CR': 'birdid.country_cr', 'US': 'birdid.country_us',
            # 亚洲
            'CN': 'birdid.country_cn', 'HK': 'birdid.country_hk', 'IN': 'birdid.country_in', 'ID': 'birdid.country_id',
            'JP': 'birdid.country_jp', 'MY': 'birdid.country_my', 'KR': 'birdid.country_kr', 'TW': 'birdid.country_tw',
            # 欧洲
            'FR': 'birdid.country_fr', 'DE': 'birdid.country_de', 'IT': 'birdid.country_it',
            'NL': 'birdid.country_nl', 'ES': 'birdid.country_es', 'GB': 'birdid.country_gb'
        }
        
        # 添加优先国家（只添加有离线数据或在 regions_data 中的）
        for code in priority_codes:
            country_name = t(code_to_key.get(code, code))
            # 检查是否存在该国家数据（离线或 regions_data 中）
            if code in offline_countries:
                country_list[country_name] = code
            else:
                # 从 regions_data 查找
                for country in self.regions_data.get('countries', []):
                    if country.get('code') == code:
                        country_list[country_name] = code
                        break
        
        # 添加"更多国家..."选项
        country_list[t("birdid.country_more")] = "MORE"
        
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
            'auto_identify': False,  # 选片时自动识别，默认关闭
            'selected_country': self.i18n.t('birdid.country_auto_gps'),
            'selected_region': self.i18n.t('birdid.region_entire_country')
        }
    
    def _save_settings(self):
        """保存设置"""
        self.settings = {
            'use_ebird': self.ebird_checkbox.isChecked(),
            'auto_identify': self.auto_identify_checkbox.isChecked(),
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
        # 设置标志，防止在应用设置时触发保存
        self._applying_settings = True
        
        self.ebird_checkbox.setChecked(self.settings.get('use_ebird', True))
        self.auto_identify_checkbox.setChecked(self.settings.get('auto_identify', False))
        
        saved_country = self.settings.get('selected_country', self.i18n.t('birdid.country_auto_gps'))
        idx = self.country_combo.findText(saved_country)
        if idx >= 0:
            self.country_combo.setCurrentIndex(idx)
        
        # 等待 _on_country_changed 填充区域列表后再设置区域
        # 使用 QTimer 延迟设置
        saved_region = self.settings.get('selected_region', self.i18n.t('birdid.region_entire_country'))
        QTimer.singleShot(100, lambda: self._apply_saved_region(saved_region))
    
    def _apply_saved_region(self, saved_region: str):
        """延迟应用保存的区域设置"""
        idx = self.region_combo.findText(saved_region)
        if idx >= 0:
            self.region_combo.setCurrentIndex(idx)
        # 设置完成后解除标志
        self._applying_settings = False

    
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
        self.region_combo.addItem(self.i18n.t("birdid.region_entire_country"))
        
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
        # 只有当不是在应用设置时才保存
        if not getattr(self, '_applying_settings', False):
            self._save_settings()
        
        # 如果已有图片，重新识别（应用新的国家/地区过滤）
        self._reidentify_if_needed()

    def _on_region_changed(self, region_display: str):
        """区域选择变化时保存设置并重新识别"""
        # 如果正在更新区域列表或正在应用设置，不触发保存
        if getattr(self, '_updating_regions', False) or getattr(self, '_applying_settings', False):
            return
        
        self._save_settings()
        
        # 如果已有图片，重新识别
        self._reidentify_if_needed()

    def _show_more_countries_dialog(self):
        """显示更多国家选择对话框 - 支持国际化和搜索"""
        from PySide6.QtWidgets import QDialog, QListWidget, QDialogButtonBox, QListWidgetItem, QLineEdit
        
        t = self.i18n.t
        is_english = self.i18n.current_lang.startswith('en')
        
        dialog = QDialog(self)
        dialog.setWindowTitle(t("birdid.country_dialog_title"))
        dialog.setMinimumSize(320, 450)
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_primary']};
            }}
            QLineEdit {{
                background-color: {COLORS['bg_input']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 6px;
                padding: 8px;
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border-color: {COLORS['accent']};
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
        layout.setSpacing(8)
        
        # 搜索框
        search_input = QLineEdit()
        search_input.setPlaceholderText(t("birdid.search_country_placeholder"))
        layout.addWidget(search_input)
        
        list_widget = QListWidget()
        
        # 根据当前语言决定显示格式
        all_countries = []
        for country in self.regions_data.get('countries', []):
            code = country.get('code', '')
            name_en = country.get('name', '')
            name_cn = country.get('name_cn', '')
            
            if is_english:
                # 英文环境：只显示英文名
                display = name_en
            else:
                # 中文环境：只显示中文名（如无中文则显示英文）
                display = name_cn if name_cn else name_en
            
            # 按英文名排序以保持一致性
            sort_key = name_en.lower()
            all_countries.append((display, code, sort_key, name_en))
        
        all_countries.sort(key=lambda x: x[2])
        
        for display, code, _, name_en in all_countries:
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, code)
            item.setData(Qt.UserRole + 1, name_en)  # 用于搜索
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        
        # 搜索过滤功能
        def filter_countries(text):
            text = text.lower()
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                display_name = item.text().lower()
                en_name = (item.data(Qt.UserRole + 1) or "").lower()
                visible = text in display_name or text in en_name
                item.setHidden(not visible)
        
        search_input.textChanged.connect(filter_countries)
        
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
                    idx = self.country_combo.findText(t("birdid.country_more"))
                    if idx >= 0:
                        self.country_combo.insertItem(idx, display)
                        self.country_list[display] = code
                self.country_combo.setCurrentText(display)
        else:
            # 用户取消，恢复到之前的选择
            saved = self.settings.get('selected_country', t('birdid.country_auto_gps'))
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
        country_label = QLabel(self.i18n.t("birdid.country"))
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
        region_label = QLabel(self.i18n.t("birdid.region"))
        region_label.setStyleSheet(f"""
            color: {COLORS['text_tertiary']};
            font-size: 11px;
        """)
        region_row.addWidget(region_label)
        
        self.region_combo = QComboBox()
        self.region_combo.addItem(self.i18n.t("birdid.region_entire_country"))
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
        
        # V4.2: 移除 eBird 过滤开关（默认启用，选择"全球"可禁用）
        # V4.2: 移除自动识别开关（已移到主界面的"识鸟"按钮）
        # 保留隐藏的 checkbox 以兼容设置保存/加载
        self.ebird_checkbox = QCheckBox()
        self.ebird_checkbox.setChecked(True)  # 默认启用
        self.ebird_checkbox.hide()
        
        self.auto_identify_checkbox = QCheckBox()
        self.auto_identify_checkbox.setChecked(False)
        self.auto_identify_checkbox.hide()
        
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

        self.results_title = QLabel(self.i18n.t("birdid.results"))
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
        self.btn_new = QPushButton(self.i18n.t("birdid.btn_select"))
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



        layout.addLayout(btn_layout)

        # 状态标签（隐藏，保留变量用于内部状态追踪）
        self.status_label = QLabel("")
        self.status_label.hide()

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
            if region_display != self.i18n.t("birdid.region_entire_country"):
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

    def update_crop_preview(self, debug_img):
        """
        V4.2: 接收选片过程中的裁剪预览图像并显示
        Args:
            debug_img: BGR numpy 数组 (带标注的鸟类裁剪图)
        """
        try:
            import cv2
            from PySide6.QtGui import QImage
            
            # BGR -> RGB
            rgb_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            
            # numpy -> QImage -> QPixmap
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # 保存并显示
            self._current_pixmap = pixmap
            self.preview_label.show()
            self._scale_preview()
            
        except Exception as e:
            print(f"[BirdIDDock] 预览更新失败: {e}")

    def show_completion_message(self, debug_dir: str):
        """
        V4.2: 处理完成后显示目录路径，隐藏预览图
        Args:
            debug_dir: debug_crops 目录路径
        """
        # 隐藏预览图
        self.preview_label.hide()
        self._current_pixmap = None
        
        # 清空结果并显示完成信息
        self.clear_results()
        
        # 创建完成信息标签
        from PySide6.QtWidgets import QLabel
        
        info_label = QLabel(f"✅ 分析完成\n\n📁 调试图目录:\n{debug_dir}")
        info_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 12px;
            padding: 16px;
            background-color: {COLORS['bg_elevated']};
            border-radius: 8px;
        """)
        info_label.setWordWrap(True)
        self.results_layout.addWidget(info_label)

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



    def reset_view(self):
        """重置视图"""
        self.drop_area.show()
        self.preview_label.hide()
        self.filename_label.hide()
        self.results_frame.hide()

        self.status_label.setText("准备就绪")
        self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text_muted']};")
        self.current_image_path = None
        self.identify_results = None
        self._current_pixmap = None
        self.clear_results()
