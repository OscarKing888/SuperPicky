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
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        mime = event.mimeData()
        if mime.hasUrls():
            urls = mime.urls()
            if urls:
                file_path = urls[0].toLocalFile()
                self.fileDropped.emit(file_path)


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
        self.conf_label.setFixedWidth(50)
        self.conf_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
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

        V4.4: 简化下拉菜单，只显示约 15 项
        - 自动定位 (Auto GPS)
        - 全球模式 (Global)
        - 分隔符
        - Top 10 常用国家 (按英文首字母 A-Z)
        - 分隔符
        - "更多国家..." 选项
        """
        from collections import OrderedDict

        t = self.i18n.t
        is_english = self.i18n.current_lang.startswith('en')

        # 使用 OrderedDict 保持插入顺序
        country_list = OrderedDict()

        # === 第一部分：特殊选项 ===
        country_list[t("birdid.country_auto_gps")] = None
        country_list[t("birdid.country_global")] = "GLOBAL"

        # === 分隔符 1 ===
        country_list["─" * 15] = "SEP1"

        # === 第二部分：Top 10 常用国家 (按英文首字母 A-Z 排序) ===
        top10_codes = ['AU', 'BR', 'CN', 'GB', 'HK', 'ID', 'JP', 'MY', 'TW', 'US']

        # 国家代码到 i18n 键的映射 (Top 10)
        top10_i18n = {
            'AU': 'birdid.country_au',
            'BR': 'birdid.country_br',
            'CN': 'birdid.country_cn',
            'GB': 'birdid.country_gb',
            'HK': 'birdid.country_hk',
            'ID': 'birdid.country_id',
            'JP': 'birdid.country_jp',
            'MY': 'birdid.country_my',
            'TW': 'birdid.country_tw',
            'US': 'birdid.country_us',
        }

        # 构建 code -> region_data 映射
        code_to_region = {}
        for region in self.regions_data.get('countries', []):
            code_to_region[region.get('code')] = region

        # 添加 Top 10 (已按英文首字母排序)
        for code in top10_codes:
            i18n_key = top10_i18n.get(code)
            if i18n_key:
                display_name = t(i18n_key)
            else:
                # 回退到 regions_data
                region = code_to_region.get(code, {})
                if is_english:
                    display_name = region.get('name', code)
                else:
                    display_name = region.get('name_cn') or region.get('name', code)
            country_list[display_name] = code

        # === 分隔符 2 ===
        country_list["─" * 15 + " "] = "SEP2"  # 添加空格使 key 不同

        # === "更多国家..." 选项 ===
        country_list[t("birdid.country_more")] = "MORE"

        return country_list

    def _populate_country_combo(self):
        """填充国家下拉菜单，并禁用分隔符项"""
        from PySide6.QtGui import QStandardItem
        from PySide6.QtWidgets import QStyledItemDelegate

        self.country_combo.clear()

        for display_name, code in self.country_list.items():
            self.country_combo.addItem(display_name)

            # 如果是分隔符，禁用该项
            if code in ("SEP1", "SEP2"):
                idx = self.country_combo.count() - 1
                # 获取模型中的 item 并设置为不可选
                model = self.country_combo.model()
                item = model.item(idx)
                if item:
                    item.setEnabled(False)
                    # 设置分隔符样式
                    item.setSelectable(False)

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
        # V4.0.4: 同时保存 country_code，避免读取时需要硬编码映射
        country_display = self.country_combo.currentText()
        country_code = self.country_list.get(country_display)
        
        # 解析 region_code
        region_display = self.region_combo.currentText()
        region_code = None
        if region_display and region_display != self.i18n.t('birdid.region_entire_country'):
            import re
            match = re.search(r'\(([A-Z]{2}-[A-Z0-9]+)\)', region_display)
            if match:
                region_code = match.group(1)
        
        self.settings = {
            'use_ebird': self.ebird_checkbox.isChecked(),
            'auto_identify': self.auto_identify_checkbox.isChecked(),
            'selected_country': country_display,
            'country_code': country_code,  # V4.0.4: 直接存储代码
            'selected_region': region_display,
            'region_code': region_code  # V4.0.4: 直接存储代码
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
        
        # V4.0.4: 优先使用 country_code 来匹配，而不是 selected_country 文本
        country_code = self.settings.get('country_code')
        saved_country = self.settings.get('selected_country', self.i18n.t('birdid.country_auto_gps'))
        
        matched = False
        if country_code:
            # 通过 country_code 找到对应的显示名称
            for display_name, code in self.country_list.items():
                if code == country_code:
                    idx = self.country_combo.findText(display_name)
                    if idx >= 0:
                        self.country_combo.setCurrentIndex(idx)
                        matched = True
                        print(f"[DEBUG] 通过 country_code={country_code} 匹配到: {display_name}")
                    break
        
        if not matched:
            # 回退：使用文本匹配
            idx = self.country_combo.findText(saved_country)
            if idx >= 0:
                self.country_combo.setCurrentIndex(idx)
                print(f"[DEBUG] 通过文本匹配到: {saved_country}")
            else:
                # 如果都找不到，可能是从"更多国家"选的，需要动态添加
                if country_code and country_code not in [None, "GLOBAL", "MORE"]:
                    # 从 regions_data 获取国家名称
                    for country in self.regions_data.get('countries', []):
                        if country.get('code') == country_code:
                            display_name = saved_country or country.get('name_cn') or country.get('name')
                            # 添加到列表
                            t = self.i18n.t
                            more_idx = self.country_combo.findText(t("birdid.country_more"))
                            if more_idx >= 0:
                                self.country_combo.insertItem(more_idx, display_name)
                                self.country_list[display_name] = country_code
                                self.country_combo.setCurrentText(display_name)
                                print(f"[DEBUG] 动态添加国家: {display_name} ({country_code})")
                            break
        
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

        # 忽略分隔符
        if country_code in ("SEP1", "SEP2"):
            return

        # 处理"更多国家"选项 (已移除，保留兼容性)
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
        """显示更多国家选择对话框 - 显示大洲和其他国家，支持搜索

        V4.4: 只显示不在 Top 10 中的区域（大洲 + 其他国家）
        - 大洲项目前面加 🌍 前缀
        - 按英文名 A-Z 排序
        """
        from PySide6.QtWidgets import QDialog, QListWidget, QDialogButtonBox, QListWidgetItem, QLineEdit

        t = self.i18n.t
        is_english = self.i18n.current_lang.startswith('en')

        # Top 10 国家代码（已在下拉菜单中）
        top10_codes = {'AU', 'BR', 'CN', 'GB', 'HK', 'ID', 'JP', 'MY', 'TW', 'US', 'GLOBAL'}

        # 大洲代码
        continent_codes = {'AF', 'AS', 'EU', 'NA', 'SA', 'OC'}

        # 大洲 i18n 映射
        continent_i18n = {
            'AF': 'birdid.continent_af',
            'AS': 'birdid.continent_as',
            'EU': 'birdid.continent_eu',
            'NA': 'birdid.continent_na',
            'SA': 'birdid.continent_sa',
            'OC': 'birdid.continent_oc',
        }

        # 其他国家 i18n 映射
        other_country_i18n = {
            'AR': 'birdid.country_ar',
            'CA': 'birdid.country_ca',
            'CH': 'birdid.country_ch',
            'CL': 'birdid.country_cl',
            'CO': 'birdid.country_co',
            'CR': 'birdid.country_cr',
            'DE': 'birdid.country_de',
            'EC': 'birdid.country_ec',
            'EG': 'birdid.country_eg',
            'ES': 'birdid.country_es',
            'FI': 'birdid.country_fi',
            'FR': 'birdid.country_fr',
            'GR': 'birdid.country_gr',
            'IN': 'birdid.country_in',
            'IT': 'birdid.country_it',
            'KE': 'birdid.country_ke',
            'KR': 'birdid.country_kr',
            'LK': 'birdid.country_lk',
            'MA': 'birdid.country_ma',
            'MG': 'birdid.country_mg',
            'MN': 'birdid.country_mn',
            'MX': 'birdid.country_mx',
            'NL': 'birdid.country_nl',
            'NO': 'birdid.country_no',
            'NP': 'birdid.country_np',
            'NZ': 'birdid.country_nz',
            'PE': 'birdid.country_pe',
            'PH': 'birdid.country_ph',
            'PL': 'birdid.country_pl',
            'PT': 'birdid.country_pt',
            'RU': 'birdid.country_ru',
            'SE': 'birdid.country_se',
            'SG': 'birdid.country_sg',
            'TH': 'birdid.country_th',
            'TZ': 'birdid.country_tz',
            'UA': 'birdid.country_ua',
            'VN': 'birdid.country_vn',
            'ZA': 'birdid.country_za',
        }

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

        # 收集所有其他区域（排除 Top 10）
        other_regions = []
        for region in self.regions_data.get('countries', []):
            code = region.get('code', '')

            # 跳过已在下拉菜单中的国家
            if code in top10_codes:
                continue

            name_en = region.get('name', code)
            name_cn = region.get('name_cn', '')

            # 获取显示名称
            if code in continent_codes:
                # 大洲：添加 🌍 前缀
                i18n_key = continent_i18n.get(code)
                if i18n_key:
                    base_name = t(i18n_key)
                else:
                    base_name = name_cn if not is_english and name_cn else name_en
                display = f"🌍 {base_name}"
            else:
                # 普通国家
                i18n_key = other_country_i18n.get(code)
                if i18n_key:
                    display = t(i18n_key)
                else:
                    if is_english:
                        display = name_en
                    else:
                        display = name_cn if name_cn else name_en

            # 按英文名排序
            sort_key = name_en.lower()
            other_regions.append((sort_key, display, code, name_en))

        # 按英文名 A-Z 排序
        other_regions.sort(key=lambda x: x[0])

        for _, display, code, name_en in other_regions:
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
                # 添加到下拉菜单并选中
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
        self._populate_country_combo()
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
        # 区域选择行 (V4.2: 二级区域已移除，隐藏整行)
        # region_row = QHBoxLayout()
        # region_label = QLabel(self.i18n.t("birdid.region"))
        # ... (label styling removed) ...
        # region_row.addWidget(region_label)
        
        self.region_combo = QComboBox()
        self.region_combo.addItem(self.i18n.t("birdid.region_entire_country"))
        self.region_combo.hide() # 隐藏
        self.region_combo.currentTextChanged.connect(self._on_region_changed)
        
        # region_row.addWidget(self.region_combo, 1)
        # filter_layout.addLayout(region_row)
        
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
        layout.addWidget(self.drop_area)

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
        self._result_crop_pixmap = None  # 保存识别完成的裁剪图，用于结果卡片点击恢复
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
        self.results_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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

        # 占位区：初始可见，有结果时隐藏
        self.placeholder_frame = QFrame()
        self.placeholder_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_elevated']};
                border-radius: 10px;
            }}
        """)
        ph_layout = QVBoxLayout(self.placeholder_frame)
        ph_layout.setAlignment(Qt.AlignCenter)
        ph_label = QLabel("拖入鸟类照片\n识别结果将显示在这里")
        ph_label.setAlignment(Qt.AlignCenter)
        ph_label.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 12px;
        """)
        ph_layout.addWidget(ph_label)
        layout.addWidget(self.placeholder_frame, 1)  # stretch=1，与 results_frame 同级

        layout.addWidget(self.results_frame, 1)  # stretch=1，填满剩余空间

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

        self.setWidget(container)


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

    # 对焦状态键映射（photo_processor 内部值 → i18n key）
    _FOCUS_STATUS_I18N = {
        'BEST':  'rating_engine.focus_best',
        'GOOD':  'rating_engine.focus_good',
        'BAD':   'rating_engine.focus_bad',
        'WORST': 'rating_engine.focus_worst',
    }
    # 对焦状态颜色
    _FOCUS_STATUS_COLOR = {
        'BEST':  '#00e5a0',  # 绿
        'GOOD':  '#7ec8e3',  # 蓝绿
        'BAD':   '#f0a500',  # 橙
        'WORST': '#e05c5c',  # 红
    }

    def update_crop_preview(self, debug_img, focus_status=None):
        """
        V4.2: 接收选片过程中的裁剪预览图像并显示，同时在结果区更新对焦状态文字
        Args:
            debug_img: BGR numpy 数组 (带标注的鸟类裁剪图)
            focus_status: 对焦状态键 "BEST"/"GOOD"/"BAD"/"WORST" 或 None
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

            # 保存并显示预览
            self._current_pixmap = pixmap
            self.preview_label.show()
            self._scale_preview()

        except Exception as e:
            print(f"[BirdIDDock] 预览更新失败: {e}")

        # 更新结果区：清空旧内容，显示当前对焦状态
        self.clear_results()
        self.placeholder_frame.hide()
        self.results_frame.show()

        if focus_status and focus_status in self._FOCUS_STATUS_I18N:
            i18n_key = self._FOCUS_STATUS_I18N[focus_status]
            raw_text = self.i18n.t(i18n_key)
            # i18n 值带前缀标点（"，精焦" / ", Critical Focus"），去掉它
            display_text = raw_text.lstrip("，, ").strip()
            color = self._FOCUS_STATUS_COLOR.get(focus_status, COLORS['text_secondary'])

            focus_label = QLabel(display_text)
            focus_label.setAlignment(Qt.AlignCenter)
            focus_label.setStyleSheet(f"""
                color: {color};
                font-size: 15px;
                font-weight: 600;
                padding: 12px;
                background-color: {COLORS['bg_elevated']};
                border-radius: 8px;
            """)
            self.results_layout.addWidget(focus_label)
            self.results_layout.addStretch()

    def show_completion_message(self, stats: dict):
        """
        V4.2: 处理完成后显示统计摘要，隐藏预览图
        Args:
            stats: photo_processor 返回的统计字典
        """
        # 隐藏预览图
        self.preview_label.hide()
        self._current_pixmap = None

        # 清空结果，切换到结果区显示完成信息
        self.clear_results()
        self.placeholder_frame.hide()
        self.results_frame.show()

        total      = stats.get('total', 0)
        star_3     = stats.get('star_3', 0)
        star_2     = stats.get('star_2', 0)
        star_1     = stats.get('star_1', 0)
        star_0     = stats.get('star_0', 0)
        no_bird    = stats.get('no_bird', 0)
        total_time = stats.get('total_time', 0)
        flying     = stats.get('flying', 0)
        focus_precise = stats.get('focus_precise', 0)
        bird_species  = stats.get('bird_species', [])

        def pct(n):
            return f"{n/total*100:.1f}%" if total > 0 else "—"

        lines = [f"✅  分析完成  |  {total} 张  |  {total_time/60:.1f} min", ""]
        if total > 0:
            lines.append(f"⭐⭐⭐  {star_3:>4}  ({pct(star_3)})")
            lines.append(f"⭐⭐    {star_2:>4}  ({pct(star_2)})")
            lines.append(f"⭐      {star_1:>4}  ({pct(star_1)})")
            lines.append(f"0⭐     {star_0:>4}  ({pct(star_0)})")
            lines.append(f"❌      {no_bird:>4}  ({pct(no_bird)})")

        if flying > 0 or focus_precise > 0:
            lines.append("")
            if flying > 0:
                lines.append(f"🟢 飞版: {flying}")
            if focus_precise > 0:
                lines.append(f"🔴 精焦: {focus_precise}")

        if bird_species:
            is_chinese = self.i18n.current_lang.startswith('zh')
            names = []
            for sp in bird_species:
                if isinstance(sp, dict):
                    name = sp.get('cn_name', '') if is_chinese else sp.get('en_name', '')
                    if not name:
                        name = sp.get('en_name', '') or sp.get('cn_name', '')
                else:
                    name = str(sp)
                if name:
                    names.append(name)
            if names:
                lines.append("")
                lines.append(f"🦜 {len(names)} 种: {', '.join(names)}")

        info_label = QLabel('\n'.join(lines))
        info_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 12px;
            font-family: {FONTS['mono']};
            padding: 16px;
            background-color: {COLORS['bg_elevated']};
            border-radius: 8px;
        """)
        info_label.setWordWrap(True)
        self.results_layout.addWidget(info_label)
        self.results_layout.addStretch()

    def clear_results(self):
        """清空结果区域"""
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def on_identify_finished(self, result: dict):
        """识别完成"""
        self.progress.hide()
        t = self.i18n.t

        # === 构建状态信息 ===
        info_lines = []

        # 1. YOLO 检测状态
        yolo_info = result.get('yolo_info')
        if yolo_info is not None:
            if isinstance(yolo_info, dict) and yolo_info.get('bird_count', 1) == 0:
                info_lines.append(t("birdid.info_yolo_fail"))
            else:
                info_lines.append(t("birdid.info_yolo_ok"))

        # 2. 地理过滤状态
        gps_info = result.get('gps_info')
        ebird_info = result.get('ebird_info')

        if gps_info and gps_info.get('latitude'):
            # GPS 过滤生效
            count = ebird_info.get('species_count', 0) if ebird_info else 0
            lat = f"{gps_info['latitude']:.2f}"
            lon = f"{gps_info['longitude']:.2f}"
            info_lines.append(t("birdid.info_gps", lat=lat, lon=lon, count=count))
            # GPS 回退提示（优先显示国家级回退，其次全局）
            if ebird_info and ebird_info.get('country_fallback'):
                country = ebird_info.get('country_code', '?')
                info_lines.append(t("birdid.info_country_fallback", country=country))
            elif ebird_info and ebird_info.get('gps_fallback'):
                info_lines.append(t("birdid.info_gps_fallback", count=count))
        elif ebird_info and ebird_info.get('enabled'):
            # 区域过滤生效
            region = ebird_info.get('region_code', '')
            count = ebird_info.get('species_count', 0)
            if region:
                info_lines.append(t("birdid.info_region", region=region, count=count))
            else:
                info_lines.append(t("birdid.info_region", region="—", count=count))
        else:
            info_lines.append(t("birdid.info_global"))

        # === 处理失败/无结果 ===
        if not result.get('success'):
            info_lines.append(t("birdid.info_identify_fail"))
            self._show_info_panel(info_lines)
            return

        results = result.get('results', [])
        if not results:
            # 无鸟或无结果
            if isinstance(yolo_info, dict) and yolo_info.get('bird_count', 1) == 0:
                info_lines.append(t("birdid.info_no_bird_hint"))
            else:
                info_lines.append(t("birdid.info_no_result"))
            self._show_info_panel(info_lines)
            return

        # === 显示信息面板 + 结果卡片 ===
        self.results_frame.show()
        self.placeholder_frame.hide()
        self.result_cards = []
        self.selected_index = 0

        # 信息标签（在结果卡片之前）
        if info_lines:
            info_label = QLabel('\n'.join(info_lines))
            info_label.setWordWrap(True)
            info_label.setStyleSheet(f"""
                color: {COLORS['text_secondary']};
                font-size: 11px;
                padding: 8px 10px;
                background-color: {COLORS['bg_elevated']};
                border-radius: 6px;
                line-height: 1.4;
            """)
            self.results_layout.addWidget(info_label)

        for i, r in enumerate(results, 1):
            card = ResultCard(
                rank=i,
                cn_name=r.get('cn_name', '未知'),
                en_name=r.get('en_name', 'Unknown'),
                confidence=r.get('confidence', 0)
            )
            card.clicked.connect(self.on_result_card_clicked)
            if i == 1:
                card.set_selected(True)
            self.result_cards.append(card)
            self.results_layout.addWidget(card)

        self.results_layout.addStretch()

        # 用 YOLO 裁剪图替换预览（正方形）
        cropped_pil = result.get('cropped_image')
        if cropped_pil:
            try:
                from PySide6.QtGui import QImage
                rgb = cropped_pil.convert('RGB')
                data = rgb.tobytes('raw', 'RGB')
                q_img = QImage(data, rgb.width, rgb.height,
                               rgb.width * 3, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                if not pixmap.isNull():
                    self._current_pixmap = pixmap
                    self._result_crop_pixmap = pixmap
                    self._scale_preview()
            except Exception as _e:
                print(f"[BirdIDDock] 裁剪图预览更新失败: {_e}")

        # 保存结果
        self.identify_results = results

        # 状态显示选中的候选
        self._update_status_label()

    def _show_info_panel(self, info_lines: list):
        """显示纯信息面板（无结果卡片时使用）"""
        self.results_frame.show()
        self.placeholder_frame.hide()
        info_label = QLabel('\n'.join(info_lines))
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 11px;
            padding: 10px 12px;
            background-color: {COLORS['bg_elevated']};
            border-radius: 6px;
            line-height: 1.5;
        """)
        self.results_layout.addWidget(info_label)

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

        # 点击结果卡片时恢复 YOLO 裁剪预览
        if getattr(self, '_result_crop_pixmap', None):
            self._current_pixmap = self._result_crop_pixmap
            self._scale_preview()

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
        self.placeholder_frame.show()
        self._result_crop_pixmap = None

        self.status_label.setText("准备就绪")
        self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text_muted']};")
        self.current_image_path = None
        self.identify_results = None
        self._current_pixmap = None
        self.clear_results()
