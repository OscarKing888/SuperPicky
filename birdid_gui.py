#!/usr/bin/env python3
"""
SuperPicky BirdID GUI - 极简鸟类识别界面
拖放图片 + 识别结果显示 + eBird 地区过滤
"""

__version__ = "1.1.0"

import os
import sys
import json

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QFileDialog, QProgressBar,
    QMessageBox, QComboBox, QCheckBox, QGroupBox
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QFont


def get_birdid_data_path(relative_path: str) -> str:
    """获取 birdid/data 目录下的资源路径"""
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'birdid', 'data', relative_path)
    return os.path.join(os.path.dirname(__file__), 'birdid', 'data', relative_path)


def get_settings_path() -> str:
    """获取设置文件路径"""
    if sys.platform == 'darwin':
        settings_dir = os.path.expanduser('~/Documents/SuperPicky_Data')
    else:
        settings_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'SuperPicky_Data')
    os.makedirs(settings_dir, exist_ok=True)
    return os.path.join(settings_dir, 'birdid_gui_settings.json')


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
    """拖放区域"""
    fileDropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 250)
        self.setStyleSheet("""
            DropArea {
                border: 3px dashed #aaa;
                border-radius: 12px;
                background-color: #f8f8f8;
            }
            DropArea:hover {
                border-color: #4a90d9;
                background-color: #f0f7ff;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # 图标
        icon_label = QLabel("🐦")
        icon_label.setFont(QFont("Arial", 48))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        # 提示文字
        hint_label = QLabel("拖放图片到此处\n或点击选择文件")
        hint_label.setAlignment(Qt.AlignCenter)
        hint_label.setStyleSheet("color: #666; font-size: 16px;")
        layout.addWidget(hint_label)

        # 支持格式
        format_label = QLabel("支持: JPG, PNG, NEF, CR2, ARW 等")
        format_label.setAlignment(Qt.AlignCenter)
        format_label.setStyleSheet("color: #999; font-size: 12px;")
        layout.addWidget(format_label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.fileDropped.emit(file_path)

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


class ResultCard(QFrame):
    """识别结果卡片"""

    def __init__(self, rank: int, cn_name: str, en_name: str, confidence: float, ebird_match: bool = False):
        super().__init__()
        self.setStyleSheet("""
            ResultCard {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 8px;
            }
        """)

        layout = QHBoxLayout(self)

        # 排名
        rank_label = QLabel(f"#{rank}")
        rank_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4a90d9; min-width: 40px;")
        layout.addWidget(rank_label)

        # 名称
        name_layout = QVBoxLayout()
        cn_label = QLabel(cn_name)
        cn_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        name_layout.addWidget(cn_label)

        en_label = QLabel(en_name)
        en_label.setStyleSheet("font-size: 12px; color: #666;")
        name_layout.addWidget(en_label)
        
        # eBird 匹配标记
        if ebird_match:
            ebird_label = QLabel("✓ eBird确认")
            ebird_label.setStyleSheet("font-size: 10px; color: #4caf50; font-weight: bold;")
            name_layout.addWidget(ebird_label)

        layout.addLayout(name_layout, 1)

        # 置信度
        conf_label = QLabel(f"{confidence:.1f}%")
        if confidence >= 80:
            conf_color = "#4caf50"
        elif confidence >= 50:
            conf_color = "#ff9800"
        else:
            conf_color = "#f44336"
        conf_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {conf_color};")
        layout.addWidget(conf_label)


class BirdIDWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"SuperPicky 鸟类识别 v{__version__}")
        self.setMinimumSize(650, 800)
        self.worker = None
        self.current_image_path = None
        
        # 加载区域数据
        self.regions_data = self.load_regions_data()
        self.country_list = self.build_country_list()
        
        # 加载设置
        self.settings = self.load_settings()

        self.setup_ui()
        
        # 应用保存的设置
        self.apply_settings()

    def load_regions_data(self) -> dict:
        """加载 eBird 区域数据"""
        regions_path = get_birdid_data_path('ebird_regions.json')
        if os.path.exists(regions_path):
            try:
                with open(regions_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载区域数据失败: {e}")
        return {'countries': []}
    
    def build_country_list(self) -> dict:
        """构建国家列表 {显示名称: 代码}"""
        country_list = {"自动检测 (GPS)": None, "全球模式 (不过滤)": "GLOBAL"}
        
        for country in self.regions_data.get('countries', []):
            code = country.get('code', '')
            name = country.get('name', '')
            name_cn = country.get('name_cn', '')
            
            if name_cn:
                display = f"{name_cn} ({name})"
            else:
                display = name
            
            country_list[display] = code
        
        return country_list
    
    def load_settings(self) -> dict:
        """加载设置"""
        settings_path = get_settings_path()
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            'use_gps': True,
            'use_ebird': True,
            'selected_country': '自动检测 (GPS)',
            'selected_region': '整个国家'
        }
    
    def save_settings(self):
        """保存设置"""
        self.settings = {
            'use_gps': self.gps_checkbox.isChecked(),
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
    
    def apply_settings(self):
        """应用保存的设置"""
        self.gps_checkbox.setChecked(self.settings.get('use_gps', True))
        self.ebird_checkbox.setChecked(self.settings.get('use_ebird', True))
        
        saved_country = self.settings.get('selected_country', '自动检测 (GPS)')
        idx = self.country_combo.findText(saved_country)
        if idx >= 0:
            self.country_combo.setCurrentIndex(idx)
        
        saved_region = self.settings.get('selected_region', '整个国家')
        idx = self.region_combo.findText(saved_region)
        if idx >= 0:
            self.region_combo.setCurrentIndex(idx)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # 标题
        title = QLabel("🐦 鸟类识别")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # ===== 位置过滤设置 =====
        filter_group = QGroupBox("📍 位置过滤")
        filter_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        filter_layout = QVBoxLayout(filter_group)
        
        # 国家和区域选择行
        location_row = QHBoxLayout()
        
        # 国家选择
        country_label = QLabel("国家:")
        country_label.setStyleSheet("font-weight: normal;")
        location_row.addWidget(country_label)
        
        self.country_combo = QComboBox()
        self.country_combo.addItems(list(self.country_list.keys()))
        self.country_combo.setMinimumWidth(180)
        self.country_combo.currentTextChanged.connect(self.on_country_changed)
        location_row.addWidget(self.country_combo)
        
        location_row.addSpacing(20)
        
        # 区域选择
        region_label = QLabel("区域:")
        region_label.setStyleSheet("font-weight: normal;")
        location_row.addWidget(region_label)
        
        self.region_combo = QComboBox()
        self.region_combo.addItem("整个国家")
        self.region_combo.setMinimumWidth(150)
        self.region_combo.currentTextChanged.connect(self.save_settings)
        location_row.addWidget(self.region_combo)
        
        location_row.addStretch()
        filter_layout.addLayout(location_row)
        
        # 开关行
        switch_row = QHBoxLayout()
        
        self.gps_checkbox = QCheckBox("使用 GPS 自动检测")
        self.gps_checkbox.setStyleSheet("font-weight: normal;")
        self.gps_checkbox.stateChanged.connect(self.save_settings)
        switch_row.addWidget(self.gps_checkbox)
        
        switch_row.addSpacing(30)
        
        self.ebird_checkbox = QCheckBox("启用 eBird 过滤")
        self.ebird_checkbox.setStyleSheet("font-weight: normal;")
        self.ebird_checkbox.stateChanged.connect(self.save_settings)
        switch_row.addWidget(self.ebird_checkbox)
        
        switch_row.addStretch()
        filter_layout.addLayout(switch_row)
        
        layout.addWidget(filter_group)

        # 拖放区 / 图片预览
        self.drop_area = DropArea()
        self.drop_area.fileDropped.connect(self.on_file_dropped)
        layout.addWidget(self.drop_area)

        # 图片预览（初始隐藏）
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(250)
        self.preview_label.hide()
        layout.addWidget(self.preview_label)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # 不确定进度
        self.progress.hide()
        layout.addWidget(self.progress)
        
        # eBird 信息标签
        self.ebird_info_label = QLabel()
        self.ebird_info_label.setStyleSheet("color: #4a90d9; font-size: 12px;")
        self.ebird_info_label.setAlignment(Qt.AlignCenter)
        self.ebird_info_label.hide()
        layout.addWidget(self.ebird_info_label)

        # 结果区域
        self.results_label = QLabel("识别结果")
        self.results_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.results_label.hide()
        layout.addWidget(self.results_label)

        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setMinimumHeight(180)
        self.results_scroll.hide()
        layout.addWidget(self.results_scroll)

        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_scroll.setWidget(self.results_widget)

        # 操作按钮
        btn_layout = QHBoxLayout()

        self.btn_new = QPushButton("选择新图片")
        self.btn_new.clicked.connect(self.drop_area.selectFile)
        btn_layout.addWidget(self.btn_new)

        self.btn_write_exif = QPushButton("写入 EXIF")
        self.btn_write_exif.clicked.connect(self.write_exif)
        self.btn_write_exif.hide()
        btn_layout.addWidget(self.btn_write_exif)

        layout.addLayout(btn_layout)

        # 状态栏
        self.statusBar().showMessage("准备就绪")
    
    def on_country_changed(self, country_display: str):
        """国家选择变化时更新区域列表"""
        self.region_combo.clear()
        self.region_combo.addItem("整个国家")
        
        country_code = self.country_list.get(country_display)
        
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
        
        self.save_settings()

    def on_file_dropped(self, file_path: str):
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "错误", f"文件不存在: {file_path}")
            return

        self.current_image_path = file_path
        self.statusBar().showMessage(f"正在识别: {os.path.basename(file_path)}")

        # 显示预览
        self.show_preview(file_path)

        # 清空之前的结果
        self.clear_results()
        self.ebird_info_label.hide()

        # 显示进度
        self.progress.show()
        self.results_label.hide()
        self.results_scroll.hide()
        self.btn_write_exif.hide()
        
        # 获取过滤设置
        use_gps = self.gps_checkbox.isChecked()
        use_ebird = self.ebird_checkbox.isChecked()
        
        country_code = None
        region_code = None
        
        country_display = self.country_combo.currentText()
        country_code_raw = self.country_list.get(country_display)
        
        if country_code_raw and country_code_raw != "GLOBAL":
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
            # 对于 RAW 文件，使用 birdid 模块加载
            ext = os.path.splitext(file_path)[1].lower()
            raw_extensions = ['.nef', '.cr2', '.cr3', '.arw', '.raf', '.orf', '.rw2', '.dng']

            if ext in raw_extensions:
                from birdid.bird_identifier import load_image
                pil_image = load_image(file_path)
                # 转换为 QPixmap
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    pil_image.save(tmp.name, 'JPEG', quality=85)
                    pixmap = QPixmap(tmp.name)
                    os.unlink(tmp.name)
            else:
                pixmap = QPixmap(file_path)

            if not pixmap.isNull():
                scaled = pixmap.scaled(400, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_label.setPixmap(scaled)
                self.drop_area.hide()
                self.preview_label.show()
        except Exception as e:
            print(f"预览加载失败: {e}")

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
            self.statusBar().showMessage(f"识别失败: {result.get('error', '未知错误')}")
            return

        results = result.get('results', [])
        if not results:
            self.statusBar().showMessage("未能识别出鸟类")
            return
        
        # 显示 eBird 过滤信息
        ebird_info = result.get('ebird_info')
        if ebird_info:
            region = ebird_info.get('region', '未知')
            species_count = ebird_info.get('species_count', 0)
            data_source = ebird_info.get('data_source', '')
            self.ebird_info_label.setText(f"🌍 {data_source} | 该区域 {species_count} 种鸟类")
            self.ebird_info_label.show()
        
        # 显示 GPS 信息
        gps_info = result.get('gps_info')
        if gps_info:
            lat = gps_info.get('latitude', 0)
            lon = gps_info.get('longitude', 0)
            self.statusBar().showMessage(f"GPS: {lat:.4f}, {lon:.4f}")

        # 显示结果
        self.results_label.show()
        self.results_scroll.show()

        for i, r in enumerate(results, 1):
            card = ResultCard(
                rank=i,
                cn_name=r.get('cn_name', '未知'),
                en_name=r.get('en_name', 'Unknown'),
                confidence=r.get('confidence', 0),
                ebird_match=r.get('ebird_match', False)
            )
            self.results_layout.addWidget(card)

        self.results_layout.addStretch()

        # 保存结果用于 EXIF 写入
        self.identify_results = results
        self.btn_write_exif.show()

        self.statusBar().showMessage(f"识别完成: {results[0]['cn_name']} ({results[0]['confidence']:.1f}%)")

    def on_identify_error(self, error_msg: str):
        """识别出错"""
        self.progress.hide()
        self.statusBar().showMessage(f"识别错误: {error_msg}")
        QMessageBox.critical(self, "识别错误", error_msg)

    def write_exif(self):
        """写入 EXIF"""
        if not self.current_image_path or not hasattr(self, 'identify_results'):
            return

        best = self.identify_results[0]
        bird_name = f"{best['cn_name']} ({best['en_name']})"

        try:
            from exiftool_manager import get_exiftool_manager
            exiftool_mgr = get_exiftool_manager()
            success = exiftool_mgr.set_metadata(self.current_image_path, {'Title': bird_name})

            if success:
                QMessageBox.information(self, "成功", f"已写入 EXIF Title:\n{bird_name}")
                self.statusBar().showMessage(f"EXIF 写入成功: {bird_name}")
            else:
                QMessageBox.warning(self, "失败", "EXIF 写入失败")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"EXIF 写入错误: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = BirdIDWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
