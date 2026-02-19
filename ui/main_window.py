# -*- coding: utf-8 -*-
"""
SuperPicky - 主窗口
PySide6 版本 - 极简艺术风格
"""

import os
import sys
import threading
import subprocess
from pathlib import Path


def get_resource_path(relative_path):
    """获取资源文件路径（兼容 PyInstaller 打包环境）"""
    # PyInstaller 打包后会设置 _MEIPASS
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    # 开发环境
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), relative_path)

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QProgressBar,
    QTextEdit, QGroupBox, QCheckBox, QMenuBar, QMenu,
    QFileDialog, QMessageBox, QSizePolicy, QFrame, QSpacerItem,
    QSystemTrayIcon, QApplication  # V4.0: 系统托盘图标
)
from PySide6.QtCore import Qt, Signal, QObject, Slot, QTimer, QPropertyAnimation, QEasingCurve, QMimeData, QThread
from PySide6.QtGui import QFont, QPixmap, QIcon, QAction, QTextCursor, QColor, QDragEnterEvent, QDropEvent

from tools.i18n import get_i18n
from advanced_config import get_advanced_config
from ui.styles import (
    GLOBAL_STYLE, TITLE_STYLE, SUBTITLE_STYLE, VERSION_STYLE, VALUE_STYLE,
    COLORS, FONTS, LOG_COLORS, PROGRESS_INFO_STYLE, PROGRESS_PERCENT_STYLE
)
from ui.custom_dialogs import StyledMessageBox
from ui.skill_level_dialog import SkillLevelDialog, SKILL_PRESETS, get_skill_level_thresholds


# V3.9: 支持拖放的目录输入框
class DropLineEdit(QLineEdit):
    """支持拖放目录的 QLineEdit"""
    pathDropped = Signal(str)  # 拖放目录后发射此信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """验证拖入的内容"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                path = urls[0].toLocalFile()
                if os.path.isdir(path):
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """处理拖放"""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isdir(path):
                self.setText(path)
                self.pathDropped.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()


class WorkerSignals(QObject):
    """工作线程信号"""
    progress = Signal(int)
    log = Signal(str, str)  # message, tag
    finished = Signal(dict)
    error = Signal(str)
    crop_preview = Signal(object, object)  # V4.2: 发送裁剪预览图像 (numpy array BGR) + focus_status str
    update_check_done = Signal(bool, object)  # V4.2: 更新检测完成 (has_update, update_info)


class WorkerThread(threading.Thread):
    """处理线程"""

    def __init__(self, dir_path, ui_settings, signals, i18n=None):
        super().__init__(daemon=True)
        self.dir_path = dir_path
        self.ui_settings = ui_settings
        self.signals = signals
        self.i18n = i18n
        self._stop_event = threading.Event()
        self.caffeinate_process = None

        self.stats = {
            'total': 0,
            'star_3': 0,
            'picked': 0,
            'star_2': 0,
            'star_1': 0,
            'star_0': 0,
            'no_bird': 0,
            'start_time': 0,
            'end_time': 0,
            'total_time': 0,
            'avg_time': 0
        }

    def run(self):
        """执行处理"""
        try:
            self._start_caffeinate()
            self.process_files()
            self.signals.finished.emit(self.stats)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self._stop_caffeinate()

    def _start_caffeinate(self):
        """启动防休眠"""
        if sys.platform != 'darwin':
            return  # 目前仅在 macOS 上支持 caffeinate
            
        try:
            # V3.8.1: 先清理残留的 caffeinate 进程，避免累积
            try:
                subprocess.run(['killall', 'caffeinate'], 
                              stdout=subprocess.DEVNULL, 
                              stderr=subprocess.DEVNULL,
                              timeout=2)
            except Exception:
                pass  # 如果没有残留进程，忽略错误
            
            self.caffeinate_process = subprocess.Popen(
                ['caffeinate', '-d', '-i'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if self.i18n:
                self.signals.log.emit(self.i18n.t("logs.caffeinate_started"), "info")
        except Exception as e:
            if self.i18n:
                self.signals.log.emit(self.i18n.t("logs.caffeinate_failed", error=str(e)), "warning")

    def _stop_caffeinate(self):
        """停止防休眠"""
        if self.caffeinate_process:
            try:
                self.caffeinate_process.terminate()
                self.caffeinate_process.wait(timeout=2)
            except Exception:
                try:
                    self.caffeinate_process.kill()
                except Exception:
                    pass
            finally:
                self.caffeinate_process = None

    def process_files(self):
        """处理文件"""
        from core.photo_processor import (
            PhotoProcessor,
            ProcessingSettings,
            ProcessingCallbacks
        )
        
        # 读取 BirdID 设置
        # V4.2: 从 ui_settings 读取识鸟开关状态（索引 8），而不是从文件
        birdid_auto_identify = self.ui_settings[8] if len(self.ui_settings) > 8 else False
        birdid_use_ebird = True
        birdid_country_code = None
        birdid_region_code = None
        
        # V4.2: 从高级配置读取识别置信度阈值
        from advanced_config import get_advanced_config
        birdid_confidence_threshold = get_advanced_config().birdid_confidence
        
        # 从设置文件读取国家/区域配置
        try:
            import json
            import re
            import sys as sys_module
            import os
            
            if sys_module.platform == 'darwin':
                birdid_settings_dir = os.path.expanduser('~/Documents/SuperPicky_Data')
            else:
                birdid_settings_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'SuperPicky_Data')
            birdid_settings_path = os.path.join(birdid_settings_dir, 'birdid_dock_settings.json')
            
            print(f"[DEBUG] 检查设置文件: {birdid_settings_path}, 存在: {os.path.exists(birdid_settings_path)}")
            
            if os.path.exists(birdid_settings_path):
                with open(birdid_settings_path, 'r', encoding='utf-8') as f:
                    birdid_settings = json.load(f)
                    # 只从文件读取国家/区域配置，auto_identify 从 ui_settings 读取
                    birdid_use_ebird = birdid_settings.get('use_ebird', True)
                    
                    # V4.0.4: 直接读取 country_code（新格式）
                    birdid_country_code = birdid_settings.get('country_code')
                    
                    # 兼容旧格式：如果没有 country_code，尝试从 selected_country 解析
                    if not birdid_country_code:
                        selected_country = birdid_settings.get('selected_country', '自动检测 (GPS)')
                        if selected_country and selected_country != '自动检测 (GPS)':
                            # 尝试从 "澳大利亚 (AU)" 格式中提取代码
                            match = re.search(r'\(([A-Z]{2,3})\)', selected_country)
                            if match:
                                birdid_country_code = match.group(1)
                            else:
                                # V4.0.4: 名称到代码的映射（兼容旧设置文件）
                                country_name_to_code = {
                                    '澳大利亚': 'AU', '中国': 'CN', '美国': 'US',
                                    '日本': 'JP', '英国': 'GB', '新西兰': 'NZ',
                                    '加拿大': 'CA', '印度': 'IN', '德国': 'DE',
                                    '法国': 'FR', '巴西': 'BR', '南非': 'ZA',
                                    '韩国': 'KR', '台湾': 'TW', '香港': 'HK',
                                    '新加坡': 'SG', '马来西亚': 'MY', '泰国': 'TH',
                                    '印度尼西亚': 'ID', '菲律宾': 'PH', '意大利': 'IT',
                                    '西班牙': 'ES', '荷兰': 'NL', '哥斯达黎加': 'CR',
                                }
                                birdid_country_code = country_name_to_code.get(selected_country.strip())
                    
                    # V4.0.4: 直接读取 region_code（新格式）
                    birdid_region_code = birdid_settings.get('region_code')
                    
                    # 兼容旧格式：如果没有 region_code，尝试从 selected_region 解析
                    if not birdid_region_code:
                        selected_region = birdid_settings.get('selected_region', '整个国家')
                        if selected_region and selected_region != '整个国家':
                            match = re.search(r'\(([A-Z]{2}-[A-Z0-9]+)\)', selected_region)
                            if match:
                                birdid_region_code = match.group(1)
            print(f"[DEBUG] BirdID 设置读取: auto_identify={birdid_auto_identify}, country={birdid_country_code}, region={birdid_region_code}, confidence={birdid_confidence_threshold}%")
        except Exception as e:
            print(f"[DEBUG] BirdID 设置读取失败: {e}")
            # BirdID 设置读取失败不影响主流程
            # 使用默认值
            birdid_use_ebird = True
            birdid_country_code = None
            birdid_region_code = None

        settings = ProcessingSettings(
            ai_confidence=self.ui_settings[0],
            sharpness_threshold=self.ui_settings[1],
            nima_threshold=self.ui_settings[2],
            save_crop=self.ui_settings[3] if len(self.ui_settings) > 3 else False,
            normalization_mode=self.ui_settings[4] if len(self.ui_settings) > 4 else 'log_compression',
            detect_flight=self.ui_settings[5] if len(self.ui_settings) > 5 else True,
            detect_exposure=self.ui_settings[6] if len(self.ui_settings) > 6 else False,  # V3.8: 默认关闭
            detect_burst=self.ui_settings[7] if len(self.ui_settings) > 7 else True,  # V4.0: 默认开启
            # BirdID 设置
            auto_identify=birdid_auto_identify,
            birdid_use_ebird=birdid_use_ebird,
            birdid_country_code=birdid_country_code,
            birdid_region_code=birdid_region_code,
            birdid_confidence_threshold=float(birdid_confidence_threshold),  # V4.2
        )

        def log_callback(msg, level="info"):
            self.signals.log.emit(msg, level)

        def progress_callback(value):
            self.signals.progress.emit(int(value))

        # V4.2: 裁剪预览回调
        def crop_preview_callback(debug_img, focus_status=None):
            self.signals.crop_preview.emit(debug_img, focus_status)

        callbacks = ProcessingCallbacks(
            log=log_callback,
            progress=progress_callback,
            crop_preview=crop_preview_callback
        )

        processor = PhotoProcessor(
            dir_path=self.dir_path,
            settings=settings,
            callbacks=callbacks
        )

        # V4.0.5: 读取 keep_temp_files 配置，决定是否清理临时文件
        from advanced_config import get_advanced_config
        adv_config = get_advanced_config()
        
        result = processor.process(
            organize_files=True,
            cleanup_temp=not adv_config.keep_temp_files
        )

        # V4.0.4: 连拍检测已移至 PhotoProcessor 内部
        # 直接读取统计信息并显示
        burst_groups = result.stats.get('burst_groups', 0)
        burst_moved = result.stats.get('burst_moved', 0)
        
        if burst_groups > 0:
            log_callback(self.i18n.t("logs.burst_complete", groups=burst_groups, moved=burst_moved), "success")
        elif settings.detect_burst:
            log_callback(self.i18n.t("logs.burst_none_detected"), "info")
            
        self.stats = result.stats


class SuperPickyMainWindow(QMainWindow):
    """SuperPicky 主窗口 - 极简艺术风格"""

    # V3.6: 重置操作的信号
    reset_log_signal = Signal(str)
    reset_complete_signal = Signal(bool, dict, dict)
    
    # V4.2.1: 日志信号，确保线程安全
    log_signal = Signal(str, str)
    reset_error_signal = Signal(str)

    def __init__(self):
        super().__init__()

        # 初始化配置和国际化
        self.config = get_advanced_config()
        self.i18n = get_i18n(self.config.language)

        # 状态变量
        self.directory_path = ""
        self.worker = None
        self.worker_signals = None
        self.current_progress = 0
        self.total_files = 0

        # 设置窗口
        self._setup_window()
        self._setup_menu()
        self._setup_ui()
        self._setup_birdid_dock()  # V4.0: 识鸟停靠面板
        self._show_initial_help()

        # 连接重置信号
        # 连接重置信号
        self.reset_log_signal.connect(self._log)
        # 修复Crash: 确保日志信号连接到主线程槽
        # noinspection PyUnresolvedReferences
        self.log_signal.connect(self._log, Qt.QueuedConnection)
        self.reset_complete_signal.connect(self._on_reset_complete)
        self.reset_error_signal.connect(self._on_reset_error)
        
        # V4.2: 更新检测信号
        self._update_signals = WorkerSignals()
        self._update_signals.update_check_done.connect(self._show_update_result_dialog)

        # V4.0: 自动启动识鸟 API 服务器
        self._birdid_server_process = None
        QTimer.singleShot(1000, self._auto_start_birdid_server)

        # V4.0.1: 启动时检查更新（延迟2秒，避免阻塞UI，没有更新时不弹窗）
        QTimer.singleShot(2000, lambda: self._check_for_updates(silent=True))
        
        # V4.2: 启动时预加载所有模型（延迟3秒，后台加载不阻塞UI）
        QTimer.singleShot(3000, self._preload_all_models)
        
        # V4.0: 设置系统托盘图标（关闭窗口时最小化到托盘）
        self._setup_system_tray()
        self._really_quit = False  # 标记是否真正退出
        self._background_mode = False  # V4.0: 标记是否进入后台模式（不停止服务器）
        
        # V4.3: 首次运行时显示水平选择对话框（延迟500ms，确保UI已完成渲染）
        if self.config.is_first_run:
            QTimer.singleShot(500, self._show_first_run_skill_level_dialog)
        else:
            # 非首次运行：根据保存的水平设置滑块
            self._apply_skill_level_thresholds(self.config.skill_level)
        
        # V4.2: 使用默认窗口大小，不最大化
        # self.showMaximized()  # 注释掉这行，使用默认大小



    def _get_app_icon(self):
        """获取应用图标"""
        icon_path = os.path.join(os.path.dirname(__file__), "..", "img", "icon.png")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return None

    def _show_message(self, title, message, msg_type="info"):
        """显示消息框"""
        if msg_type == "info":
            return StyledMessageBox.information(self, title, message)
        elif msg_type == "warning":
            return StyledMessageBox.warning(self, title, message)
        elif msg_type == "error":
            return StyledMessageBox.critical(self, title, message)
        elif msg_type == "question":
            return StyledMessageBox.question(self, title, message)
        else:
            return StyledMessageBox.information(self, title, message)

    def _setup_window(self):
        """设置窗口属性"""
        self.setWindowTitle(self.i18n.t("app.window_title"))
        self.setMinimumSize(680, 600)
        self.resize(850, 750)

        # 应用全局样式表
        self.setStyleSheet(GLOBAL_STYLE)

        # 设置图标
        icon_path = get_resource_path("img/icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def _setup_menu(self):
        """设置菜单栏"""
        menubar = self.menuBar()

        # 识鸟菜单
        birdid_menu = menubar.addMenu(self.i18n.t("menu.birdid"))
        


        # 识鸟面板（可勾选显示/隐藏）
        self.birdid_dock_action = QAction(self.i18n.t("menu.toggle_dock"), self)
        self.birdid_dock_action.setCheckable(True)
        self.birdid_dock_action.setChecked(True)
        self.birdid_dock_action.triggered.connect(self._toggle_birdid_dock)
        birdid_menu.addAction(self.birdid_dock_action)
        
        birdid_menu.addSeparator()
        
        # V4.0: 后台运行（最小化到托盘，保持识鸟服务）
        minimize_tray_action = QAction(self.i18n.t("menu.background_mode"), self)
        minimize_tray_action.triggered.connect(self._minimize_to_tray)
        birdid_menu.addAction(minimize_tray_action)

        # 设置菜单
        settings_menu = menubar.addMenu(self.i18n.t("menu.settings_menu"))
        
        # 参数设置
        settings_action = QAction(self.i18n.t("menu.settings"), self)
        settings_action.triggered.connect(self._show_advanced_settings)
        settings_menu.addAction(settings_action)
        
        # V4.3: 摄影水平设置
        skill_level_action = QAction(self.i18n.t("skill_level.section_title") + "...", self)
        skill_level_action.triggered.connect(self._show_skill_level_dialog)
        settings_menu.addAction(skill_level_action)
        
        settings_menu.addSeparator()
        
        # 界面语言子菜单
        lang_menu = settings_menu.addMenu(self.i18n.t("menu.language"))
        
        # 简体中文
        zh_action = QAction(self.i18n.t("menu.lang_zh"), self)
        zh_action.setCheckable(True)
        zh_action.setChecked(self.config.language == "zh_CN")
        zh_action.triggered.connect(lambda: self._change_language("zh_CN"))
        lang_menu.addAction(zh_action)
        
        # English
        en_action = QAction(self.i18n.t("menu.lang_en"), self)
        en_action.setCheckable(True)
        en_action.setChecked(self.config.language == "en")
        en_action.triggered.connect(lambda: self._change_language("en"))
        lang_menu.addAction(en_action)
        
        self.lang_actions = {"zh_CN": zh_action, "en": en_action}

        # 帮助菜单
        help_menu = menubar.addMenu(self.i18n.t("menu.help"))
        
        # 检查更新
        update_action = QAction(self.i18n.t("menu.check_update"), self)
        update_action.triggered.connect(self._check_for_updates)
        help_menu.addAction(update_action)
        
        help_menu.addSeparator()
        
        # 关于
        about_action = QAction(self.i18n.t("menu.about"), self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_ui(self):
        """设置主 UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(0)

        # 头部区域
        self._create_header_section(main_layout)
        main_layout.addSpacing(24)

        # 目录选择
        self._create_directory_section(main_layout)
        main_layout.addSpacing(20)

        # 参数设置
        self._create_parameters_section(main_layout)
        main_layout.addSpacing(20)

        # 日志区域
        self._create_log_section(main_layout)
        main_layout.addSpacing(16)

        # 进度区域
        self._create_progress_section(main_layout)
        main_layout.addSpacing(8)

        # 控制按钮
        self._create_button_section(main_layout)

    def _setup_birdid_dock(self):
        """设置识鸟停靠面板"""
        from .birdid_dock import BirdIDDockWidget

        self.birdid_dock = BirdIDDockWidget(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.birdid_dock)
        
        # 设置 dock 初始宽度为最小值，让主区域更宽
        self.birdid_dock.setFixedWidth(280)
        # 延迟解除固定宽度限制，让用户可以调整
        QTimer.singleShot(100, lambda: self.birdid_dock.setFixedWidth(16777215))  # QWIDGETSIZE_MAX

        # 更新菜单动作的状态
        self.birdid_dock.visibilityChanged.connect(self._on_birdid_dock_visibility_changed)

    def _on_birdid_dock_visibility_changed(self, visible):
        """识鸟面板可见性变化"""
        if hasattr(self, 'birdid_dock_action'):
            self.birdid_dock_action.setChecked(visible)
            # 这里的文字其实不用动态改变，保持 "打开/关闭" 即可，或者更复杂点
            # 暂时保持简单
            pass # self.birdid_dock_action.setText("关闭识鸟面板" if visible else "打开识鸟面板")
    
    def _setup_system_tray(self):
        """V4.0: 设置系统托盘图标"""
        # 检查系统是否支持托盘图标
        if not QSystemTrayIcon.isSystemTrayAvailable():
            print("⚠️ 系统不支持托盘图标")
            return
        
        # 创建托盘图标
        self.tray_icon = QSystemTrayIcon(self)
        
        # 设置图标（使用裁剪后的托盘专用图标）
        icon_path = get_resource_path("img/icon_tray.png")
        if not os.path.exists(icon_path):
            # 回退到原始图标
            icon_path = get_resource_path("img/icon.png")
        if os.path.exists(icon_path):
            self.tray_icon.setIcon(QIcon(icon_path))
        else:
            # 使用窗口图标作为备选
            self.tray_icon.setIcon(self.windowIcon())
        
        # 创建托盘菜单
        tray_menu = QMenu()
        
        # 显示/隐藏主窗口
        show_action = QAction(self.i18n.t("server.tray_show_window"), self)
        show_action.triggered.connect(self._show_main_window)
        tray_menu.addAction(show_action)
        
        tray_menu.addSeparator()
        
        # 服务器状态（只读显示）
        self.tray_server_status = QAction(self.i18n.t("server.tray_server_running"), self)
        self.tray_server_status.setEnabled(False)
        tray_menu.addAction(self.tray_server_status)
        
        tray_menu.addSeparator()
        
        # 完全退出
        quit_action = QAction(self.i18n.t("server.tray_quit"), self)
        quit_action.triggered.connect(self._quit_app)
        tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        
        # 点击托盘图标显示窗口
        self.tray_icon.activated.connect(self._on_tray_activated)
        
        # 设置提示文字
        self.tray_icon.setToolTip(self.i18n.t("server.tray_tooltip"))
        
        # 显示托盘图标
        self.tray_icon.show()
        
        print(self.i18n.t("server.tray_icon_enabled"))
    
    def _on_tray_activated(self, reason):
        """托盘图标被点击"""
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            # 单击：显示/隐藏窗口
            self._show_main_window()
        elif reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            # 双击：显示窗口
            self._show_main_window()
    
    def _show_main_window(self):
        """显示主窗口"""
        # macOS: 恢复 Dock 图标
        if sys.platform == 'darwin':
            try:
                from AppKit import NSApp, NSApplicationActivationPolicyRegular
                NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
                print("✅ 已恢复 Dock 图标")
            except ImportError:
                pass
            except Exception as e:
                print(f"⚠️ 恢复 Dock 图标失败: {e}")
        
        self.show()
        self.raise_()
        self.activateWindow()
        # 确保窗口获得焦点
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
    
    def _quit_app(self):
        """完全退出应用"""
        self._really_quit = True
        
        # 停止识鸟服务器
        if hasattr(self, '_birdid_server_process') and self._birdid_server_process:
            try:
                self._birdid_server_process.terminate()
                self._birdid_server_process.wait(timeout=2)
            except Exception:
                pass
        
        # 隐藏托盘图标
        if hasattr(self, 'tray_icon'):
            self.tray_icon.hide()
        
        # 退出应用
        QApplication.quit()

    def _minimize_to_tray(self):
        """V4.0: 进入后台模式（服务器继续运行，GUI 完全退出）"""
        from server_manager import get_server_status, start_server_daemon
        
        # 1. 确保服务器以守护进程模式运行
        status = get_server_status()
        if not status['healthy']:
            print("🚀 启动守护进程服务器...")
            success, msg, pid = start_server_daemon()
            if not success:
                self._log(f"❌ 无法启动后台服务器: {msg}", "error")
                return
            print(f"✅ 服务器已启动 (PID: {pid})")
        else:
            print(f"✅ 服务器已在运行 (PID: {status['pid']})")
        
        # 2. 显示提示
        QMessageBox.information(
            self,
            self.i18n.t("menu.background_mode_title"),
            self.i18n.t("menu.background_mode_msg"),
            QMessageBox.Ok
        )
        
        # 3. 设置后台模式标志，然后退出 GUI
        self._background_mode = True  # 告诉 closeEvent 不要停止服务器
        print("✅ GUI 即将退出，服务器继续运行")
        
        # 隐藏托盘图标
        if hasattr(self, 'tray_icon') and self.tray_icon:
            self.tray_icon.hide()
        
        # 退出应用
        QApplication.quit()
    
    def _on_birdid_check_changed(self, state):
        """识鸟开关状态变化 - 同步到 BirdID Dock 设置"""
        import json
        try:
            if sys.platform == 'darwin':
                settings_dir = os.path.expanduser('~/Documents/SuperPicky_Data')
            else:
                settings_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'SuperPicky_Data')
            os.makedirs(settings_dir, exist_ok=True)
            settings_path = os.path.join(settings_dir, 'birdid_dock_settings.json')
            
            # 读取现有设置
            settings = {}
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
            
            # 更新 auto_identify
            settings['auto_identify'] = (state == 2)  # Qt.Checked = 2
            
            # 保存设置
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            
            # 同步到 BirdID Dock（如果存在）
            if hasattr(self, 'birdid_dock') and self.birdid_dock:
                self.birdid_dock.auto_identify_checkbox.setChecked(state == 2)
        except Exception as e:
            print(f"同步识鸟设置失败: {e}")

    def _create_header_section(self, parent_layout):
        """创建头部区域 - 品牌展示"""
        header = QFrame()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # 左侧: 品牌
        brand_layout = QHBoxLayout()
        brand_layout.setSpacing(16)

        # 品牌图标
        icon_path = get_resource_path("img/icon.png")
        if os.path.exists(icon_path):
            icon_container = QFrame()
            icon_container.setFixedSize(48, 48)
            icon_container.setStyleSheet(f"""
                QFrame {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 {COLORS['accent']}, stop:1 #00a080);
                    border-radius: 12px;
                }}
            """)
            icon_inner_layout = QHBoxLayout(icon_container)
            icon_inner_layout.setContentsMargins(2, 2, 2, 2)

            icon_label = QLabel()
            pixmap = QPixmap(icon_path).scaled(44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(pixmap)
            icon_inner_layout.addWidget(icon_label)
            brand_layout.addWidget(icon_container)

        # 品牌文字
        brand_text_layout = QVBoxLayout()
        brand_text_layout.setSpacing(2)

        title_label = QLabel(self.i18n.t("app.brand_name"))
        title_label.setStyleSheet(TITLE_STYLE)
        brand_text_layout.addWidget(title_label)

        subtitle_label = QLabel(self.i18n.t("labels.subtitle"))
        subtitle_label.setStyleSheet(SUBTITLE_STYLE)
        brand_text_layout.addWidget(subtitle_label)

        brand_layout.addLayout(brand_text_layout)
        header_layout.addLayout(brand_layout)

        header_layout.addStretch()

        # 右侧: 版本号 + commit hash
        # 右侧: 版本号 + commit hash
        from constants import APP_VERSION
        from core.build_info import COMMIT_HASH
        
        # V4.0.5: 动态获取版本号和 Commit Hash
        commit_hash = COMMIT_HASH
        if commit_hash == "154984fd": # 默认占位符
             # 开发环境尝试获取真实 hash
             try:
                 import subprocess
                 hash_short = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
                 commit_hash = hash_short
             except:
                 pass

        version_text = f"V{APP_VERSION}\n{commit_hash}"
        
        version_label = QLabel(version_text)
        version_label.setStyleSheet(VERSION_STYLE)
        version_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        header_layout.addWidget(version_label)


        parent_layout.addWidget(header)

    def _create_directory_section(self, parent_layout):
        """创建目录选择区域"""
        # Section 标签
        section_label = QLabel(self.i18n.t("labels.photo_directory").upper())
        section_label.setObjectName("sectionLabel")
        parent_layout.addWidget(section_label)
        parent_layout.addSpacing(8)

        # 输入区域
        dir_layout = QHBoxLayout()
        dir_layout.setSpacing(8)

        # V3.9: 使用支持拖放的 DropLineEdit
        self.dir_input = DropLineEdit()
        self.dir_input.setPlaceholderText(self.i18n.t("labels.dir_placeholder"))
        self.dir_input.returnPressed.connect(self._on_path_entered)
        self.dir_input.editingFinished.connect(self._on_path_entered)  # V3.9: 失焦时也验证
        self.dir_input.pathDropped.connect(self._on_path_dropped)  # V3.9: 拖放目录
        dir_layout.addWidget(self.dir_input, 1)

        browse_btn = QPushButton(self.i18n.t("labels.browse"))
        browse_btn.setObjectName("browse")
        browse_btn.setMinimumWidth(100)
        browse_btn.clicked.connect(self._browse_directory)
        dir_layout.addWidget(browse_btn)

        parent_layout.addLayout(dir_layout)

    def _create_parameters_section(self, parent_layout):
        """创建参数设置区域"""
        # 参数卡片容器
        params_frame = QFrame()
        params_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_elevated']};
                border-radius: 10px;
            }}
        """)

        params_layout = QVBoxLayout(params_frame)
        params_layout.setContentsMargins(20, 16, 20, 16)
        params_layout.setSpacing(16)

        # 头部: 标题 + 飞鸟检测开关
        header_layout = QHBoxLayout()

        params_title = QLabel(self.i18n.t("labels.selection_params"))
        params_title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px; font-weight: 500;")
        header_layout.addWidget(params_title)

        header_layout.addStretch()

        # 飞鸟检测开关
        flight_layout = QHBoxLayout()
        flight_layout.setSpacing(10)

        flight_label = QLabel(self.i18n.t("labels.flight_detection"))
        flight_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        flight_layout.addWidget(flight_label)

        self.flight_check = QCheckBox()
        self.flight_check.setChecked(True)
        flight_layout.addWidget(self.flight_check)

        header_layout.addLayout(flight_layout)
        
        # V4.0: 连拍检测开关
        burst_layout = QHBoxLayout()
        burst_layout.setSpacing(10)
        
        burst_label = QLabel(self.i18n.t("labels.burst"))
        burst_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        burst_layout.addWidget(burst_label)
        
        self.burst_check = QCheckBox()
        self.burst_check.setChecked(True)  # 默认开启
        burst_layout.addWidget(self.burst_check)
        
        header_layout.addLayout(burst_layout)
        
        # V3.8: 曝光检测开关
        exposure_layout = QHBoxLayout()
        exposure_layout.setSpacing(10)
        
        exposure_label = QLabel(self.i18n.t("menu.exposure_label"))
        exposure_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        exposure_layout.addWidget(exposure_label)
        
        self.exposure_check = QCheckBox()
        self.exposure_check.setChecked(False)  # V4.2: 默认关闭
        exposure_layout.addWidget(self.exposure_check)
        
        header_layout.addLayout(exposure_layout)
        
        # V4.2: 自动识鸟开关
        birdid_layout = QHBoxLayout()
        birdid_layout.setSpacing(10)
        
        birdid_label = QLabel(self.i18n.t("menu.birdid_label"))
        birdid_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        birdid_layout.addWidget(birdid_label)
        
        self.birdid_check = QCheckBox()
        # 从保存的设置中读取状态
        birdid_saved_state = False
        try:
            import json
            if sys.platform == 'darwin':
                settings_dir = os.path.expanduser('~/Documents/SuperPicky_Data')
            else:
                settings_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'SuperPicky_Data')
            settings_path = os.path.join(settings_dir, 'birdid_dock_settings.json')
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    birdid_settings = json.load(f)
                    birdid_saved_state = birdid_settings.get('auto_identify', False)
        except Exception:
            pass
        self.birdid_check.setChecked(birdid_saved_state)
        self.birdid_check.stateChanged.connect(self._on_birdid_check_changed)
        birdid_layout.addWidget(self.birdid_check)
        
        header_layout.addLayout(birdid_layout)
        
        # V4.3: 摄影水平显示标签
        skill_level_layout = QHBoxLayout()
        skill_level_layout.setSpacing(4)
        
        self.skill_level_label = QLabel("")
        self.skill_level_label.setStyleSheet(f"""
            color: {COLORS['accent']};
            font-size: 11px;
            padding: 2px 6px;
            background-color: {COLORS['accent']}15;
            border-radius: 4px;
        """)
        skill_level_layout.addWidget(self.skill_level_label)
        
        header_layout.addLayout(skill_level_layout)
        
        params_layout.addLayout(header_layout)

        # 隐藏变量（从高级配置读取，避免硬编码）
        self.ai_confidence = int(self.config.min_confidence * 100)  # V4.2: 读取用户设置的检测敏感度
        self.norm_mode = "log_compression"

        # 滑块区域
        sliders_layout = QVBoxLayout()
        sliders_layout.setSpacing(16)

        # 锐度阈值
        sharp_layout = QHBoxLayout()
        sharp_layout.setSpacing(16)

        sharp_label = QLabel(self.i18n.t("labels.sharpness_short"))
        sharp_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; min-width: 80px;")
        sharp_layout.addWidget(sharp_label)

        self.sharp_slider = QSlider(Qt.Horizontal)
        self.sharp_slider.setRange(200, 600)  # 新范围 200-600
        self.sharp_slider.setValue(400)  # 新默认值
        self.sharp_slider.setSingleStep(10)  # V4.0: 更精细的调节（键盘方向键）
        self.sharp_slider.setPageStep(10)    # V4.0: 点击滑块轨道的步进值
        self.sharp_slider.valueChanged.connect(self._on_sharp_changed)
        sharp_layout.addWidget(self.sharp_slider)

        self.sharp_value = QLabel("400")  # 新默认值
        self.sharp_value.setStyleSheet(VALUE_STYLE)
        self.sharp_value.setFixedWidth(50)
        self.sharp_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sharp_layout.addWidget(self.sharp_value)

        sliders_layout.addLayout(sharp_layout)

        # 美学阈值
        nima_layout = QHBoxLayout()
        nima_layout.setSpacing(16)

        nima_label = QLabel(self.i18n.t("labels.aesthetics"))
        nima_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; min-width: 80px;")
        nima_layout.addWidget(nima_label)

        self.nima_slider = QSlider(Qt.Horizontal)
        self.nima_slider.setRange(40, 70)  # 新范围 4.0-7.0
        self.nima_slider.setValue(50)  # 默认值 5.0
        self.nima_slider.valueChanged.connect(self._on_nima_changed)
        nima_layout.addWidget(self.nima_slider)

        self.nima_value = QLabel("5.0")  # 默认值
        self.nima_value.setStyleSheet(VALUE_STYLE)
        self.nima_value.setFixedWidth(50)
        self.nima_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        nima_layout.addWidget(self.nima_value)

        sliders_layout.addLayout(nima_layout)

        params_layout.addLayout(sliders_layout)
        parent_layout.addWidget(params_frame)

    def _create_log_section(self, parent_layout):
        """创建日志区域"""
        # 日志头部
        log_header = QHBoxLayout()

        log_label = QLabel(self.i18n.t("labels.console").upper())
        log_label.setObjectName("sectionLabel")
        log_header.addWidget(log_label)

        log_header.addStretch()

        # 状态指示器
        status_layout = QHBoxLayout()
        status_layout.setSpacing(6)

        self.status_dot = QLabel()
        self.status_dot.setFixedSize(6, 6)
        self.status_dot.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['accent']};
                border-radius: 3px;
            }}
        """)
        status_layout.addWidget(self.status_dot)

        self.status_label = QLabel(self.i18n.t("labels.ready"))
        self.status_label.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: 11px;")
        status_layout.addWidget(self.status_label)

        log_header.addLayout(status_layout)
        parent_layout.addLayout(log_header)
        parent_layout.addSpacing(8)

        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(260)
        parent_layout.addWidget(self.log_text, 1)

    def _create_progress_section(self, parent_layout):
        """创建进度区域"""
        # 进度条 - 直接添加到父布局
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        parent_layout.addWidget(self.progress_bar)
        
        parent_layout.addSpacing(6)

        # 进度信息
        progress_info_layout = QHBoxLayout()
        progress_info_layout.setContentsMargins(0, 0, 0, 0)

        self.progress_info_label = QLabel("")
        self.progress_info_label.setStyleSheet(PROGRESS_INFO_STYLE)
        progress_info_layout.addWidget(self.progress_info_label)

        progress_info_layout.addStretch()

        self.progress_percent_label = QLabel("")
        self.progress_percent_label.setStyleSheet(PROGRESS_PERCENT_STYLE)
        progress_info_layout.addWidget(self.progress_percent_label)

        parent_layout.addLayout(progress_info_layout)

    def _create_button_section(self, parent_layout):
        """创建按钮区域"""
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.setSpacing(8)

        # 重置按钮 (幽灵按钮)
        self.reset_btn = QPushButton(self.i18n.t("labels.reset_short"))
        self.reset_btn.setObjectName("tertiary")
        self.reset_btn.setMinimumWidth(100)
        self.reset_btn.setMinimumHeight(40)
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self._reset_directory)
        btn_layout.addWidget(self.reset_btn)

        # V4.1: 重新评星按钮暂时禁用（计算逻辑复杂度高，预览结果不一致）
        # TODO: 未来版本重构后恢复此功能
        # self.post_da_btn = QPushButton(self.i18n.t("labels.re_rate"))
        # self.post_da_btn.setObjectName("secondary")
        # self.post_da_btn.setMinimumWidth(100)
        # self.post_da_btn.setMinimumHeight(40)
        # self.post_da_btn.setEnabled(False)
        # self.post_da_btn.clicked.connect(self._open_post_adjustment)
        # btn_layout.addWidget(self.post_da_btn)

        # 开始按钮 (主按钮)
        self.start_btn = QPushButton(self.i18n.t("labels.start_processing"))
        self.start_btn.setMinimumWidth(140)
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self._start_processing)
        btn_layout.addWidget(self.start_btn)

        parent_layout.addLayout(btn_layout)

    # ========== 槽函数 ==========

    @Slot()
    def _on_sharp_changed(self):
        """锐度滑块变化"""
        value = self.sharp_slider.value()
        rounded = round(value / 10) * 10  # V4.0: 改为 10 步进
        self.sharp_slider.blockSignals(True)
        self.sharp_slider.setValue(rounded)
        self.sharp_slider.blockSignals(False)
        self.sharp_value.setText(str(rounded))
        
        # V4.3: 检测是否为自选模式（手动调整滑块）
        self._check_custom_mode()

    @Slot()
    def _on_nima_changed(self):
        """NIMA 滑块变化"""
        value = self.nima_slider.value() / 10.0
        self.nima_value.setText(f"{value:.1f}")
        
        # V4.3: 检测是否为自选模式（手动调整滑块）
        self._check_custom_mode()

    @Slot()
    def _on_path_entered(self):
        """路径输入回车或失焦"""
        directory = self.dir_input.text().strip()
        if directory and os.path.isdir(directory):
            # V3.9: 防止重复处理（editingFinished 和 returnPressed 可能同时触发）
            normalized = os.path.normpath(directory)
            if normalized != os.path.normpath(self.directory_path or ""):
                self._handle_directory_selection(directory)
        elif directory:
            StyledMessageBox.critical(
                self,
                self.i18n.t("errors.error_title"),
                self.i18n.t("errors.dir_not_exist", directory=directory)
            )

    @Slot()
    def _browse_directory(self):
        """浏览目录"""
        directory = QFileDialog.getExistingDirectory(
            self,
            self.i18n.t("labels.select_photo_dir"),
            "",
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self._handle_directory_selection(directory)
    
    @Slot(str)
    def _on_path_dropped(self, directory: str):
        """V3.9: 处理拖放的目录"""
        if directory and os.path.isdir(directory):
            self._handle_directory_selection(directory)

    def _handle_directory_selection(self, directory):
        """处理目录选择"""
        # V3.9: 归一化路径并防止重复
        directory = os.path.normpath(directory)
        if directory == os.path.normpath(self.directory_path or ""):
            return  # 同一个目录，跳过
        
        self.directory_path = directory
        self.dir_input.setText(directory)

        self._log(self.i18n.t("messages.dir_selected", directory=directory))

        self.start_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

        self._check_report_csv()

        # V4.1: 检测历史记录 - 只问是否重置（重新评星功能已禁用）
        history_db = os.path.join(directory, ".superpicky", "report.db")
        history_manifest = os.path.join(directory, ".superpicky_manifest.json")

        if os.path.exists(history_db) or os.path.exists(history_manifest):
            reply = StyledMessageBox.question(
                self,
                self.i18n.t("messages.history_detected_title"),
                self.i18n.t("messages.history_reset_msg"),
                yes_text=self.i18n.t("labels.yes"),
                no_text=self.i18n.t("labels.no")
            )
            if reply == StyledMessageBox.Yes:
                # V4.0.4: 快速复原（不重置EXIF，因为马上要重新处理）
                QTimer.singleShot(100, self._quick_restore_directory)

    def _check_report_csv(self):
        """检查是否有 report.db"""
        if not self.directory_path:
            # self.post_da_btn.setEnabled(False)  # V4.1: 重新评星按钮已禁用
            return

        report_path = os.path.join(self.directory_path, ".superpicky", "report.db")
        if os.path.exists(report_path):
            # self.post_da_btn.setEnabled(True)  # V4.1: 重新评星按钮已禁用
            self._log(self.i18n.t("messages.report_detected"))
        else:
            pass  # self.post_da_btn.setEnabled(False)  # V4.1: 重新评星按钮已禁用

    def _update_status(self, text, color=None):
        """更新状态指示器"""
        self.status_label.setText(text)
        if color:
            self.status_dot.setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    border-radius: 3px;
                }}
            """)

    @Slot()
    def _start_processing(self):
        """开始处理"""
        if not self.directory_path:
            StyledMessageBox.warning(
                self,
                self.i18n.t("messages.hint"),
                self.i18n.t("messages.select_dir_first")
            )
            return

        if self.worker and self.worker.is_alive():
            StyledMessageBox.warning(
                self,
                self.i18n.t("messages.hint"),
                self.i18n.t("messages.processing")
            )
            return

        # 确认弹窗 - 动态构建消息
        extra_notes = []
        if self.flight_check.isChecked():
            extra_notes.append(self.i18n.t("dialogs.note_flight"))
        if self.birdid_check.isChecked():
            extra_notes.append(self.i18n.t("dialogs.note_birdid"))
            # 显示当前国家/区域设置
            if hasattr(self, 'birdid_dock') and self.birdid_dock:
                country_display = self.birdid_dock.country_combo.currentText()
                region_display = self.birdid_dock.region_combo.currentText()
                # 构建显示文本
                location_info = f"  🌍 {country_display}"
                if region_display and region_display != self.i18n.t("birdid.region_entire_country"):
                    location_info += f" - {region_display}"
                extra_notes.append(location_info)
            # V4.3: 检查是否选择了国家，如果是 Auto Detect GPS 则提示
            if hasattr(self, 'birdid_dock') and self.birdid_dock:
                country_display = self.birdid_dock.country_combo.currentText()
                country_code = self.birdid_dock.country_list.get(country_display)
                if country_code is None:  # "自动检测 (GPS)" 模式
                    reply = StyledMessageBox.question(
                        self,
                        self.i18n.t("birdid.country_prompt_title"),
                        self.i18n.t("birdid.country_prompt_message"),
                        yes_text=self.i18n.t("labels.yes"),
                        no_text=self.i18n.t("labels.no")
                    )
                    if reply == StyledMessageBox.Yes:
                        # 用户选择现在选择国家
                        self.birdid_dock.country_combo.showPopup()
                        return  # 等用户选择后再开始
        if self.burst_check.isChecked():
            extra_notes.append(self.i18n.t("dialogs.note_burst"))
        
        notes_block = ""
        if extra_notes:
            notes_block = "\n" + "\n".join(extra_notes) + "\n"

        base_msg = self.i18n.t("dialogs.file_organization_msg", extra_notes=notes_block)
        
        reply = StyledMessageBox.question(
            self,
            self.i18n.t("dialogs.file_organization_title"),
            base_msg,
            yes_text=self.i18n.t("labels.yes"),
            no_text=self.i18n.t("labels.no")
        )

        if reply != StyledMessageBox.Yes:
            return

        # 清空日志和进度
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.progress_info_label.setText("")
        self.progress_percent_label.setText("")

        self._update_status(self.i18n.t("labels.processing"), COLORS['warning'])
        self._log(self.i18n.t("logs.processing_start"))

        # 准备 UI 设置
        ui_settings = [
            self.ai_confidence,
            self.sharp_slider.value(),
            self.nima_slider.value() / 10.0,
            True,  # V4.0.5: 始终保存裁切，用于 debug_crop_path 持久化
            self.norm_mode,
            self.flight_check.isChecked(),
            self.exposure_check.isChecked(),  # V3.8: 曝光检测开关
            self.burst_check.isChecked(),     # V4.0: 连拍检测开关
            self.birdid_check.isChecked(),    # V4.2: 识鸟开关
        ]

        # 创建信号
        self.worker_signals = WorkerSignals()
        self.worker_signals.progress.connect(self._on_progress)
        self.worker_signals.log.connect(self._on_log)
        self.worker_signals.finished.connect(self._on_finished)
        self.worker_signals.error.connect(self._on_error)
        # V4.2: 裁剪预览信号连接到 BirdID Dock
        if hasattr(self, 'birdid_dock') and self.birdid_dock:
            self.worker_signals.crop_preview.connect(self.birdid_dock.update_crop_preview)

        # 禁用按钮
        self.start_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

        # 启动工作线程
        self.worker = WorkerThread(
            self.directory_path,
            ui_settings,
            self.worker_signals,
            self.i18n
        )
        self.worker.start()

    @Slot(int)
    def _on_progress(self, value):
        """进度更新"""
        self.progress_bar.setValue(value)
        self.progress_percent_label.setText(f"{value}%")

    @Slot(str, str)
    def _on_log(self, message, tag):
        """日志更新"""
        self._log(message, tag)

    @Slot(dict)
    def _on_finished(self, stats):
        """处理完成"""
        self.start_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        # self.post_da_btn.setEnabled(True)  # V4.1: 重新评星按钮已禁用
        self.progress_bar.setValue(100)
        self.progress_percent_label.setText("100%")
        self.progress_info_label.setText(self.i18n.t("labels.complete"))

        self._update_status(self.i18n.t("labels.complete"), COLORS['success'])

        # 显示报告（不清空之前的日志）
        report = self._format_statistics_report(stats)
        self._log(report)

        # 显示 Lightroom 指南
        self._show_lightroom_guide()

        # V4.2: 通知 BirdIDDock 显示完成信息（传入 stats 替代 debug_dir）
        if hasattr(self, 'birdid_dock') and self.birdid_dock:
            self.birdid_dock.show_completion_message(stats)

        # 播放完成音效
        self._play_completion_sound()

        # 打开目录
        if self.directory_path and os.path.exists(self.directory_path):
            if sys.platform == 'darwin':
                subprocess.Popen(['open', self.directory_path])
            elif sys.platform.startswith('win'):
                os.startfile(self.directory_path)
            else:
                try:
                    subprocess.Popen(['xdg-open', self.directory_path])
                except Exception:
                    pass

    @Slot(str)
    def _on_error(self, error_msg):
        """处理错误"""
        self._log(f"Error: {error_msg}", "error")
        self._update_status(self.i18n.t("errors.error_title"), COLORS['error'])
        self.start_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

    @Slot()
    def _quick_restore_directory(self):
        """V4.0.4: 快速复原目录（只移动文件，不重置EXIF）
        
        用于重新处理时的确认弹窗，因为EXIF会被新的处理结果覆盖
        """
        self._do_reset_directory(skip_exif_reset=True, skip_confirm=True)
    
    @Slot()
    def _reset_directory(self):
        """完整重置目录（移动文件 + 重置EXIF）"""
        self._do_reset_directory(skip_exif_reset=False, skip_confirm=False)
    
    def _do_reset_directory(self, skip_exif_reset=False, skip_confirm=False):
        """执行目录重置
        
        Args:
            skip_exif_reset: 是否跳过EXIF重置（快速复原模式）
            skip_confirm: 是否跳过确认弹窗
        """
        if not self.directory_path:
            StyledMessageBox.warning(
                self,
                self.i18n.t("messages.hint"),
                self.i18n.t("messages.select_dir_first")
            )
            return

        if not skip_confirm:
            reply = StyledMessageBox.question(
                self,
                self.i18n.t("messages.reset_confirm_title"),
                self.i18n.t("messages.reset_confirm"),
                yes_text=self.i18n.t("labels.yes"),
                no_text=self.i18n.t("labels.no")
            )

            if reply != StyledMessageBox.Yes:
                return

        self.log_text.clear()
        self.reset_btn.setEnabled(False)
        self.start_btn.setEnabled(False)

        # V4.0.4: 根据模式显示不同状态
        if skip_exif_reset:
            self._update_status(self.i18n.t("labels.quick_restoring"), COLORS['warning'])
            self._log(self.i18n.t("logs.quick_restore_start"))
        else:
            self._update_status(self.i18n.t("labels.resetting"), COLORS['warning'])
            self._log(self.i18n.t("logs.reset_start"))

        directory_path = self.directory_path
        i18n = self.i18n
        log_signal = self.reset_log_signal
        complete_signal = self.reset_complete_signal
        error_signal = self.reset_error_signal
        _skip_exif_reset = skip_exif_reset  # 传递给线程

        def run_reset():
            restore_stats = {'restored': 0, 'failed': 0}
            exif_stats = {'success': 0, 'failed': 0}

            def emit_log(msg):
                log_signal.emit(msg)

            try:
                from tools.exiftool_manager import get_exiftool_manager
                from tools.find_bird_util import reset
                import shutil

                exiftool_mgr = get_exiftool_manager()
                
                # V4.0.5: 先清理所有子目录（burst_XXX、鸟种目录等）
                emit_log(i18n.t("logs.reset_step0"))
                rating_dirs = ['3star_excellent', '2star_good', '1star_average', '0star_reject',
                               '3星_优选', '2星_良好', '1星_普通', '0星_放弃']
                subdir_stats = {'dirs_removed': 0, 'files_restored': 0}
                
                for rating_dir in rating_dirs:
                    rating_path = os.path.join(directory_path, rating_dir)
                    if not os.path.exists(rating_path):
                        continue
                    
                    for entry in os.listdir(rating_path):
                        entry_path = os.path.join(rating_path, entry)
                        if os.path.isdir(entry_path):
                            # 递归将所有文件移回评分目录
                            for root, dirs, files in os.walk(entry_path):
                                for filename in files:
                                    src = os.path.join(root, filename)
                                    dst = os.path.join(rating_path, filename)
                                    if os.path.isfile(src):
                                        try:
                                            if os.path.exists(dst):
                                                os.remove(dst)
                                            shutil.move(src, dst)
                                            subdir_stats['files_restored'] += 1
                                        except Exception as e:
                                            emit_log(i18n.t("logs.move_failed", filename=filename, error=e))
                            
                            # 删除子目录
                            try:
                                if os.path.exists(entry_path):
                                    shutil.rmtree(entry_path)
                                subdir_stats['dirs_removed'] += 1
                            except Exception as e:
                                emit_log(i18n.t("logs.burst_clean_failed", entry=entry, error=e))
                
                if subdir_stats['dirs_removed'] > 0:
                    emit_log(i18n.t("logs.burst_cleaned", dirs=subdir_stats['dirs_removed'], files=subdir_stats['files_restored']))
                else:
                    emit_log(i18n.t("logs.burst_no_clean"))

                emit_log(i18n.t("logs.reset_step1"))
                restore_stats = exiftool_mgr.restore_files_from_manifest(
                    directory_path, log_callback=emit_log, i18n=i18n
                )

                restored_count = restore_stats.get('restored', 0)
                if restored_count > 0:
                    emit_log(i18n.t("logs.restored_files", count=restored_count))
                
                # V4.0.5: Manifest 可能不包含所有文件，扫描评分目录将残留文件移回根目录
                fallback_restored = 0
                for rating_dir in rating_dirs:
                    rating_path = os.path.join(directory_path, rating_dir)
                    if not os.path.exists(rating_path):
                        continue
                    
                    for filename in os.listdir(rating_path):
                        src = os.path.join(rating_path, filename)
                        dst = os.path.join(directory_path, filename)
                        if os.path.isfile(src):
                            try:
                                if os.path.exists(dst):
                                    os.remove(dst)
                                shutil.move(src, dst)
                                fallback_restored += 1
                            except Exception as e:
                                emit_log(i18n.t("logs.move_failed", filename=filename, error=e))
                
                if fallback_restored > 0:
                    emit_log(i18n.t("logs.restored_files", count=fallback_restored))
                
                total_restored = restored_count + fallback_restored
                if total_restored == 0:
                    emit_log(i18n.t("logs.no_files_to_restore"))

                # V4.0.4: 根据模式决定是否重置EXIF
                if _skip_exif_reset:
                    emit_log("\n" + i18n.t("logs.skip_exif_reset"))
                    success = True
                else:
                    emit_log("\n" + i18n.t("logs.reset_step2"))
                    success = reset(directory_path, log_callback=emit_log, i18n=i18n)
                
                # V3.9: 删除评分目录（所有文件已移走）
                emit_log(i18n.t("logs.reset_step3"))
                deleted_dirs = 0
                for rating_dir in rating_dirs:
                    rating_path = os.path.join(directory_path, rating_dir)
                    if os.path.exists(rating_path) and os.path.isdir(rating_path):
                        try:
                            shutil.rmtree(rating_path)
                            emit_log(i18n.t("logs.empty_dir_deleted", dir=rating_dir))
                            deleted_dirs += 1
                        except Exception as e:
                            emit_log(i18n.t("logs.empty_dir_delete_failed", dir=rating_dir, error=e))
                
                # V4.0.5: 清理 .superpicky 隐藏目录和 manifest 文件
                superpicky_dir = os.path.join(directory_path, ".superpicky")
                if os.path.exists(superpicky_dir):
                    try:
                        shutil.rmtree(superpicky_dir)
                        emit_log("  ✅ .superpicky/")
                        deleted_dirs += 1
                    except Exception:
                        # 尝试系统命令强制删除
                        try:
                            import subprocess
                            subprocess.run(['rm', '-rf', superpicky_dir], check=True)
                            emit_log("  ✅ .superpicky/ (force)")
                            deleted_dirs += 1
                        except Exception as e2:
                            emit_log(f"  ⚠️ .superpicky 删除失败: {e2}")
                
                manifest_file = os.path.join(directory_path, ".superpicky_manifest.json")
                if os.path.exists(manifest_file):
                    try:
                        os.remove(manifest_file)
                        emit_log("  ✅ .superpicky_manifest.json")
                    except Exception as e:
                        emit_log(f"  ⚠️ manifest 删除失败: {e}")
                
                # 清理 macOS ._burst_XXX 残留文件
                for filename in os.listdir(directory_path):
                    if filename.startswith('._burst_') or filename.startswith('._其他') or filename.startswith('._栗'):
                        try:
                            os.remove(os.path.join(directory_path, filename))
                        except Exception:
                            pass
                
                if deleted_dirs > 0:
                    emit_log(i18n.t("logs.empty_dirs_cleaned", count=deleted_dirs))
                else:
                    emit_log(i18n.t("logs.no_empty_dirs"))

                emit_log("\n" + i18n.t("logs.reset_complete"))
                complete_signal.emit(success, restore_stats, exif_stats)

            except Exception as e:
                import traceback
                error_msg = str(e)
                emit_log(f"\n{i18n.t('errors.error_title')}: {error_msg}")
                traceback.print_exc()
                error_signal.emit(error_msg)

        threading.Thread(target=run_reset, daemon=True).start()

    def _on_reset_complete(self, success, restore_stats=None, exif_stats=None):
        """重置完成"""
        if success:
            self._update_status(self.i18n.t("labels.ready"), COLORS['accent'])
            self._log(self.i18n.t("messages.reset_complete_log"))

            msg_parts = [self.i18n.t("messages.reset_complete_msg") + "\n"]

            if restore_stats:
                restored = restore_stats.get('restored', 0)
                if restored > 0:
                    msg_parts.append(self.i18n.t("messages.files_restored", count=restored))

            if exif_stats:
                exif_success = exif_stats.get('success', 0)
                if exif_success > 0:
                    msg_parts.append(self.i18n.t("messages.exif_reset_count", count=exif_success))

            msg_parts.append("\n" + self.i18n.t("messages.ready_for_analysis"))

            self._show_message(
                self.i18n.t("messages.reset_complete_title"),
                "\n".join(msg_parts),
                "info"
            )
        else:
            self._update_status(self.i18n.t("labels.error"), COLORS['error'])
            self._log(self.i18n.t("messages.reset_failed_log"))

        self.reset_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self._check_report_csv()

    def _on_reset_error(self, error_msg):
        """重置错误"""
        self._log(f"Error: {error_msg}", "error")
        self._update_status("Error", COLORS['error'])
        self._show_message(
            self.i18n.t("errors.error_title"),
            error_msg,
            "error"
        )
        self.reset_btn.setEnabled(True)
        self.start_btn.setEnabled(True)

    @Slot()
    def _open_post_adjustment(self):
        """打开重新评星对话框"""
        if not self.directory_path:
            self._show_message(
                self.i18n.t("messages.hint"),
                self.i18n.t("messages.select_dir_first"),
                "warning"
            )
            return

        report_path = os.path.join(self.directory_path, ".superpicky", "report.db")
        if not os.path.exists(report_path):
            StyledMessageBox.warning(
                self,
                self.i18n.t("messages.hint"),
                self.i18n.t("messages.no_report_csv")
            )
            return

        from .post_adjustment_dialog import PostAdjustmentDialog
        dialog = PostAdjustmentDialog(
            self,
            self.directory_path,
            current_sharpness=self.sharp_slider.value(),
            current_nima=self.nima_slider.value() / 10.0,
            on_complete_callback=self._on_post_adjustment_complete,
            log_callback=self._log
        )
        dialog.exec()

    def _on_post_adjustment_complete(self):
        """重新评星完成回调"""
        self._log(self.i18n.t("messages.post_adjust_complete"))

    @Slot()
    def _show_advanced_settings(self):
        """显示高级设置"""
        from .advanced_settings_dialog import AdvancedSettingsDialog
        dialog = AdvancedSettingsDialog(self)
        result = dialog.exec()
        
        # V4.2: 如果用户保存了设置，更新主窗口的变量并显示新配置
        if result:
            # 重新加载配置
            self.config = get_advanced_config()
            # 更新 ai_confidence 变量
            self.ai_confidence = int(self.config.min_confidence * 100)
            # 在控制台显示更新后的设置
            self._log(f"✅ 参数设置已更新:")
            self._log(f"   检测敏感度: {self.ai_confidence}%")
            self._log(f"   最低锐度: {self.config.min_sharpness}")
            self._log(f"   最低美学: {self.config.min_nima}")
            self._log(f"   识别确信度: {self.config.birdid_confidence}%")

    def _change_language(self, lang_code):
        """切换界面语言"""
        from ui.custom_dialogs import StyledMessageBox
        
        # 更新菜单选中状态
        for code, action in self.lang_actions.items():
            action.setChecked(code == lang_code)
        
        # 保存设置
        self.config.set_language(lang_code)
        if self.config.save():
            # 根据目标语言显示对应的提示
            if lang_code == "en":
                title = "Language Changed"
                msg = "Language changed. Restart the app to take effect."
            else:
                title = "语言已更改"
                msg = "界面语言已更改，重启应用后生效。"
            StyledMessageBox.information(self, title, msg)

    @Slot()
    def _show_about(self):
        """显示关于对话框"""
        from .about_dialog import AboutDialog
        dialog = AboutDialog(self, self.i18n)
        dialog.exec()

    @Slot()
    def _toggle_birdid_dock(self, checked):
        """显示/隐藏识鸟停靠面板"""
        if hasattr(self, 'birdid_dock'):
            self.birdid_dock.setVisible(checked)



    def _auto_start_birdid_server(self):
        """自动启动识鸟 API 服务器（使用服务器管理器） - 在后台线程中运行"""
        import threading
        
        def start_server_task():
            try:
                from server_manager import get_server_status, start_server_daemon
                
                # 检查是否已有服务器在运行
                status = get_server_status()
                if status['healthy']:
                    self.log_signal.emit(self.i18n.t("server.api_reused"), "success")
                    return
                
                # 启动服务器（守护进程模式）
                success, msg, pid = start_server_daemon(log_callback=lambda m: print(m))
                
                if success:
                    self.log_signal.emit(self.i18n.t("server.api_auto_started", port=5156), "success")
                else:
                    self.log_signal.emit(self.i18n.t("server.start_failed", error=msg), "warning")
                    
            except Exception as e:
                self.log_signal.emit(self.i18n.t("server.start_failed", error=str(e)), "warning")
        
        # 在后台线程中启动服务器，不阻塞UI
        thread = threading.Thread(target=start_server_task, daemon=True)
        thread.start()

    def _stop_birdid_server(self):
        """停止识鸟 API 服务器（使用服务器管理器）"""
        try:
            from server_manager import stop_server
            success, msg = stop_server()
            if success:
                self._log(self.i18n.t("server.api_stopped"), "info")
            else:
                self._log(f"停止服务器失败: {msg}", "warning")
        except Exception as e:
            self._log(f"停止服务器异常: {e}", "error")

    # ========== 辅助方法 ==========

    def _log(self, message, tag=None):
        """输出日志"""
        from datetime import datetime
        
        # 线程安全检查：如果在非主线程中调用，通过信号发送（修复 preloading_models 导致的 Crash）
        # tag 可能是 None，但 Signal(str, str) 不接受 None，所以转为空字符串
        if QThread.currentThread() != self.thread():
            self.log_signal.emit(message, tag if tag else "")
            return

        print(message)

        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)

        # 根据标签选择颜色
        if tag == "error":
            color = LOG_COLORS['error']
        elif tag == "warning":
            color = LOG_COLORS['warning']
        elif tag == "success":
            color = LOG_COLORS['success']
        elif tag == "info":
            color = LOG_COLORS['info']
        else:
            color = LOG_COLORS['default']

        # 时间戳
        timestamp = datetime.now().strftime("%H:%M:%S")
        time_color = LOG_COLORS['time']

        # V3.9: 格式化消息（转义 HTML 特殊字符，防止 < > & 被解释为 HTML）
        import html
        html_message = html.escape(message).replace('\n', '<br>')

        # 对于简短消息添加时间戳
        if len(message) < 100 and '\n' not in message:
            cursor.insertHtml(
                f'<span style="color: {time_color};">{timestamp}</span> '
                f'<span style="color: {color};">{html_message}</span><br>'
            )
        else:
            cursor.insertHtml(f'<span style="color: {color};">{html_message}</span><br>')

        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    def _show_initial_help(self):
        """显示初始帮助信息"""
        t = self.i18n.t
        from constants import APP_VERSION
        help_text = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {t("help.welcome_title", version=APP_VERSION)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{t("help.usage_steps_title")}
  1. {t("help.step1")}
  2. {t("help.step2")}
  3. {t("help.step3")}
  4. {t("help.step4")}

{t("help.rating_rules_title")}
  {t("help.rule_3_star")}
    {t("help.rule_picked", percentage=self.config.picked_top_percentage)}
  {t("help.rule_2_star")}
  {t("help.rule_1_star")}
  {t("help.rule_0_star")}
  {t("help.rule_flying")}
  {t("help.rule_focus")}
  {t("help.rule_exposure")}
  {t("help.burst_info")}

{t("help.ready")}"""
        self._log(help_text)

    def _format_statistics_report(self, stats):
        """格式化统计报告"""
        t = self.i18n.t
        total = stats.get('total', 0)
        star_3 = stats.get('star_3', 0)
        star_2 = stats.get('star_2', 0)
        star_1 = stats.get('star_1', 0)
        star_0 = stats.get('star_0', 0)
        no_bird = stats.get('no_bird', 0)
        total_time = stats.get('total_time', 0)
        avg_time = stats.get('avg_time', 0)
        picked = stats.get('picked', 0)
        flying = stats.get('flying', 0)

        bird_total = star_3 + star_2 + star_1 + star_0

        report = "\n" + "━" * 50 + "\n"
        report += f"  {t('report.title')}\n"
        report += "━" * 50 + "\n\n"

        report += t("report.total_photos", total=total) + "\n"
        report += t("report.total_time", time_sec=total_time, time_min=total_time/60) + "\n"
        report += t("report.avg_time", avg=avg_time) + "\n\n"

        if total > 0:
            report += f"  ⭐⭐⭐  {star_3:>4}  ({star_3/total*100:>5.1f}%)\n"
            if picked > 0 and star_3 > 0:
                report += f"    └─ 🏆  {picked} ({picked/star_3*100:.0f}%)\n"
            report += f"  ⭐⭐    {star_2:>4}  ({star_2/total*100:>5.1f}%)\n"
            report += f"  ⭐      {star_1:>4}  ({star_1/total*100:>5.1f}%)\n"
            if star_0 > 0:
                report += f"  0⭐     {star_0:>4}  ({star_0/total*100:>5.1f}%)\n"
            report += f"  ❌      {no_bird:>4}  ({no_bird/total*100:>5.1f}%)\n\n"
            report += t("report.bird_total", count=bird_total, percent=bird_total/total*100) + "\n"

            if flying > 0:
                report += f"{t('help.rule_flying')}: {flying}\n"
            
            # V4.2: 精焦统计（红色标签）
            focus_precise = stats.get('focus_precise', 0)
            if focus_precise > 0:
                report += f"{t('help.rule_focus')}: {focus_precise}\n"
            
            # V4.2: 识别鸟种统计 (language-aware)
            bird_species = stats.get('bird_species', [])
            if bird_species:
                # Pick the correct language name based on current locale
                is_chinese = self.i18n.current_lang.startswith('zh')
                species_names = []
                for sp in bird_species:
                    if isinstance(sp, dict):
                        name = sp.get('cn_name', '') if is_chinese else sp.get('en_name', '')
                        # Fallback to the other language if preferred is empty
                        if not name:
                            name = sp.get('en_name', '') if is_chinese else sp.get('cn_name', '')
                        if name:
                            species_names.append(name)
                    else:
                        # Legacy support: if it's still a string (old format)
                        species_names.append(str(sp))
                if species_names:
                    report += "\n" + t("logs.bird_species_identified", count=len(species_names), species=', '.join(species_names))

        report += "\n" + "━" * 50
        return report

    def _show_lightroom_guide(self):
        """显示 Lightroom 指南"""
        t = self.i18n.t
        guide = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {t("lightroom_guide.title")}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{t("lightroom_guide.method1_title")}
  1. {t("lightroom_guide.method1_step1")}
  2. {t("lightroom_guide.method1_step2")}
  3. {t("lightroom_guide.method1_step3")}
  4. {t("lightroom_guide.method1_step4")}
  5. {t("lightroom_guide.method1_step5")}

{t("lightroom_guide.sort_title")}
  · {t("lightroom_guide.sort_step3_city")}
  · {t("lightroom_guide.sort_step3_state")}
  · {t("lightroom_guide.field_caption")}

{t("lightroom_guide.debug_title")}
  {t("lightroom_guide.debug_tip")}
  · {t("lightroom_guide.debug_explain1")}
  · {t("lightroom_guide.debug_explain2")}
  · {t("lightroom_guide.debug_explain3")}
  · {t("lightroom_guide.debug_explain4")}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        self._log(guide)

    def _play_completion_sound(self):
        """播放完成音效"""
        sound_path = os.path.join(
            os.path.dirname(__file__), "..",
            "img", "toy-story-short-happy-audio-logo-short-cartoony-intro-outro-music-125627.mp3"
        )

        if os.path.exists(sound_path) and sys.platform == 'darwin':
            try:
                subprocess.Popen(
                    ['afplay', sound_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                pass

    def closeEvent(self, event):
        """窗口关闭事件"""
        # V4.0: 后台模式不停止服务器
        if getattr(self, '_background_mode', False):
            print("✅ 后台模式：服务器继续运行")
            if hasattr(self, 'tray_icon') and self.tray_icon:
                self.tray_icon.hide()
            event.accept()
            return
        
        if self.worker and self.worker.is_alive():
            reply = StyledMessageBox.question(
                self,
                self.i18n.t("messages.exit_title"),
                self.i18n.t("messages.exit_confirm"),
                yes_text=self.i18n.t("buttons.cancel"),
                no_text=self.i18n.t("labels.yes")
            )

            if reply == StyledMessageBox.No:  # 用户点击"是"退出
                self.worker._stop_event.set()
                self.worker._stop_caffeinate()  # V3.8.1: 确保终止 caffeinate 进程
                self._stop_birdid_server()  # V4.0: 停止识鸟 API 服务
                event.accept()
            else:
                event.ignore()
        else:
            self._stop_birdid_server()  # V4.0: 停止识鸟 API 服务
            event.accept()

    # ========== V4.2: 模型预加载功能 ==========

    def _preload_all_models(self):
        """后台预加载所有AI模型（不阻塞UI）"""
        import threading
        
        def preload_task():
            try:
                # 使用信号发送日志，确保线程安全
                self.log_signal.emit(self.i18n.t("preload.preloading_models"), "info")
                
                # 1. YOLO 检测模型 - 使用GUI日志回调
                from ai_model import load_yolo_model
                load_yolo_model(log_callback=lambda msg, tag="info": self.log_signal.emit(msg, tag))
                self.log_signal.emit(self.i18n.t("preload.yolo_loaded"), "success")
                
                # 2. 关键点检测模型
                from core.keypoint_detector import get_keypoint_detector
                kp_detector = get_keypoint_detector()
                kp_detector.load_model()
                self.log_signal.emit(self.i18n.t("preload.keypoint_loaded"), "success")
                
                # 3. 飞版检测模型
                from core.flight_detector import get_flight_detector
                flight_detector = get_flight_detector()
                flight_detector.load_model()
                self.log_signal.emit(self.i18n.t("preload.flight_loaded"), "success")
                
                # 4. 识鸟模型
                from birdid.bird_identifier import get_bird_model
                get_bird_model()
                self.log_signal.emit(self.i18n.t("preload.birdid_loaded"), "success")
                
                self.log_signal.emit(self.i18n.t("preload.preload_complete"), "success")
                
            except Exception as e:
                self.log_signal.emit(self.i18n.t("preload.preload_failed", error=str(e)), "warning")
        
        # 在后台线程中执行，不阻塞UI
        thread = threading.Thread(target=preload_task, daemon=True)
        thread.start()

    # ========== V4.0.1: 更新检测功能 ==========

    def _check_for_updates(self, silent=False):
        """检查更新
        
        Args:
            silent: 如果为 True，只在有更新时显示弹窗（用于启动时自动检查）
        """
        import threading
        
        if not silent:
            self._log(self.i18n.t("update.checking"), "info")
        
        def _do_check():
            try:
                from tools.update_checker import UpdateChecker
                checker = UpdateChecker()  # 使用 update_checker.CURRENT_VERSION
                has_update, update_info = checker.check_for_updates()
                print(f"[DEBUG] 更新检查完成: has_update={has_update}, silent={silent}")
                
                # 静默模式下，只有有更新时才弹窗
                if silent and not has_update:
                    print("[DEBUG] 静默模式，无更新，跳过弹窗")
                    return
                    
                # 使用信号发送到主线程
                self._update_signals.update_check_done.emit(has_update, update_info)
            except Exception as e:
                import traceback
                print(f"⚠️ 更新检测失败: {e}")
                traceback.print_exc()
                # 静默模式下不显示错误
                if not silent:
                    error_info = {'error': str(e), 'current_version': '4.0.0', 'version': '检查失败'}
                    self._update_signals.update_check_done.emit(False, error_info)
        
        # 在后台线程执行
        thread = threading.Thread(target=_do_check, daemon=True)
        thread.start()

    def _show_update_result_dialog(self, has_update: bool, update_info):
        """显示更新检测结果对话框"""
        try:
            print("[DEBUG] _show_update_result_dialog 开始执行")
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
            import webbrowser
            
            dialog = QDialog(self)
            dialog.setWindowTitle(self.i18n.t("update.window_title"))
            dialog.setMinimumWidth(420)
            dialog.setStyleSheet(f"""
                QDialog {{
                    background-color: {COLORS['bg_primary']};
                }}
                QLabel {{
                    color: {COLORS['text_primary']};
                    font-size: 13px;
                }}
            """)
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(24, 24, 24, 24)
            layout.setSpacing(12)
            
            # 获取版本信息
            current_version = update_info.get('current_version', '4.0.0') if update_info else '4.0.0'
            latest_version = update_info.get('version', '未知') if update_info else '未知'
            has_error = update_info.get('error') if update_info else None
            
            if has_error:
                title = QLabel(self.i18n.t("update.check_failed_title"))
                title.setStyleSheet(f"color: {COLORS['warning']}; font-size: 18px; font-weight: 600;")
            elif has_update:
                title = QLabel(self.i18n.t("update.new_version_found"))
                title.setStyleSheet(f"color: {COLORS['accent']}; font-size: 18px; font-weight: 600;")
            else:
                title = QLabel(self.i18n.t("update.up_to_date_title"))
                title.setStyleSheet(f"color: {COLORS['success']}; font-size: 18px; font-weight: 600;")
            layout.addWidget(title)
            
            layout.addSpacing(4)
            
            # 版本信息区域
            version_frame = QFrame()
            version_frame.setStyleSheet(f"background-color: {COLORS['bg_elevated']}; border-radius: 8px;")
            version_layout = QVBoxLayout(version_frame)
            version_layout.setContentsMargins(16, 12, 16, 12)
            version_layout.setSpacing(8)
            
            # 当前版本
            current_row = QHBoxLayout()
            current_label = QLabel(self.i18n.t("update.current_version_label"))
            current_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
            current_row.addWidget(current_label)
            current_row.addStretch()
            current_value = QLabel(f"V{current_version}")
            current_value.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px; font-weight: 500;")
            current_row.addWidget(current_value)
            version_layout.addLayout(current_row)
            
            # 发布版本
            latest_row = QHBoxLayout()
            latest_label = QLabel(self.i18n.t("update.latest_version_label"))
            latest_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
            latest_row.addWidget(latest_label)
            latest_row.addStretch()
            latest_value = QLabel(f"V{latest_version}")
            if has_update:
                latest_value.setStyleSheet(f"color: {COLORS['accent']}; font-size: 13px; font-weight: 600;")
            else:
                latest_value.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px; font-weight: 500;")
            latest_row.addWidget(latest_value)
            version_layout.addLayout(latest_row)
            
            layout.addWidget(version_frame)
            
            # 提示和下载按钮
            if not has_error:
                msg = QLabel(self.i18n.t("update.download_hint"))
                msg.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: 12px;")
                layout.addWidget(msg)
                
                layout.addSpacing(8)
                
                download_url = "https://superpicky.jamesphotography.com.au/#download"
                
                # 下载按钮区域
                btn_frame = QFrame()
                btn_frame.setStyleSheet(f"background-color: {COLORS['bg_elevated']}; border-radius: 8px;")
                btn_layout = QHBoxLayout(btn_frame)
                btn_layout.setContentsMargins(16, 12, 16, 12)
                btn_layout.setSpacing(12)
                
                mac_btn = QPushButton(self.i18n.t("update.mac_version"))
                mac_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLORS['accent']};
                        color: {COLORS['bg_void']};
                        border: none;
                        border-radius: 6px;
                        padding: 10px 16px;
                        font-size: 13px;
                        font-weight: 500;
                    }}
                    QPushButton:hover {{
                        background-color: #00e6b8;
                    }}
                """)
                mac_btn.clicked.connect(lambda: webbrowser.open(download_url))
                btn_layout.addWidget(mac_btn)
                
                win_btn = QPushButton(self.i18n.t("update.windows_version"))
                win_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLORS['bg_card']};
                        border: 1px solid {COLORS['border']};
                        color: {COLORS['text_secondary']};
                        border-radius: 6px;
                        padding: 10px 16px;
                        font-size: 13px;
                        font-weight: 500;
                    }}
                    QPushButton:hover {{
                        border-color: {COLORS['text_muted']};
                        color: {COLORS['text_primary']};
                    }}
                """)
                win_btn.clicked.connect(lambda: webbrowser.open(download_url))
                btn_layout.addWidget(win_btn)
                
                layout.addWidget(btn_frame)
            
            layout.addSpacing(8)
            
            # 关闭按钮
            close_layout = QHBoxLayout()
            close_layout.addStretch()
            
            close_btn = QPushButton(self.i18n.t("update.close"))
            close_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['bg_card']};
                    border: 1px solid {COLORS['border']};
                    color: {COLORS['text_secondary']};
                    border-radius: 6px;
                    padding: 8px 24px;
                    font-size: 13px;
                }}
                QPushButton:hover {{
                    border-color: {COLORS['text_muted']};
                    color: {COLORS['text_primary']};
                }}
            """)
            close_btn.clicked.connect(dialog.accept)
            close_layout.addWidget(close_btn)
            
            layout.addLayout(close_layout)
            
            print("[DEBUG] 即将显示弹窗")
            dialog.exec()
            print("[DEBUG] 弹窗已关闭")
            
        except Exception as e:
            import traceback
            print(f"[ERROR] 显示更新弹窗失败: {e}")
            traceback.print_exc()

    # ========== V4.3: 摄影水平预设 ==========
    
    def _show_skill_level_dialog(self):
        """菜单打开水平选择对话框"""
        dialog = SkillLevelDialog(self.i18n, self)
        dialog.level_selected.connect(self._on_skill_level_selected)
        dialog.exec()
    
    def _show_first_run_skill_level_dialog(self):
        """首次运行：显示水平选择对话框"""
        dialog = SkillLevelDialog(self.i18n, self)
        dialog.level_selected.connect(self._on_skill_level_selected)
        dialog.exec()
    
    def _on_skill_level_selected(self, level_key: str):
        """处理水平选择"""
        # 保存设置
        self.config.set_skill_level(level_key)
        self.config.set_is_first_run(False)
        self.config.save()
        
        # 应用阈值到滑块
        self._apply_skill_level_thresholds(level_key)
        
        # 更新水平显示标签
        self._update_skill_level_label(level_key)
        
        print(f"✅ 已选择摄影水平: {level_key}")
    
    def _apply_skill_level_thresholds(self, level_key: str):
        """应用水平预设的阈值到滑块"""
        sharpness, aesthetics = get_skill_level_thresholds(level_key, self.config)
        
        # 阻止信号防止触发 _check_custom_mode
        self._applying_preset = True
        
        self.sharp_slider.blockSignals(True)
        self.sharp_slider.setValue(int(sharpness))
        self.sharp_slider.blockSignals(False)
        self.sharp_value.setText(str(int(sharpness)))
        
        self.nima_slider.blockSignals(True)
        self.nima_slider.setValue(int(aesthetics * 10))
        self.nima_slider.blockSignals(False)
        self.nima_value.setText(f"{aesthetics:.1f}")
        
        self._applying_preset = False
        
        # 更新水平显示标签
        self._update_skill_level_label(level_key)
    
    def _check_custom_mode(self):
        """检查当前滑块值是否与任何预设匹配，如果不匹配则切换到自选模式"""
        # 如果正在应用预设，跳过检查
        if getattr(self, '_applying_preset', False):
            return
        
        current_sharpness = self.sharp_slider.value()
        current_aesthetics = self.nima_slider.value() / 10.0
        
        # 检查是否匹配某个预设
        for level_key, preset in SKILL_PRESETS.items():
            if (current_sharpness == preset["sharpness"] and 
                abs(current_aesthetics - preset["aesthetics"]) < 0.05):
                # 匹配预设
                if self.config.skill_level != level_key:
                    self.config.set_skill_level(level_key)
                    self.config.save()
                    self._update_skill_level_label(level_key)
                return
        
        # 不匹配任何预设，切换到自选模式
        # 不匹配任何预设，切换到自选模式
        if self.config.skill_level != "custom":
            self.config.set_skill_level("custom")
            self._update_skill_level_label("custom")
            print(f"🎛️ 已切换到自选模式")
            
        # 始终更新自选值并保存
        self.config.set_custom_sharpness(current_sharpness)
        self.config.set_custom_aesthetics(current_aesthetics)
        self.config.save()
    
    def _update_skill_level_label(self, level_key: str):
        """更新主界面的水平显示标签"""
        if hasattr(self, 'skill_level_label'):
            level_names = {
                "beginner": self.i18n.t("skill_level.beginner"),
                "intermediate": self.i18n.t("skill_level.intermediate"),
                "master": self.i18n.t("skill_level.master"),
                "custom": self.i18n.t("skill_level.custom")
            }
            level_name = level_names.get(level_key, level_key)
            self.skill_level_label.setText(self.i18n.t("skill_level.current_label", level=level_name))
