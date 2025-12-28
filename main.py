#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky - 简化版 (Pure Tkinter, 无PyQt依赖)
Version: 3.3 - 多鸟检测优化 + 目录分类
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import csv
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from find_bird_util import reset, raw_to_jpeg
from ai_model import load_yolo_model, detect_and_draw_birds
from utils import write_to_csv, log_message
from exiftool_manager import get_exiftool_manager
from advanced_config import get_advanced_config
from advanced_settings_dialog import AdvancedSettingsDialog
from post_adjustment_dialog import PostAdjustmentDialog
from i18n import get_i18n

# 尝试导入主题和图片库
try:
    from ttkthemes import ThemedTk
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False
    print("提示: 安装 ttkthemes 可获得更美观的主题 (pip install ttkthemes)")

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("提示: 需要安装 Pillow 才能显示图标 (pip install Pillow)")

# V3.3: 文件夹名称映射（支持所有星级）
RATING_FOLDER_NAMES = {
    3: "3星_优选",
    2: "2星_良好",
    1: "1星_普通",
    0: "0星_放弃",
    -1: "0星_放弃"
}



class WorkerThread(threading.Thread):
    """处理线程"""

    def __init__(self, dir_path, ui_settings, progress_callback, finished_callback, log_callback, i18n=None):
        super().__init__(daemon=True)
        self.dir_path = dir_path
        self.ui_settings = ui_settings
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
        self.log_callback = log_callback
        self.i18n = i18n
        self._stop_event = threading.Event()
        self.caffeinate_process = None  # caffeinate进程（防休眠）

        # 统计数据
        self.stats = {
            'total': 0,
            'star_3': 0,  # 优选照片（3星）
            'picked': 0,  # 精选照片（3星中美学+锐度双Top的）
            'star_2': 0,  # 良好照片（2星）
            'star_1': 0,  # 普通照片（1星）
            'star_0': 0,  # 0星照片（技术质量差）
            'no_bird': 0,  # 无鸟照片（-1星）
            'start_time': 0,
            'end_time': 0,
            'total_time': 0,
            'avg_time': 0
        }

    def _format_time(self, seconds):
        """格式化时间：秒转为 分钟+秒 格式"""
        if seconds < 60:
            if self.i18n:
                return f"{seconds:.1f}s"
            else:
                return f"{seconds:.1f}秒"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            if self.i18n:
                return f"{minutes}m{secs:.0f}s"
            else:
                return f"{minutes}分{secs:.0f}秒"

    def _start_caffeinate(self):
        """启动caffeinate防止系统休眠和屏幕保护程序"""
        try:
            # -d: 防止显示器休眠（同时阻止屏幕保护程序）
            # -i: 防止系统空闲休眠
            self.caffeinate_process = subprocess.Popen(
                ['caffeinate', '-d', '-i'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if self.i18n:
                self.log_callback(self.i18n.t("logs.caffeinate_started"))
            else:
                self.log_callback("☕ 已启动防休眠保护（处理期间Mac不会休眠或启动屏幕保护程序）")
        except Exception as e:
            if self.i18n:
                self.log_callback(self.i18n.t("logs.caffeinate_failed", error=str(e)))
            else:
                self.log_callback(f"⚠️  防休眠启动失败: {e}（不影响正常处理）")
            self.caffeinate_process = None

    def _stop_caffeinate(self):
        """停止caffeinate"""
        if self.caffeinate_process:
            try:
                self.caffeinate_process.terminate()
                self.caffeinate_process.wait(timeout=2)
                if self.i18n:
                    self.log_callback(self.i18n.t("logs.caffeinate_stopped"))
                else:
                    self.log_callback("☕ 已停止防休眠保护")
            except Exception:
                # 如果terminate失败，强制kill
                try:
                    self.caffeinate_process.kill()
                except Exception:
                    pass
            finally:
                self.caffeinate_process = None

    def run(self):
        """执行处理"""
        try:
            # 启动防休眠保护
            self._start_caffeinate()

            # 执行主要处理逻辑
            self.process_files()

            if self.finished_callback:
                self.finished_callback(self.stats)
        except Exception as e:
            self.log_callback(f"❌ 错误: {e}")
        finally:
            # 确保停止caffeinate（即使出错也要停止）
            self._stop_caffeinate()

    def process_files(self):
        """处理文件 - 调用核心处理器"""
        from core.photo_processor import (
            PhotoProcessor,
            ProcessingSettings,
            ProcessingCallbacks
        )
        
        # 转换 UI 设置为 ProcessingSettings
        settings = ProcessingSettings(
            ai_confidence=self.ui_settings[0],
            sharpness_threshold=self.ui_settings[1],
            nima_threshold=self.ui_settings[2],
            save_crop=self.ui_settings[3] if len(self.ui_settings) > 3 else False,
            normalization_mode=self.ui_settings[4] if len(self.ui_settings) > 4 else 'log_compression'
        )
        
        # 创建回调（包装日志以支持 i18n）
        callbacks = ProcessingCallbacks(
            log=self._log_wrapper,
            progress=self.progress_callback
        )
        
        # 创建核心处理器
        processor = PhotoProcessor(
            dir_path=self.dir_path,
            settings=settings,
            callbacks=callbacks
        )
        
        # 执行处理
        result = processor.process(
            organize_files=True,
            cleanup_temp=True
        )
        
        # 更新统计数据
        self.stats = result.stats
    
    def _log_wrapper(self, msg: str, level: str = "info"):
        """日志包装器 - 支持 i18n（如果需要的话）"""
        # 目前直接传递，未来可以在这里添加 i18n 翻译
        self.log_callback(msg)



class AboutWindow:
    """关于窗口"""
    def __init__(self, parent, i18n):
        self.window = tk.Toplevel(parent)
        self.i18n = i18n
        self.window.title(self.i18n.t("menu.about"))
        self.window.geometry("700x600")
        self.window.resizable(False, False)

        # 设置窗口图标（如果有的话）
        # self.window.iconbitmap("icon.ico")

        # 创建主容器
        main_frame = ttk.Frame(self.window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建滚动文本区域
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 创建文本框
        self.text = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=("Arial", 10),
            padx=10,
            pady=10
        )
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.text.yview)

        # 配置文本样式
        self.text.tag_configure("title", font=("Arial", 18, "bold"), spacing1=10)
        self.text.tag_configure("version", font=("Arial", 10), foreground="gray")
        self.text.tag_configure("section", font=("Arial", 12, "bold"), spacing1=15, spacing3=5)
        self.text.tag_configure("subsection", font=("Arial", 11, "bold"), spacing1=10, spacing3=5)
        self.text.tag_configure("body", font=("Arial", 10), spacing1=5)
        self.text.tag_configure("link", font=("Arial", 10), foreground="blue", underline=True)
        self.text.tag_configure("code", font=("Courier", 9), background="#f0f0f0")

        # 填充内容
        self._populate_content()

        # 禁止编辑
        self.text.config(state=tk.DISABLED)

        # 添加关闭按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=(10, 0))

        close_btn = ttk.Button(btn_frame, text="关闭", command=self.window.destroy, width=15)
        close_btn.pack()

        # 窗口居中
        self._center_window()

    def _populate_content(self):
        """填充关于窗口的内容"""
        content = """慧眼选鸟 (SuperPicky)

版本: V3.3
发布日期: 2025-10-28

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👨‍💻 作者信息

开发者: 詹姆斯·于震 (James Yu)
网站: www.jamesphotography.com.au
YouTube: youtube.com/@JamesZhenYu
邮箱: james@jamesphotography.com.au

关于作者:
詹姆斯·于震是一位澳籍华裔职业摄影师，著有畅销三部曲《詹姆斯的风景摄影笔记》（总销量超10万册），他开发慧眼选鸟以提高鸟类摄影师后期筛选效率，让摄影师将更多时间专注于拍摄而非选片。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 软件简介

慧眼选鸟是一款专为鸟类摄影师设计的智能照片筛选工具。

✓ 自动识别鸟类 - 使用先进的AI技术检测照片中的鸟类主体
✓ 多维度评分 - 综合锐度、美学、技术质量等指标智能评级
✓ 精选推荐 - 自动标记美学与锐度双优的顶级作品
✓ 无缝集成 - 直接写入EXIF元数据，与Lightroom完美配合
✓ 批量处理 - 支持RAW格式，高效处理大量照片

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 使用的开源技术

慧眼选鸟基于以下优秀的开源项目构建：

1. Ultralytics YOLOv11
   用于鸟类目标检测与分割，精确识别照片中的鸟类位置和轮廓。
   许可证: AGPL-3.0
   项目地址: github.com/ultralytics/ultralytics

2. PyIQA (Image Quality Assessment)
   用于图像质量评估，主要使用NIMA美学评分模型。
   许可证: CC BY-NC-SA 4.0 (非商业使用)
   项目地址: github.com/chaofengc/IQA-PyTorch
   引用: Chen et al., "TOPIQ", IEEE TIP, 2024

3. ExifTool
   用于EXIF元数据读写，将评分和旗标写入RAW文件。
   许可证: Perl Artistic License / GPL
   项目地址: exiftool.org

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📜 版权与许可

版权所有 © 2024-2025 詹姆斯·于震 (James Yu)

慧眼选鸟是基于开源技术开发的非商业用途摄影工具。

使用条款:
✓ 允许: 个人使用、教育学习、分享推荐
✗ 禁止: 商业用途、销售盈利、移除版权

免责声明:
本软件按"现状"提供，不提供任何保证。作者不对使用本软件产生的任何后果负责。

重要提示:
• AI模型可能误判，请勿完全依赖自动评分
• 处理前请备份原始文件
• 重要项目建议先小批量测试

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔄 开源声明

慧眼选鸟遵循其依赖项目的开源许可要求：

• AGPL-3.0 (YOLOv11): 修改并分发需开源，网络服务需提供源代码
• CC BY-NC-SA 4.0 (PyIQA): 限制非商业使用

商业使用: 如需商业用途，请联系作者及相关开源项目获取商业许可

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🙏 致谢

感谢以下项目和开发者：
• Ultralytics团队 - 提供了卓越的YOLOv11目标检测框架
• Chaofeng Chen和Jiadi Mo - 开发了PyIQA图像质量评估工具箱
• Phil Harvey - 开发了强大的ExifTool元数据处理工具
• 所有鸟类摄影师 - 你们的反馈和建议推动了慧眼选鸟的不断改进

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📧 联系方式

如果您在使用过程中遇到问题、有改进建议，或希望合作开发：

邮箱: james@jamesphotography.com.au

詹姆斯独立开发的更多免费工具：
慧眼选鸟：AI 鸟类摄影选片工具
慧眼识鸟：AI 鸟种识别工具 （Mac/Win Lightroom 插件）
慧眼找鸟：eBird信息检索工具  Web 测试版
慧眼去星：AI 银河去星软件（Max Photoshop 插件）
图忆作品集：Tui Portfolio IOS 手机专用 
镜书：AI 旅游日记写作助手 IOS 手机专用

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

慧眼选鸟 - 让AI帮你挑选最美的瞬间 🦅📸
"""

        self.text.config(state=tk.NORMAL)
        self.text.insert("1.0", content)
        self.text.config(state=tk.DISABLED)

    def _center_window(self):
        """将窗口居中显示"""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')


class SuperPickyApp:
    def __init__(self, root):
        self.root = root

        # 初始化国际化（基于高级配置的语言设置）
        self.config = get_advanced_config()
        self.i18n = get_i18n(self.config.language)

        self.root.title(self.i18n.t("app.window_title"))
        self.root.geometry("750x700")  # V3.1: 增加窗口高度，确保所有控件可见
        self.root.minsize(700, 650)  # 设置最小尺寸
        # 允许窗口调整大小（默认行为）

        # 创建菜单栏
        self._create_menu()

        # 设置图标
        icon_path = os.path.join(os.path.dirname(__file__), "img", "icon.png")
        if os.path.exists(icon_path) and PIL_AVAILABLE:
            try:
                icon_img = Image.open(icon_path)
                icon_photo = ImageTk.PhotoImage(icon_img)
                self.root.iconphoto(True, icon_photo)
            except Exception as e:
                print(f"图标加载失败: {e}")

        self.directory_path = ""
        self.worker = None

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.show_initial_help()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.create_control_panel(main_frame)

    def create_control_panel(self, parent):
        """创建控制面板"""
        # 标题
        title = ttk.Label(parent, text=self.i18n.t("labels.app_title"), font=("Arial", 16, "bold"))
        title.pack(pady=10)

        # 目录选择
        dir_frame = ttk.LabelFrame(parent, text=self.i18n.t("labels.select_photo_dir"), padding=10)
        dir_frame.pack(fill=tk.X, padx=10, pady=5)

        self.dir_entry = ttk.Entry(dir_frame, font=("Arial", 11))
        self.dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        # V3.1: 支持粘贴路径并按回车确认
        self.dir_entry.bind('<Return>', self._on_path_entered)
        self.dir_entry.bind('<KP_Enter>', self._on_path_entered)

        ttk.Button(dir_frame, text=self.i18n.t("labels.browse"), command=self.browse_directory, width=10).pack(side=tk.LEFT)

        # 参数设置
        settings_frame = ttk.LabelFrame(parent, text=self.i18n.t("labels.rating_params"), padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)

        # V3.1: 隐藏置信度和归一化选择
        self.ai_var = tk.IntVar(value=50)
        self.norm_var = tk.StringVar(value="对数压缩(V3.1) - 大小鸟公平")

        # 鸟锐度阈值
        sharp_frame = ttk.Frame(settings_frame)
        sharp_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sharp_frame, text=self.i18n.t("labels.sharpness"), width=14, font=("Arial", 11)).pack(side=tk.LEFT)
        self.sharp_var = tk.IntVar(value=500)
        self.sharp_slider = ttk.Scale(sharp_frame, from_=100, to=1000, variable=self.sharp_var, orient=tk.HORIZONTAL)
        self.sharp_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.sharp_label = ttk.Label(sharp_frame, text="500", width=6, font=("Arial", 11))
        self.sharp_label.pack(side=tk.LEFT)
        self.sharp_slider.configure(command=lambda v: self._update_sharp_label(v))

        # 摄影美学阈值（NIMA）- V3.1: 范围4.5-5.5，默认4.8
        nima_frame = ttk.Frame(settings_frame)
        nima_frame.pack(fill=tk.X, pady=5)
        ttk.Label(nima_frame, text=self.i18n.t("labels.nima"), width=14, font=("Arial", 11)).pack(side=tk.LEFT)
        self.nima_var = tk.DoubleVar(value=5.0)
        self.nima_slider = ttk.Scale(nima_frame, from_=4.5, to=5.5, variable=self.nima_var, orient=tk.HORIZONTAL)
        self.nima_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.nima_label = ttk.Label(nima_frame, text="5.0", width=6, font=("Arial", 11))
        self.nima_label.pack(side=tk.LEFT)
        self.nima_slider.configure(command=lambda v: self.nima_label.configure(text=f"{float(v):.1f}"))

        # 进度显示
        progress_frame = ttk.LabelFrame(parent, text=self.i18n.t("labels.processing"), padding=10)
        progress_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)

        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # 日志框（V3.1: 减小固定高度，允许自适应）
        log_scroll = ttk.Scrollbar(progress_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(progress_frame, height=10, state='disabled', yscrollcommand=log_scroll.set,
                                font=("Menlo", 13), bg='#1e1e1e', fg='#d4d4d4',
                                spacing1=4, spacing2=2, spacing3=4, padx=8, pady=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)

        # 配置日志颜色
        self.log_text.tag_config("success", foreground="#00ff88")
        self.log_text.tag_config("error", foreground="#ff0066")
        self.log_text.tag_config("warning", foreground="#ffaa00")
        self.log_text.tag_config("info", foreground="#00aaff")

        # 控制按钮
        btn_frame = ttk.Frame(parent, padding=10)
        btn_frame.pack(fill=tk.X)

        button_container = ttk.Frame(btn_frame)
        button_container.pack(side=tk.RIGHT)

        ttk.Label(button_container, text="V3.3", font=("Arial", 9)).pack(side=tk.RIGHT, padx=10)

        self.reset_btn = ttk.Button(button_container, text=self.i18n.t("buttons.reset"), command=self.reset_directory, width=15, state='disabled')
        self.reset_btn.pack(side=tk.RIGHT, padx=5)

        self.post_da_btn = ttk.Button(button_container, text=self.i18n.t("buttons.post_adjust"), command=self.open_post_adjustment, width=15, state='disabled')
        self.post_da_btn.pack(side=tk.RIGHT, padx=5)

        self.start_btn = ttk.Button(button_container, text=self.i18n.t("buttons.start"), command=self.start_processing, width=15)
        self.start_btn.pack(side=tk.RIGHT, padx=5)

    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 设置菜单
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.i18n.t("menu.settings"), menu=settings_menu)
        settings_menu.add_command(label=self.i18n.t("menu.advanced_settings"), command=self.show_advanced_settings)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.i18n.t("menu.help"), menu=help_menu)
        help_menu.add_command(label=self.i18n.t("menu.about"), command=self.show_about)

    def show_advanced_settings(self):
        """显示高级设置对话框"""
        dialog = AdvancedSettingsDialog(self.root)
        dialog.show()

    def show_about(self):
        """显示关于窗口"""
        AboutWindow(self.root, self.i18n)

    def _check_report_csv(self):
        """检测目录中是否存在 report.csv，控制二次选鸟按钮状态"""
        if not self.directory_path:
            self.post_da_btn.config(state='disabled')
            return

        report_path = os.path.join(self.directory_path, "_tmp", "report.csv")
        if os.path.exists(report_path):
            self.post_da_btn.config(state='normal')
            self.log(f"📊 {self.i18n.t('messages.report_detected')}\n")
        else:
            self.post_da_btn.config(state='disabled')

    def open_post_adjustment(self):
        """打开二次选鸟对话框"""
        if not self.directory_path:
            messagebox.showwarning(self.i18n.t("messages.hint"), self.i18n.t("messages.select_dir_first"))
            return

        report_path = os.path.join(self.directory_path, "_tmp", "report.csv")
        if not os.path.exists(report_path):
            messagebox.showwarning(self.i18n.t("messages.hint"), self.i18n.t("messages.no_report_csv"))
            return

        # 打开对话框，传递当前UI的阈值设置
        PostAdjustmentDialog(
            self.root,
            self.directory_path,
            current_sharpness=self.sharp_var.get(),
            current_nima=self.nima_var.get(),
            on_complete_callback=self._on_post_adjustment_complete
        )

    def _on_post_adjustment_complete(self):
        """二次选鸟完成后的回调"""
        self.log("✅ 二次选鸟完成！评分已更新到EXIF元数据\n")

    def _update_sharp_label(self, value):
        """更新锐度滑块标签（步长50）"""
        rounded_value = round(float(value) / 50) * 50
        self.sharp_var.set(int(rounded_value))
        self.sharp_label.configure(text=f"{int(rounded_value)}")

    def _on_path_entered(self, event):
        """处理粘贴路径后按回车键事件（V3.1）"""
        directory = self.dir_entry.get().strip()
        if directory:
            # 验证目录是否存在
            if os.path.isdir(directory):
                self._handle_directory_selection(directory)
            else:
                messagebox.showerror(self.i18n.t("errors.error_title"),
                                   self.i18n.t("errors.dir_not_exist", directory=directory))
                self.log(f"❌ {self.i18n.t('errors.dir_not_exist', directory=directory)}\n", "error")

    def browse_directory(self):
        """浏览目录"""
        directory = filedialog.askdirectory(title=self.i18n.t("labels.select_photo_dir"))
        if directory:
            self._handle_directory_selection(directory)

    def _handle_directory_selection(self, directory):
        """处理目录选择"""
        self.directory_path = directory
        self.dir_entry.delete(0, tk.END)
        self.dir_entry.insert(0, directory)
        self.reset_btn.config(state='normal')
        self.log(f"📂 {self.i18n.t('messages.dir_selected', directory=directory)}\n")

        # 检测是否存在 report.csv，启用/禁用"二次选鸟"按钮
        self._check_report_csv()

    def reset_directory(self):
        """重置目录"""
        if not self.directory_path:
            messagebox.showwarning(self.i18n.t("messages.hint"), self.i18n.t("messages.select_dir_first"))
            return

        if messagebox.askyesno(self.i18n.t("messages.reset_confirm_title"), self.i18n.t("messages.reset_confirm")):
            # 清空并显示日志窗口
            self.log_text.config(state='normal')
            self.log_text.delete(1.0, tk.END)
            self.log_text.config(state='disabled')

            # 禁用按钮，防止重复操作
            self.reset_btn.config(state='disabled')
            self.start_btn.config(state='disabled')

            self.log(self.i18n.t("logs.separator"))
            self.log(self.i18n.t("logs.reset_start"))
            self.log(self.i18n.t("logs.separator"))
            self.log(self.i18n.t("logs.reset_dir", directory=self.directory_path) + "\n")

            # 强制更新UI，显示日志
            self.root.update()

            # 在后台线程执行重置操作,使用线程安全的日志回调
            def run_reset():
                try:
                    # V3.3: 先恢复文件位置（如果有 manifest）
                    exiftool_mgr = get_exiftool_manager()
                    restore_stats = exiftool_mgr.restore_files_from_manifest(
                        self.directory_path, 
                        log_callback=self.thread_safe_log
                    )
                    
                    # 然后清除 EXIF 元数据（原有逻辑）
                    success = reset(self.directory_path, log_callback=self.thread_safe_log, i18n=self.i18n)
                    # 在主线程中处理完成后的UI更新
                    self.root.after(0, lambda: self._on_reset_complete(success))
                except Exception as e:
                    self.root.after(0, lambda: self._on_reset_error(str(e)))

            reset_thread = threading.Thread(target=run_reset, daemon=True)
            reset_thread.start()

    def _on_reset_complete(self, success):
        """重置完成回调（在主线程中执行）"""
        if success:
            self.log("\n" + self.i18n.t("logs.separator"))
            self.log(self.i18n.t("logs.reset_complete"))
            self.log(self.i18n.t("logs.separator"))
            messagebox.showinfo(self.i18n.t("messages.reset_complete_title"), self.i18n.t("messages.reset_complete"))
        else:
            self.log("\n" + self.i18n.t("logs.separator"))
            self.log(self.i18n.t("logs.reset_failed"))
            self.log(self.i18n.t("logs.separator"))
            messagebox.showerror(self.i18n.t("messages.reset_failed_title"), self.i18n.t("messages.reset_failed"))

        # 恢复按钮状态
        self.reset_btn.config(state='normal')
        self.start_btn.config(state='normal')

        # 检查是否有report.csv（重置后应该没有）
        self._check_report_csv()

    def _on_reset_error(self, error_msg):
        """重置错误回调（在主线程中执行）"""
        self.log("\n" + self.i18n.t("logs.separator"))
        self.log(self.i18n.t('errors.reset_error', error=error_msg))
        self.log(self.i18n.t("logs.separator"))
        messagebox.showerror(self.i18n.t("errors.error_title"),
                           self.i18n.t("errors.reset_failed_msg", error=error_msg))

        # 恢复按钮状态
        self.reset_btn.config(state='normal')
        self.start_btn.config(state='normal')

    def start_processing(self):
        """开始处理"""
        if not self.directory_path:
            messagebox.showwarning(self.i18n.t("messages.hint"), self.i18n.t("messages.select_dir_first"))
            return

        if self.worker and self.worker.is_alive():
            messagebox.showwarning(self.i18n.t("messages.hint"), self.i18n.t("messages.processing"))
            return

        # V3.3: 处理前确认弹窗，告知用户文件将被移动
        confirm_message = """处理完成后，照片将按评分移动到对应文件夹：

• 3星优选 → 3星_优选/
• 2星良好 → 2星_良好/
• 1星普通 → 1星_普通/
• 0星/无鸟 → 0星_放弃/

处理完成后原目录将清空。
如需恢复原始目录结构，可使用"重置目录"功能。"""
        
        if not messagebox.askyesno("文件整理提示", confirm_message):
            return

        # 清空日志和进度
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        self.progress_bar['value'] = 0

        if self.i18n:
            self.log(self.i18n.t("logs.processing_start") + "\n")
        else:
            self.log("开始处理照片...\n")

        # 获取归一化模式
        selected_text = self.norm_var.get()
        mode_key = selected_text.split(" - ")[0].strip()

        norm_mapping = {
            "对数压缩(V3.1)": "log_compression",
            "原始方差": None,
            "log归一化": "log",
            "gentle归一化": "gentle",
            "sqrt归一化": "sqrt",
            "linear归一化": "linear"
        }
        selected_norm = norm_mapping.get(mode_key, "log_compression")

        # V3.1: ui_settings = [ai_confidence, sharpness_threshold, nima_threshold, save_crop, normalization]
        ui_settings = [
            self.ai_var.get(),
            self.sharp_var.get(),
            self.nima_var.get(),
            False,  # V3.1: 不保存crop图片
            selected_norm
        ]

        # 启动Worker线程
        self.worker = WorkerThread(
            self.directory_path,
            ui_settings,
            self.update_progress,
            self.on_finished,
            self.thread_safe_log,
            self.i18n
        )

        self.start_btn.config(state='disabled')
        self.reset_btn.config(state='disabled')
        self.worker.start()

    def update_progress(self, value):
        """更新进度条"""
        self.root.after(0, lambda: self.progress_bar.configure(value=value))

    def thread_safe_log(self, message, tag=None):
        """线程安全的日志输出"""
        self.root.after(0, lambda: self.log(message, tag))

    def log(self, message, tag=None):
        """输出日志"""
        self.log_text.config(state='normal')
        if tag:
            self.log_text.insert(tk.END, message + "\n", tag)
        else:
            self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        # 强制更新UI，确保日志实时显示
        self.log_text.update_idletasks()

    def on_finished(self, stats):
        """处理完成回调"""
        self.start_btn.config(state='normal')
        self.reset_btn.config(state='normal')
        self.post_da_btn.config(state='normal')  # 启用二次选鸟
        self.progress_bar['value'] = 100

        # V3.1: 清空日志窗口，然后显示最终报告（方便查看，无需滚动）
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')

        # 显示统计报告
        report = self._format_statistics_report(stats)
        self.log(report)

        # 显示Lightroom使用指南
        self.show_lightroom_guide()

        # 播放完成音效
        self._play_completion_sound()

    def _format_statistics_report(self, stats):
        """格式化统计报告"""
        total = stats['total']
        star_3 = stats['star_3']
        star_2 = stats['star_2']
        star_1 = stats['star_1']
        star_0 = stats.get('star_0', 0)
        no_bird = stats['no_bird']
        total_time = stats['total_time']
        avg_time = stats['avg_time']

        # 有鸟照片
        bird_total = star_3 + star_2 + star_1 + star_0

        report = "\n"
        report += "=" * 50 + "\n"
        report += f"📊 {self.i18n.t('report.title')}\n"
        report += "=" * 50 + "\n"
        report += self.i18n.t('report.total_photos', total=total) + "\n"
        report += self.i18n.t('report.total_time', time_sec=total_time, time_min=total_time/60) + "\n"
        report += self.i18n.t('report.avg_time', avg=avg_time) + "\n\n"

        picked = stats.get('picked', 0)

        percent_3 = star_3/total*100 if total > 0 else 0
        report += f"⭐⭐⭐ {self.i18n.t('report.star_3', count=star_3, percent=percent_3)}\n"
        if picked > 0:
            percent_picked = picked/star_3*100 if star_3 > 0 else 0
            report += f"  └─ {self.i18n.t('report.picked_detail', count=picked, percent=percent_picked)}\n"

        percent_2 = star_2/total*100 if total > 0 else 0
        report += f"⭐⭐ {self.i18n.t('report.star_2', count=star_2, percent=percent_2)}\n"

        percent_1 = star_1/total*100 if total > 0 else 0
        report += f"⭐ {self.i18n.t('report.star_1', count=star_1, percent=percent_1)}\n"

        if star_0 > 0:
            percent_0 = star_0/total*100 if total > 0 else 0
            report += self.i18n.t('report.star_0', count=star_0, percent=percent_0) + "\n"

        percent_no_bird = no_bird/total*100 if total > 0 else 0
        report += f"❌ {self.i18n.t('report.no_bird', count=no_bird, percent=percent_no_bird)}\n\n"

        percent_bird = bird_total/total*100 if total > 0 else 0
        report += self.i18n.t('report.bird_total', count=bird_total, percent=percent_bird) + "\n\n"

        report += "=" * 50 + "\n"
        report += f"💡 {self.i18n.t('report.tips_title')}\n"

        # 智能提示
        if no_bird / total > 0.8 if total > 0 else False:
            report += f"   {self.i18n.t('report.tip_high_no_bird')}\n"
        if star_3 == 0:
            report += f"   {self.i18n.t('report.tip_no_excellent')}\n"
        if star_3 / bird_total > 0.5 if bird_total > 0 else False:
            report += f"   {self.i18n.t('report.tip_high_excellent')}\n"
        if avg_time > 2000:
            report += f"   {self.i18n.t('report.tip_slow_processing', speed=avg_time/1000)}\n"

        report += "=" * 50 + "\n"

        return report

    def show_lightroom_guide(self):
        """显示Lightroom使用指南"""
        separator = "━" * 60
        guide = f"""
{separator}
  📸 {self.i18n.t("lightroom_guide.title")}
{separator}

【{self.i18n.t("lightroom_guide.method1_title")}】
  1️⃣ {self.i18n.t("lightroom_guide.method1_step1")}
  2️⃣ {self.i18n.t("lightroom_guide.method1_step2")}
  3️⃣ {self.i18n.t("lightroom_guide.method1_step3")}

【{self.i18n.t("lightroom_guide.method2_title")}】{self.i18n.t("lightroom_guide.method2_recommended")}
  {self.i18n.t("lightroom_guide.method2_intro")}

  1️⃣ {self.i18n.t("lightroom_guide.method2_step1")}
  2️⃣ {self.i18n.t("lightroom_guide.method2_step2")}
  3️⃣ {self.i18n.t("lightroom_guide.method2_step3")}

【{self.i18n.t("lightroom_guide.filter_title")}】
  {self.i18n.t("lightroom_guide.filter_method1")}
    • {self.i18n.t("lightroom_guide.filter_method1_step1")}
    • {self.i18n.t("lightroom_guide.filter_method1_step2")}
    • {self.i18n.t("lightroom_guide.filter_method1_step3")}

  {self.i18n.t("lightroom_guide.filter_method2")}
    • {self.i18n.t("lightroom_guide.filter_method2_step1")}
    • {self.i18n.t("lightroom_guide.filter_method2_step2")}

【{self.i18n.t("lightroom_guide.sort_title")}】
  1️⃣ {self.i18n.t("lightroom_guide.sort_step1")}
  2️⃣ {self.i18n.t("lightroom_guide.sort_step2")}
  3️⃣ {self.i18n.t("lightroom_guide.sort_step3")}
     {self.i18n.t("lightroom_guide.sort_step3_city")}
     {self.i18n.t("lightroom_guide.sort_step3_state")}
     {self.i18n.t("lightroom_guide.sort_step3_country")}
  4️⃣ {self.i18n.t("lightroom_guide.sort_step4")}

【{self.i18n.t("lightroom_guide.fields_title")}】
  • {self.i18n.t("lightroom_guide.field_rating")}
  • {self.i18n.t("lightroom_guide.field_pick")}
  • {self.i18n.t("lightroom_guide.field_city")}
  • {self.i18n.t("lightroom_guide.field_state")}
  • {self.i18n.t("lightroom_guide.field_country")}

【{self.i18n.t("lightroom_guide.workflow_title")}】
  ✅ {self.i18n.t("lightroom_guide.workflow_step1")}
  ✅ {self.i18n.t("lightroom_guide.workflow_step2")}
  ✅ {self.i18n.t("lightroom_guide.workflow_step3")}

💡 {self.i18n.t("lightroom_guide.csv_note")}
{separator}
"""
        self.log(guide)

    def show_initial_help(self):
        """显示初始帮助信息"""
        separator = "━" * 60
        help_text = f"""{separator}
  {self.i18n.t("help.welcome_title")}
{separator}
{self.i18n.t("help.usage_steps_title")}
  1️⃣ {self.i18n.t("help.step1")}
  2️⃣ {self.i18n.t("help.step2")}
  3️⃣ {self.i18n.t("help.step3")}
  4️⃣ {self.i18n.t("help.step4")}

📊 {self.i18n.t("help.rating_rules_title")}
  • {self.i18n.t("help.rule_3_star")}
    └─ {self.i18n.t("help.rule_picked")}
  • {self.i18n.t("help.rule_2_star")}
  • {self.i18n.t("help.rule_1_star")}
  • {self.i18n.t("help.rule_0_star")}
  • {self.i18n.t("help.rule_rejected")}

{self.i18n.t("help.folder_info")}

{self.i18n.t("help.ready")}
{separator}
  {self.i18n.t("help.tools_list_title")}
  1.  {self.i18n.t("help.tool_1")}
  2.  {self.i18n.t("help.tool_2")}
  3.  {self.i18n.t("help.tool_3")}
  4.  {self.i18n.t("help.tool_4")}
  5.  {self.i18n.t("help.tool_5")}
  6.  {self.i18n.t("help.tool_6")}
"""
        self.log(help_text)

    def _play_completion_sound(self):
        """播放选鸟完成音效"""
        sound_path = os.path.join(os.path.dirname(__file__), "img",
                                  "toy-story-short-happy-audio-logo-short-cartoony-intro-outro-music-125627.mp3")

        if not os.path.exists(sound_path):
            # 如果音效文件不存在，静默失败
            return

        try:
            # 使用afplay（macOS内置音频播放器）在后台播放
            subprocess.Popen(['afplay', sound_path],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except Exception:
            # 如果播放失败，静默失败（不影响主要功能）
            pass

    def on_closing(self):
        """窗口关闭事件"""
        if self.worker and self.worker.is_alive():
            if messagebox.askokcancel(self.i18n.t("messages.exit_title"), self.i18n.t("messages.exit_confirm")):
                self.worker._stop_event.set()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    if THEME_AVAILABLE:
        root = ThemedTk(theme="arc")
    else:
        root = tk.Tk()

    app = SuperPickyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
