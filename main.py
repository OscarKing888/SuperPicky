#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky - PySide6 版本入口点
Version: 3.9.3 - Focus Status Fix
"""

import sys
import os
import traceback
import threading
import atexit
import signal
import faulthandler
from datetime import datetime

# V3.9.3: 修复 macOS PyInstaller 打包后的多进程问题
# 必须在所有其他导入之前设置
import multiprocessing
if sys.platform == 'darwin':
    multiprocessing.set_start_method('spawn', force=True)

# V3.9.4: 防止 PyInstaller 打包后 spawn 模式创建重复进程/窗口
# 这是 macOS PyInstaller 的标准做法
multiprocessing.freeze_support()

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from ui.main_window import SuperPickyMainWindow

# V3.9.3: 全局窗口引用，防止重复创建
_main_window = None

# 退出追踪
_exit_called = False
_exit_reason = None
_exit_stack = None
_faulthandler_file = None


def exit_hook():
    """程序退出时的钩子函数（用于调试）"""
    global _exit_called, _exit_reason, _exit_stack
    
    if not _exit_called:
        _exit_called = True
        _exit_reason = "atexit"
        
        # 获取当前堆栈
        import traceback
        _exit_stack = traceback.format_stack()
        
        # 写入退出日志
        try:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".superpicky")
            os.makedirs(log_dir, exist_ok=True)
            exit_log_file = os.path.join(log_dir, "exit_log.txt")
            
            with open(exit_log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"程序退出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"退出原因: {_exit_reason}\n")
                f.write(f"退出代码: {sys.exitfunc_code if hasattr(sys, 'exitfunc_code') else 'N/A'}\n")
                f.write("\n退出时的调用堆栈:\n")
                f.write("".join(_exit_stack))
                f.write("="*80 + "\n\n")
            
            print(f"\n[退出钩子] 程序退出，原因: {_exit_reason}", file=sys.stderr)
            print(f"[退出钩子] 退出日志已保存到: {exit_log_file}", file=sys.stderr)
            print("[退出钩子] 调用堆栈:", file=sys.stderr)
            print("".join(_exit_stack), file=sys.stderr)
        except Exception as e:
            print(f"[退出钩子] 写入退出日志失败: {e}", file=sys.stderr)
        
        # 这里可以设置断点进行调试
        # import pdb; pdb.set_trace()  # 取消注释以启用断点


def signal_handler(signum, frame):
    """信号处理器（捕获退出信号）"""
    global _exit_called, _exit_reason, _exit_stack
    
    signal_names = {
        signal.SIGINT: "SIGINT (Ctrl+C)",
        signal.SIGTERM: "SIGTERM (终止信号)",
        signal.SIGBREAK: "SIGBREAK (Windows Ctrl+Break)",
    }
    
    signal_name = signal_names.get(signum, f"Signal {signum}")
    
    if not _exit_called:
        _exit_called = True
        _exit_reason = signal_name
        
        import traceback
        _exit_stack = traceback.format_stack(frame)
        
        print(f"\n[信号处理] 收到信号: {signal_name}", file=sys.stderr)
        print("[信号处理] 调用堆栈:", file=sys.stderr)
        print("".join(_exit_stack), file=sys.stderr)
        
        # 写入退出日志
        try:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".superpicky")
            os.makedirs(log_dir, exist_ok=True)
            exit_log_file = os.path.join(log_dir, "exit_log.txt")
            
            with open(exit_log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"程序退出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"退出原因: {signal_name}\n")
                f.write("\n退出时的调用堆栈:\n")
                f.write("".join(_exit_stack))
                f.write("="*80 + "\n\n")
        except Exception:
            pass
    
    # 调用默认信号处理
    if signum == signal.SIGINT:
        sys.exit(130)  # Ctrl+C 的标准退出代码
    elif signum == signal.SIGTERM:
        sys.exit(143)  # SIGTERM 的标准退出代码
    else:
        sys.exit(1)


def write_crash_log(exc_type, exc_value, exc_traceback, thread_name=None):
    """写入崩溃日志到文件"""
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".superpicky")
        os.makedirs(log_dir, exist_ok=True)
        crash_log_file = os.path.join(log_dir, "crash_log.txt")
        
        with open(crash_log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"崩溃时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if thread_name:
                f.write(f"线程: {thread_name}\n")
            f.write(f"异常类型: {exc_type.__name__}\n")
            f.write(f"异常消息: {str(exc_value)}\n")
            f.write("\n完整堆栈跟踪:\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
            f.write("="*80 + "\n\n")
        
        # 同时输出到stderr
        print("\n" + "="*80, file=sys.stderr)
        print(f"崩溃时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
        if thread_name:
            print(f"线程: {thread_name}", file=sys.stderr)
        print(f"异常类型: {exc_type.__name__}", file=sys.stderr)
        print(f"异常消息: {str(exc_value)}", file=sys.stderr)
        print("\n完整堆栈跟踪:", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
        print("="*80, file=sys.stderr)
        print(f"\n崩溃日志已保存到: {crash_log_file}", file=sys.stderr)
    except Exception as e:
        print(f"无法写入崩溃日志: {e}", file=sys.stderr)


def global_exception_handler(exc_type, exc_value, exc_traceback):
    """全局异常处理器（捕获主线程未处理的异常）"""
    if exc_type is KeyboardInterrupt:
        # 允许正常退出
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    thread_name = threading.current_thread().name
    write_crash_log(exc_type, exc_value, exc_traceback, thread_name)
    
    # 调用默认异常处理器
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def thread_exception_handler(args):
    """线程异常处理器（捕获子线程未处理的异常）"""
    exc_type = args.exc_type
    exc_value = args.exc_value
    exc_traceback = args.exc_traceback
    thread_name = args.thread.name if hasattr(args, 'thread') else "Unknown"
    
    write_crash_log(exc_type, exc_value, exc_traceback, thread_name)
    
    # 输出到stderr
    print(f"\n线程 {thread_name} 发生未捕获的异常:", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)


def main():
    """主函数"""
    global _main_window, _faulthandler_file
    
    # 注册退出钩子函数
    atexit.register(exit_hook)

    # 启用faulthandler，捕获底层崩溃堆栈
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".superpicky")
        os.makedirs(log_dir, exist_ok=True)
        faulthandler_log = os.path.join(log_dir, "faulthandler_log.txt")
        _faulthandler_file = open(faulthandler_log, 'a', encoding='utf-8')
        faulthandler.enable(file=_faulthandler_file, all_threads=True)
    except Exception as e:
        print(f"⚠️  faulthandler 启用失败: {e}", file=sys.stderr)
    
    # 注册信号处理器（捕获退出信号）
    try:
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
        if sys.platform == 'win32':
            try:
                signal.signal(signal.SIGBREAK, signal_handler)  # Windows Ctrl+Break
            except AttributeError:
                pass  # SIGBREAK 在某些Python版本中可能不存在
    except Exception as e:
        print(f"⚠️  注册信号处理器失败: {e}", file=sys.stderr)
    
    # 设置全局异常处理器
    sys.excepthook = global_exception_handler
    threading.excepthook = thread_exception_handler
    
    # 记录程序启动
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".superpicky")
        os.makedirs(log_dir, exist_ok=True)
        exit_log_file = os.path.join(log_dir, "exit_log.txt")
        with open(exit_log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"程序启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    except Exception:
        pass
    
    # V3.9.3: 检查是否已有 QApplication 实例
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    else:
        print("⚠️  检测到已存在的 QApplication 实例")
    
    # 设置应用属性
    app.setApplicationName("SuperPicky")
    app.setApplicationDisplayName("慧眼选鸟")
    app.setOrganizationName("JamesPhotography")
    app.setOrganizationDomain("jamesphotography.com.au")
    
    # 设置应用图标
    icon_path = os.path.join(os.path.dirname(__file__), "img", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # 注：Qt6/PySide6 默认启用 HiDPI 支持，无需手动设置
    
    # V3.9.3: 防止重复创建窗口
    if _main_window is None:
        _main_window = SuperPickyMainWindow()
        _main_window.show()
    else:
        print("⚠️  检测到已存在的主窗口实例")
        _main_window.raise_()
        _main_window.activateWindow()
    
    # 运行事件循环
    try:
        exit_code = app.exec()
        # 记录正常退出
        try:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".superpicky")
            os.makedirs(log_dir, exist_ok=True)
            exit_log_file = os.path.join(log_dir, "exit_log.txt")
            with open(exit_log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"程序退出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"退出原因: QApplication.exec() 正常返回\n")
                f.write(f"退出代码: {exit_code}\n")
                f.write("="*80 + "\n\n")
        except Exception:
            pass
        sys.exit(exit_code)
    except SystemExit as e:
        # 记录 SystemExit
        global _exit_reason
        _exit_reason = f"SystemExit (code: {e.code})"
        try:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".superpicky")
            os.makedirs(log_dir, exist_ok=True)
            exit_log_file = os.path.join(log_dir, "exit_log.txt")
            with open(exit_log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"程序退出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"退出原因: SystemExit\n")
                f.write(f"退出代码: {e.code}\n")
                import traceback
                f.write("\n调用堆栈:\n")
                f.write("".join(traceback.format_stack()))
                f.write("="*80 + "\n\n")
        except Exception:
            pass
        raise
    except Exception as e:
        # 记录未捕获的异常
        try:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".superpicky")
            os.makedirs(log_dir, exist_ok=True)
            exit_log_file = os.path.join(log_dir, "exit_log.txt")
            with open(exit_log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"程序退出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"退出原因: 未捕获的异常\n")
                f.write(f"异常类型: {type(e).__name__}\n")
                f.write(f"异常消息: {str(e)}\n")
                import traceback
                f.write("\n完整堆栈跟踪:\n")
                traceback.print_exception(type(e), e, e.__traceback__, file=f)
                f.write("="*80 + "\n\n")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
