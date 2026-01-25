#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky V3.2 - 高级配置管理
用于管理所有可配置的硬编码参数
"""

import json
import os
from pathlib import Path
import sys


class AdvancedConfig:
    """高级配置类 - 管理所有硬编码参数"""

    # 默认配置
    DEFAULT_CONFIG = {
        # 评分阈值（影响0星判定）
        "min_confidence": 0.5,      # AI置信度最低阈值 (0.3-0.7) - 低于此值判定为0星
        "min_sharpness": 100,       # 锐度最低阈值 - 低于此值判定为0星（头部区域锐度）
        "min_nima": 3.5,            # NIMA美学最低阈值 (3.0-5.0) - 低于此值判定为0星
        # V3.2: 移除 max_brisque（不再使用 BRISQUE 评估）

        # 精选设置
        "picked_top_percentage": 25, # 精选旗标Top百分比 (10-50) - 3星照片中美学+锐度双排名在此百分比内的设为精选
        
        # 曝光检测设置 V3.8
        "exposure_threshold": 0.10,  # 曝光阈值 (0.05-0.20) - 过曝/欠曝像素占比超过此值将降级一星
        
        # 连拍检测设置 V3.9
        "burst_time_threshold": 250,  # 连拍时间阈值(ms) (150-500) - 相邻照片时间差小于此值视为连拍
        "burst_min_count": 4,         # 连拍最少张数 (3-10) - 至少此数量连续照片才算连拍组

        # GPU Worker 设置
        "gpu_worker_count_adjust": 0,   # GPU Worker 数量调整 (-10 到 +10) - 在自动计算的基础上增加或减少的数量
        "max_gpu_worker_count": 1,      # GPU Worker 最大数量限制(1-8) - 避免显存占用过高导致崩溃
        "gpu_single_thread_mode": True, # GPU 评分单线程队列模式 - 避免并发导致显存崩溃

        # GPU Batch 设置
        "gpu_batch_size": 2,             # GPU 批量大小 (0=自动)
        "gpu_batch_min_size": 1,         # GPU 批量最小值
        "gpu_batch_max_size": 8,         # GPU 批量最大值
        "gpu_batch_mem_per_item_gb": 1.0,   # 单张批量显存基准(GB)，用于自动计算
        "gpu_batch_mem_overhead_gb": 2.0,   # 固定显存开销(GB)，用于自动计算
        "gpu_batch_max_wait_ms": 0,      # GPU 批量凑批等待时间(ms)，0表示不等待

        # 模型设备控制
        "force_cpu_for_iqa": True,            # IQA/TOPIQ 强制使用 CPU（避免与 YOLO 争抢显存）
        "force_cpu_for_flight_detector": True,  # FlightDetector 强制使用 CPU

        # CPU Worker 设置
        "cpu_worker_count_adjust": 0,    # CPU Worker 数量调整 (-10 到 +10) - 在自动计算的基础上增加或减少的数量
        "max_cpu_worker_count": 8,       # CPU Worker 最大数量限制 (1-128) - 自动计算时不会超过此值
        "cpu_rate_worker_count": 0,      # CPU 评分线程数 (0=自动) - GPU可用时作为辅助评分
        "cpu_io_worker_count": 2,        # CPU IO线程数 - HEIF转换/EXIF写入使用
        "cpu_rate_backlog_threshold": 1, # 评分队列积压阈值 - 超过后允许CPU评分辅助
        "cpu_rate_assist_enabled": True, # 是否允许CPU参与评分（GPU可用时）

        # 输出设置
        "save_csv": True,           # 是否保存CSV报告
        "log_level": "detailed",    # 日志详细程度: "simple" | "detailed"
        "debug_log": True,          # 是否启用调试日志 (True/False) - 用于排查问题

        # 语言设置（后续实现）
        "language": "zh_CN",        # zh_CN | en_US
    }

    def __init__(self, config_file=None):
        """初始化配置"""
        # 如果没有指定配置文件路径，使用用户目录
        if config_file is None:
            # 获取用户主目录下的配置目录
            if sys.platform == "darwin":  # macOS
                config_dir = Path.home() / "Library" / "Application Support" / "SuperPicky"
            elif sys.platform == "win32":  # Windows
                config_dir = Path.home() / "AppData" / "Local" / "SuperPicky"
            else:  # Linux
                config_dir = Path.home() / ".config" / "SuperPicky"

            # 创建配置目录（如果不存在）
            config_dir.mkdir(parents=True, exist_ok=True)

            # 配置文件路径
            self.config_file = str(config_dir / "advanced_config.json")
        else:
            self.config_file = config_file

        self.config = self.DEFAULT_CONFIG.copy()
        self.load()

    def load(self):
        """从文件加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # 合并配置（保留默认值中有但加载配置中没有的项）
                    self.config.update(loaded_config)
                print(f"✅ 已加载高级配置: {self.config_file}")
            except Exception as e:
                print(f"⚠️  加载配置失败，使用默认值: {e}")

    def save(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"✅ 已保存高级配置: {self.config_file}")
            return True
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")
            return False

    def reset_to_default(self):
        """重置为默认配置"""
        self.config = self.DEFAULT_CONFIG.copy()

    # Getter方法
    @property
    def min_confidence(self):
        return self.config["min_confidence"]

    @property
    def min_sharpness(self):
        return self.config["min_sharpness"]

    @property
    def min_nima(self):
        return self.config["min_nima"]

    # V3.2: 移除 max_brisque 属性

    @property
    def picked_top_percentage(self):
        return self.config["picked_top_percentage"]
    
    @property
    def exposure_threshold(self):
        return self.config.get("exposure_threshold", 0.10)
    
    @property
    def burst_time_threshold(self):
        return self.config.get("burst_time_threshold", 250)
    
    @property
    def burst_min_count(self):
        return self.config.get("burst_min_count", 4)

    @property
    def gpu_worker_count_adjust(self):
        return self.config.get("gpu_worker_count_adjust", 0)

    @property
    def max_gpu_worker_count(self):
        return self.config.get("max_gpu_worker_count", 1)

    @property
    def gpu_single_thread_mode(self):
        return self.config.get("gpu_single_thread_mode", True)

    @property
    def gpu_batch_size(self):
        return self.config.get("gpu_batch_size", 0)

    @property
    def gpu_batch_min_size(self):
        return self.config.get("gpu_batch_min_size", 1)

    @property
    def gpu_batch_max_size(self):
        return self.config.get("gpu_batch_max_size", 8)

    @property
    def gpu_batch_mem_per_item_gb(self):
        return self.config.get("gpu_batch_mem_per_item_gb", 1.0)

    @property
    def gpu_batch_mem_overhead_gb(self):
        return self.config.get("gpu_batch_mem_overhead_gb", 2.0)

    @property
    def gpu_batch_max_wait_ms(self):
        return self.config.get("gpu_batch_max_wait_ms", 0)

    @property
    def force_cpu_for_iqa(self):
        return self.config.get("force_cpu_for_iqa", False)

    @property
    def force_cpu_for_flight_detector(self):
        return self.config.get("force_cpu_for_flight_detector", False)

    @property
    def cpu_worker_count_adjust(self):
        return self.config.get("cpu_worker_count_adjust", 0)

    @property
    def max_cpu_worker_count(self):
        return self.config.get("max_cpu_worker_count", 8)

    @property
    def cpu_rate_worker_count(self):
        return self.config.get("cpu_rate_worker_count", 0)

    @property
    def cpu_io_worker_count(self):
        return self.config.get("cpu_io_worker_count", 2)

    @property
    def cpu_rate_backlog_threshold(self):
        return self.config.get("cpu_rate_backlog_threshold", 8)

    @property
    def cpu_rate_assist_enabled(self):
        return self.config.get("cpu_rate_assist_enabled", True)

    @property
    def save_csv(self):
        return self.config["save_csv"]

    @property
    def log_level(self):
        return self.config["log_level"]

    @property
    def debug_log(self):
        return self.config.get("debug_log", True)

    @property
    def language(self):
        return self.config["language"]

    # Setter方法
    def set_min_confidence(self, value):
        """设置AI置信度阈值 (0.3-0.7)"""
        self.config["min_confidence"] = max(0.3, min(0.7, float(value)))

    def set_min_sharpness(self, value):
        """设置锐度最低阈值 (100-500) - 头部区域锐度"""
        self.config["min_sharpness"] = max(100, min(500, int(value)))

    def set_min_nima(self, value):
        """设置美学最低阈值 (3.0-5.0)"""
        self.config["min_nima"] = max(3.0, min(5.0, float(value)))

    # V3.2: 移除 set_max_brisque 方法

    def set_picked_top_percentage(self, value):
        """设置精选旗标Top百分比 (10-50)"""
        self.config["picked_top_percentage"] = max(10, min(50, int(value)))
    
    def set_exposure_threshold(self, value):
        """设置曝光阈值 (0.05-0.20)"""
        self.config["exposure_threshold"] = max(0.05, min(0.20, float(value)))
    
    def set_burst_time_threshold(self, value):
        """设置连拍时间阈值 (150-500ms)"""
        self.config["burst_time_threshold"] = max(150, min(500, int(value)))
    
    def set_burst_min_count(self, value):
        """设置连拍最少张数 (3-10)"""
        self.config["burst_min_count"] = max(3, min(10, int(value)))

    def set_gpu_worker_count_adjust(self, value):
        """设置GPU Worker数量调整 (-10 到 +10)"""
        self.config["gpu_worker_count_adjust"] = max(-10, min(10, int(value)))

    def set_max_gpu_worker_count(self, value):
        """设置GPU Worker最大数量限制 (1-8)"""
        self.config["max_gpu_worker_count"] = max(1, min(8, int(value)))

    def set_gpu_single_thread_mode(self, value):
        """设置GPU评分单线程模式"""
        self.config["gpu_single_thread_mode"] = bool(value)

    def set_gpu_batch_size(self, value):
        """设置GPU批量大小(0=自动)"""
        self.config["gpu_batch_size"] = max(0, int(value))

    def set_gpu_batch_min_size(self, value):
        """设置GPU批量最小值"""
        self.config["gpu_batch_min_size"] = max(1, int(value))

    def set_gpu_batch_max_size(self, value):
        """设置GPU批量最大值"""
        self.config["gpu_batch_max_size"] = max(1, int(value))

    def set_gpu_batch_mem_per_item_gb(self, value):
        """设置单张批量显存基准(GB)"""
        self.config["gpu_batch_mem_per_item_gb"] = max(0.1, float(value))

    def set_gpu_batch_mem_overhead_gb(self, value):
        """设置固定显存开销(GB)"""
        self.config["gpu_batch_mem_overhead_gb"] = max(0.0, float(value))

    def set_gpu_batch_max_wait_ms(self, value):
        """设置GPU批量凑批等待时间(ms)"""
        self.config["gpu_batch_max_wait_ms"] = max(0, int(value))

    def set_force_cpu_for_iqa(self, value):
        """设置是否强制IQA/TOPIQ使用CPU"""
        self.config["force_cpu_for_iqa"] = bool(value)

    def set_force_cpu_for_flight_detector(self, value):
        """设置是否强制飞鸟检测使用CPU"""
        self.config["force_cpu_for_flight_detector"] = bool(value)

    def set_cpu_worker_count_adjust(self, value):
        """设置CPU Worker数量调整 (-10 到 +10)"""
        self.config["cpu_worker_count_adjust"] = max(-10, min(10, int(value)))

    def set_max_cpu_worker_count(self, value):
        """设置CPU Worker最大数量限制 (1-128)"""
        self.config["max_cpu_worker_count"] = max(1, min(128, int(value)))

    def set_cpu_rate_worker_count(self, value):
        """设置CPU评分线程数 (0=自动)"""
        self.config["cpu_rate_worker_count"] = max(0, min(128, int(value)))

    def set_cpu_io_worker_count(self, value):
        """设置CPU IO线程数"""
        self.config["cpu_io_worker_count"] = max(1, min(128, int(value)))

    def set_cpu_rate_backlog_threshold(self, value):
        """设置CPU评分辅助触发的队列积压阈值"""
        self.config["cpu_rate_backlog_threshold"] = max(0, int(value))

    def set_cpu_rate_assist_enabled(self, value):
        """设置是否允许CPU参与评分"""
        self.config["cpu_rate_assist_enabled"] = bool(value)

    def set_save_csv(self, value):
        """设置是否保存CSV"""
        self.config["save_csv"] = bool(value)

    def set_log_level(self, value):
        """设置日志详细程度"""
        if value in ["simple", "detailed"]:
            self.config["log_level"] = value

    def set_debug_log(self, value):
        """设置是否启用调试日志"""
        self.config["debug_log"] = bool(value)

    def set_language(self, value):
        """设置语言"""
        if value in ["zh_CN", "en_US"]:
            self.config["language"] = value

    def get_dict(self):
        """获取配置字典（用于传递给其他模块）"""
        return self.config.copy()


# 全局配置实例
_config_instance = None


def get_advanced_config():
    """获取全局配置实例（单例模式）"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AdvancedConfig()
    return _config_instance
