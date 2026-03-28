"""
SuperPicky 配置管理模块
统一管理常量、可覆盖运行时配置与懒加载资源注册
"""
import json
import os
import sys
import platform
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch


# =========================
# 基础路径工具
# =========================

def resource_path(relative_path: str) -> str:
    """获取资源文件路径，支持 PyInstaller 打包"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)


def get_app_config_dir(app_name: str = 'SuperPicky') -> Path:
    """跨平台应用配置目录"""
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / app_name
    if sys.platform == 'win32':
        return Path.home() / 'AppData' / 'Local' / app_name
    return Path.home() / '.config' / app_name


def get_app_data_dir(app_name: str = 'SuperPicky') -> Path:
    """跨平台用户数据目录（文档目录）"""
    return Path.home() / 'Documents' / f'{app_name}_Data'


def get_patch_dir(app_name: str = 'SuperPicky') -> Path:
    """在线补丁目录"""
    return get_app_config_dir(app_name) / 'code_updates'


def get_birdid_settings_path(app_name: str = 'SuperPicky') -> Path:
    """BirdID Dock 设置文件路径"""
    return get_app_data_dir(app_name) / 'birdid_dock_settings.json'


# =========================
# 可覆盖配置（ENV + 配置文件）
# =========================

_override_cache: Optional[Dict[str, Any]] = None
_override_lock = threading.RLock()


def _load_override_file() -> Dict[str, Any]:
    global _override_cache
    with _override_lock:
        if _override_cache is not None:
            return _override_cache

        cfg_path = get_app_config_dir() / 'advanced_config.json'
        if not cfg_path.exists():
            _override_cache = {}
            return _override_cache

        try:
            _override_cache = json.loads(cfg_path.read_text(encoding='utf-8'))
        except Exception:
            _override_cache = {}
        return _override_cache


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    norm = str(value).strip().lower()
    if norm in {'1', 'true', 'yes', 'on'}:
        return True
    if norm in {'0', 'false', 'no', 'off'}:
        return False
    return default


def _env_or_override(name: str, override_key: Optional[str], default: Any) -> Any:
    env_value = os.getenv(name)
    if env_value is not None and str(env_value).strip() != '':
        return env_value

    if override_key:
        loaded = _load_override_file()
        if override_key in loaded:
            return loaded.get(override_key)

    return default


# =========================
# 静态常量分层
# =========================

@dataclass
class FileConfig:
    """文件处理相关配置"""
    RAW_EXTENSIONS: List[str] = None
    JPG_EXTENSIONS: List[str] = None

    def __post_init__(self):
        if self.RAW_EXTENSIONS is None:
            self.RAW_EXTENSIONS = [
                '.nef', '.cr2', '.cr3', '.arw', '.raf',
                '.orf', '.rw2', '.pef', '.dng', '.3fr', '.iiq'
            ]
        if self.JPG_EXTENSIONS is None:
            self.JPG_EXTENSIONS = ['.jpg', '.jpeg']


@dataclass
class DirectoryConfig:
    """目录名称配置"""
    EXCELLENT_DIR: str = '优秀'
    STANDARD_DIR: str = '标准'
    NO_BIRDS_DIR: str = '没鸟'
    TEMP_DIR: str = '_temp'
    REDBOX_DIR: str = 'Redbox'
    CROP_TEMP_DIR: str = '.crop_temp'

    OLD_ALGORITHM_EXCELLENT: str = '老算法优秀'
    NEW_ALGORITHM_EXCELLENT: str = '新算法优秀'
    BOTH_ALGORITHMS_EXCELLENT: str = '双算法优秀'
    ALGORITHM_DIFF_DIR: str = '算法差异'

    LOG_FILE: str = '.process_log.txt'
    REPORT_FILE: str = '.report.db'
    COMPARISON_REPORT_FILE: str = '.algorithm_comparison.csv'


@dataclass
class AIConfig:
    """AI 模型相关配置"""
    MODEL_FILE: str = 'models/yolo11l-seg.pt'
    BIRD_CLASS_ID: int = 14
    TARGET_IMAGE_SIZE: int = 1024
    CENTER_THRESHOLD: float = 0.15
    SHARPNESS_NORMALIZATION: str = None

    def get_model_path(self) -> str:
        return resource_path(self.MODEL_FILE)


@dataclass
class UIConfig:
    """UI 相关静态配置"""
    CONFIDENCE_SCALE: float = 100.0
    AREA_SCALE: float = 1000.0
    SHARPNESS_SCALE: int = 20
    PROGRESS_MIN: int = 0
    PROGRESS_MAX: int = 100
    BEEP_COUNT: int = 3


@dataclass
class CSVConfig:
    """CSV 报告相关配置"""
    HEADERS: List[str] = None

    def __post_init__(self):
        if self.HEADERS is None:
            self.HEADERS = [
                'filename', 'found_bird', 'AI score', 'bird_centre_x',
                'bird_centre_y', 'bird_area', 's_bird_area',
                'laplacian_var', 'sobel_var', 'fft_high_freq', 'contrast',
                'edge_density', 'background_complexity', 'motion_blur',
                'normalized_new', 'composite_score', 'result_new',
                'dominant_bool', 'centred_bool', 'sharp_bool', 'class_id'
            ]


@dataclass
class ServerConfig:
    """BirdID 服务运行时配置（支持 ENV/配置文件覆盖）"""
    HOST: str = '127.0.0.1'
    PORT: int = 5156
    HEALTH_TIMEOUT_SECONDS: float = 2.0
    STARTUP_WAIT_SECONDS: float = 10.0
    POLL_INTERVAL_SECONDS: float = 0.5

    @classmethod
    def load(cls) -> 'ServerConfig':
        host = str(_env_or_override('SUPERPICKY_SERVER_HOST', None, cls.HOST))
        port = int(_env_or_override('SUPERPICKY_SERVER_PORT', None, cls.PORT))
        health_timeout = float(_env_or_override('SUPERPICKY_SERVER_HEALTH_TIMEOUT', None, cls.HEALTH_TIMEOUT_SECONDS))
        startup_wait = float(_env_or_override('SUPERPICKY_SERVER_STARTUP_WAIT', None, cls.STARTUP_WAIT_SECONDS))
        poll = float(_env_or_override('SUPERPICKY_SERVER_POLL_INTERVAL', None, cls.POLL_INTERVAL_SECONDS))
        return cls(
            HOST=host,
            PORT=port,
            HEALTH_TIMEOUT_SECONDS=health_timeout,
            STARTUP_WAIT_SECONDS=startup_wait,
            POLL_INTERVAL_SECONDS=poll,
        )


@dataclass
class EndpointConfig:
    """远程端点配置（支持 ENV 覆盖）"""
    MIRROR_BASE_URL: str = 'http://1.119.150.179:59080/superpicky'
    UPDATE_DOWNLOAD_PAGE: str = 'https://superpicky.jamesphotography.com.au/#download'
    EBIRD_API_BASE: str = 'https://api.ebird.org/v2'
    NOMINATIM_REVERSE_URL: str = 'https://nominatim.openstreetmap.org/reverse'

    @classmethod
    def load(cls) -> 'EndpointConfig':
        return cls(
            MIRROR_BASE_URL=str(_env_or_override('SUPERPICKY_MIRROR_BASE_URL', None, cls.MIRROR_BASE_URL)),
            UPDATE_DOWNLOAD_PAGE=str(_env_or_override('SUPERPICKY_DOWNLOAD_PAGE', None, cls.UPDATE_DOWNLOAD_PAGE)),
            EBIRD_API_BASE=str(_env_or_override('SUPERPICKY_EBIRD_API_BASE', None, cls.EBIRD_API_BASE)),
            NOMINATIM_REVERSE_URL=str(_env_or_override('SUPERPICKY_NOMINATIM_REVERSE_URL', None, cls.NOMINATIM_REVERSE_URL)),
        )


class Config:
    """主配置类"""

    def __init__(self):
        self.file = FileConfig()
        self.directory = DirectoryConfig()
        self.ai = AIConfig()
        self.ui = UIConfig()
        self.csv = CSVConfig()
        self.server = ServerConfig.load()
        self.endpoints = EndpointConfig.load()

    def get_directory_names(self) -> Dict[str, str]:
        return {
            'excellent': self.directory.EXCELLENT_DIR,
            'standard': self.directory.STANDARD_DIR,
            'no_birds': self.directory.NO_BIRDS_DIR,
            'temp': self.directory.TEMP_DIR,
            'redbox': self.directory.REDBOX_DIR,
            'crop_temp': self.directory.CROP_TEMP_DIR,
        }

    def is_raw_file(self, filename: str) -> bool:
        _, ext = os.path.splitext(filename)
        return ext.lower() in self.file.RAW_EXTENSIONS

    def is_jpg_file(self, filename: str) -> bool:
        _, ext = os.path.splitext(filename)
        return ext.lower() in self.file.JPG_EXTENSIONS


# =========================
# 懒加载资源注册器
# =========================

_MISSING = object()


class LazyRegistry:
    """线程安全懒加载注册器。"""

    def __init__(self):
        self._values: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def get_or_create(self, key: str, factory: Callable[[], Any]) -> Any:
        value = self._values.get(key, _MISSING)
        if value is not _MISSING:
            return value
        with self._lock:
            value = self._values.get(key, _MISSING)
            if value is _MISSING:
                value = factory()
                self._values[key] = value
            return value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._values[key] = value

    def clear(self, key: str) -> None:
        with self._lock:
            self._values.pop(key, None)

    def clear_all(self) -> None:
        with self._lock:
            self._values.clear()


_lazy_registry = LazyRegistry()


def get_lazy_registry() -> LazyRegistry:
    return _lazy_registry


# =========================
# 设备选择
# =========================

def get_best_device():
    """获取最佳计算设备"""
    try:
        system = platform.system()
        if system == 'Darwin':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')

        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    except Exception:
        return torch.device('cpu')


# 全局配置实例
config = Config()
