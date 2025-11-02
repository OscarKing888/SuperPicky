# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

# 获取当前工作目录
base_path = os.path.abspath('.')

# Python虚拟环境路径
venv_path = '/Users/jameszhenyu/PycharmProjects/SuperPicky_SandBox/.venv/lib/python3.12/site-packages'

a = Analysis(
    ['main.py'],
    pathex=[base_path],
    binaries=[],
    datas=[
        # AI模型文件（只使用yolo11m-seg.pt）
        (os.path.join(base_path, 'models/yolo11m-seg.pt'), 'models'),

        # ExifTool 完整打包（整个 exiftool_bundle 目录，包含可执行文件 + Perl库）
        (os.path.join(base_path, 'exiftool_bundle'), 'exiftool_bundle'),

        # 图片资源
        (os.path.join(base_path, 'img'), 'img'),

        # 国际化语言包（V3.2新增）
        (os.path.join(base_path, 'locales'), 'locales'),

        # PyIQA 预训练模型（V3.2新增 - 避免首次运行下载208MB）
        (os.path.join(base_path, 'temp_models/pyiqa'), 'pyiqa_models'),

        # Ultralytics配置文件
        (os.path.join(venv_path, 'ultralytics/cfg/default.yaml'), 'ultralytics/cfg'),
        (os.path.join(venv_path, 'ultralytics/utils'), 'ultralytics/utils'),
        (os.path.join(venv_path, 'ultralytics/nn'), 'ultralytics/nn'),

        # PyIQA 完整目录结构（修复 FileNotFoundError）
        (os.path.join(venv_path, 'pyiqa/models'), 'pyiqa/models'),
        (os.path.join(venv_path, 'pyiqa/archs'), 'pyiqa/archs'),
        (os.path.join(venv_path, 'pyiqa/data'), 'pyiqa/data'),
        (os.path.join(venv_path, 'pyiqa/utils'), 'pyiqa/utils'),
        (os.path.join(venv_path, 'pyiqa/metrics'), 'pyiqa/metrics'),
        (os.path.join(venv_path, 'pyiqa/losses'), 'pyiqa/losses'),
        (os.path.join(venv_path, 'pyiqa/matlab_utils'), 'pyiqa/matlab_utils'),
    ],
    hiddenimports=[
        'ultralytics',
        'torch',
        'torchvision',
        'PIL',
        'PIL._tkinter_finder',
        'tkinter',
        'tkinter.ttk',
        'cv2',
        'numpy',
        'yaml',
        'ttkthemes',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_agg',
        # PyIQA 隐藏导入（修复 FileNotFoundError）
        'pyiqa',
        'pyiqa.models',
        'pyiqa.archs',
        'pyiqa.data',
        'pyiqa.utils',
        'pyiqa.metrics',
        'pyiqa.losses',
        'pyiqa.matlab_utils',
        # PyIQA 依赖库（V3.2.1新增）
        'scipy',
        'scipy.stats',
        'scipy.special',
        'scipy.optimize',
        'scipy.linalg',
        'scipy.io',
        'timm',
        'timm.models',
        'timm.models.layers',
        'einops',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SuperPicky',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity='Developer ID Application: James Zhen Yu (JWR6FDB52H)',
    entitlements_file='entitlements.plist',
    icon='img/SuperPicky-V0.02.icns',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SuperPicky',
)

app = BUNDLE(
    coll,
    name='SuperPicky.app',
    icon='img/SuperPicky-V0.02.icns',
    bundle_identifier='com.jamesphotography.superpicky',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
        'CFBundleName': 'SuperPicky',
        'CFBundleDisplayName': 'SuperPicky - 慧眼选鸟',
        'CFBundleVersion': '3.2.3',
        'CFBundleShortVersionString': '3.2.3',
        'NSHumanReadableCopyright': 'Copyright © 2025 James Zhen Yu. All rights reserved.',
        'LSMinimumSystemVersion': '10.15',
        'NSRequiresAquaSystemAppearance': False,
    },
)
