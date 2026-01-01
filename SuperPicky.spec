import os
import site
from PyInstaller.utils.hooks import collect_data_files

# 获取当前工作目录
base_path = os.path.abspath('.')

# 动态获取 site-packages 路径
site_packages = site.getsitepackages()[0]
user_site = site.getusersitepackages()

# V3.7: 切换到 TOPIQ 美学评分模型
# 处理 ultralytics
if os.path.exists(os.path.join(user_site, 'ultralytics')):
    ultralytics_base = user_site
elif os.path.exists(os.path.join(site_packages, 'ultralytics')):
    ultralytics_base = site_packages
else:
    ultralytics_base = '/Users/jameszhenyu/Library/Python/3.9/lib/python/site-packages'

# 动态收集数据文件
ultralytics_datas = collect_data_files('ultralytics')

# 组合所有数据文件 (V3.7: 添加 TOPIQ 权重)
all_datas = [
    # AI模型文件
    (os.path.join(base_path, 'models/yolo11m-seg.pt'), 'models'),
    # V3.5: 鸟类关键点检测模型
    (os.path.join(base_path, 'models/cub200_keypoint_resnet50.pth'), 'models'),
    # V3.5: 飞行姿态检测模型
    (os.path.join(base_path, 'models/superFlier_efficientnet.pth'), 'models'),
    # V3.7: TOPIQ 美学评分模型 (替代 NIMA)
    (os.path.join(base_path, 'models/cfanet_iaa_ava_res50-3cd62bb3.pth'), 'models'),

    # ExifTool 完整打包
    (os.path.join(base_path, 'exiftool_bundle'), 'exiftool_bundle'),

    # 图片资源
    (os.path.join(base_path, 'img'), 'img'),

    # 国际化语言包
    (os.path.join(base_path, 'locales'), 'locales'),
    
    # Ultralytics 配置（手动添加完整 cfg 目录）
    (os.path.join(ultralytics_base, 'ultralytics/cfg'), 'ultralytics/cfg'),
]

# 添加动态收集的数据
all_datas.extend(ultralytics_datas)

a = Analysis(
    ['main.py'],
    pathex=[base_path],
    binaries=[],
    datas=all_datas,
    hiddenimports=[
        'ultralytics',
        'torch',
        'torchvision',
        'PIL',
        'cv2',
        'numpy',
        'yaml',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_agg',
        # V3.6: PySide6 GUI
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        # V3.7: TOPIQ 依赖 timm (ResNet50)
        'timm',
        'timm.models',
        'timm.models.resnet',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_cv2.py'],  # V3.6: cv2 预加载钩子防止递归错误
    excludes=['PyQt5', 'PyQt6', 'tkinter'],
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
    codesign_identity=None,  # V3.6: 禁用签名以便测试
    entitlements_file=None,  # V3.6: 禁用权限以便测试
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
        'CFBundleVersion': '3.7.0',
        'CFBundleShortVersionString': '3.7.0',
        'NSHumanReadableCopyright': 'Copyright © 2025 James Zhen Yu. All rights reserved.',
        'LSMinimumSystemVersion': '10.15',
        'NSRequiresAquaSystemAppearance': False,
    },
)
