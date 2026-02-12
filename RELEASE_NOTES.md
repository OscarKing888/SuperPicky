# SuperPicky Release Notes

## V4.0.4 beta (2026-02-09) - 连拍优化与稳定性改进

### Bug Fixes
- [Fix] 启用识鸟但无结果时，照片放入"其他鸟类"子目录而非根目录
- [Fix] 版本号统一从 constants.py 获取，避免版本不一致

### Improvements
- [UI] 确认对话框中显示当前选择的国家/区域识别设置
- [Build] 新增 M3 Mac 专用打包脚本 (create_pkg_dmg_v4.0.4_m3.sh)

---

## V4.0.5 (2026-02-12) - 中文路径兼容与日志优化

### Bug Fixes
- [Fix] 修复识鸟模型在包含中文路径（如 `/Volumes/我的硬盘/`）下无法加载的问题
- [Fix] 修正识鸟日志显示，明确标记来源文件名

### Improvements
- [Log] 优化识鸟日志输出为非阻塞模式，实时反馈进度
- [Build] 统一版本号管理，确保各模块同步


## V4.0.3 (2026-02-01) - 摄影水平预设与 AI 识鸟

### New Features
- [New] 摄影水平预设 (Photography Skill Levels)
  - 新手 (Beginner): 锐度 > 300, 美学 > 4.5 (保留更多照片)
  - 初级 (Intermediate): 锐度 > 380, 美学 > 4.8 (推荐)
  - 大师 (Master): 锐度 > 520, 美学 > 5.5 (极致严格)
  
- [New] AI 鸟类识别 (Bird Species Identification)
  - 支持全球 11,000+ 种鸟类识别
  - 自动写入照片 EXIF/IPTC 元数据
  - 中英双语结果支持
  
- [New] Lightroom 插件集成
  - 在 Adobe Lightroom Classic 中直接调用 AI 识鸟
  - 无需导出即可查看识别结果

### Improvements
- [UI] 首次启动自动弹出水平选择向导
- [UI] 主界面参数区新增当前水平标签显示
- [Fix] 修复部分翻译显示的语言错误

---

## V4.0.2 (2026-01-25) - Bug 修复

### Bug Fixes
- [Fix] Intel Mac 启动崩溃问题修复
- [Fix] 连拍检测时间阈值逻辑优化
- [Fix] 部分 RAW 文件 EXIF 写入失败问题

---

## V4.0.1 (2026-01-20) - Windows 版本与对焦检测增强

### New Features
- [New] Windows 版本发布 (支持 NVIDIA GPU 加速)
- [New] 对焦点检测增强
  - 支持 Nikon Z6-3 DX 模式
  - 对焦在头部区域 (BEST) 锐度权重 x1.1
  - 对焦在身体区域 (GOOD) 无惩罚
  - 对焦在区域外 (BAD) 锐度权重 x0.7
  - 完全脱焦 (WORST) 锐度权重 x0.5

### Improvements
- [Perf] ExifTool 常驻进程优化，EXIF 写入速度提升 50%
- [Perf] 识鸟 GPS 区域缓存，避免重复网络请求

---

## V4.0.0 (2026-01-15) - 评分引擎重构

### Breaking Changes
- [Change] TOPIQ 替代 NIMA 作为美学评分模型
  - 更准确的画面美感评估
  - 全图评估而非裁剪区域

### New Features
- [New] 对焦点验证系统
  - 从 RAW 文件提取相机对焦点位置
  - 多层验证: 头部圆/分割掩码/BBox/画面边缘
  - 支持 Nikon, Sony, Canon, Olympus, Fujifilm, Panasonic
  
- [New] ISO 锐度归一化
  - 高 ISO 噪点会虚高锐度值
  - ISO 800 以上每翻倍扣 5%

### Improvements
- [Perf] 0 星和 -1 星照片跳过对焦检测，节省 ExifTool 调用
- [UI] 调试图显示对焦点位置、头部区域、分割掩码

---

## Downloads

### macOS Apple Silicon (M1/M2/M3/M4)
- GitHub: [v4.0.4 beta](https://github.com/jamesphotography/SuperPicky/releases/tag/v4.0.4)
- Google Drive: [SuperPicky_v4.0.4_Silicon.dmg](https://drive.google.com/file/d/1JpPJUjSe64YJL-E-4nn8lsIqtDXae7CH/view?usp=sharing)
- 百度网盘: [SuperPicky_v4.0.4_Silicon.dmg](https://pan.baidu.com/s/1OGiZYBFaKYpxPZDHfLxFwA?pwd=i1eq) 提取码: i1eq

### macOS Intel
- GitHub: [v4.0.4 beta](https://github.com/jamesphotography/SuperPicky/releases/tag/v4.0.4)
- Google Drive: [SuperPicky_v4.0.4_Intel.dmg](https://drive.google.com/file/d/1DghPd1yBRkMjEdM-GpxGNV8GFQBXgXiS/view?usp=sharing)
- 百度网盘: [SuperPicky_v4.0.4_Intel.dmg](https://pan.baidu.com/s/1iyDhNj1QP-Vg82QN9yoACQ?pwd=xh6u) 提取码: xh6u

### Windows (v4.0.4)

**CPU 版本 (1GB)**
- GitHub: [SuperPicky4.0.4_Win64_No_CUDA.zip](https://github.com/jamesphotography/SuperPicky/releases/download/v4.0.4/SuperPicky4.0.4_Win64_No_CUDA.zip)
- Google Drive: [SuperPicky_Win_v4.0.4_CPU](https://drive.google.com/file/d/1QWfDsb4L7wMoUoo17p6iiW1L9Lf-OAEv/view?usp=sharing)
- 百度网盘: [SuperPicky_Win_v4.0.4_CPU](https://pan.baidu.com/s/1zhWtMwAzthrLAeKxKf4SZw?pwd=2mrt) 提取码: 2mrt

**CUDA-GPU 版本 (3.4GB)**
- Google Drive: [SuperPicky_Win_v4.0.4_CUDA](https://drive.google.com/file/d/1QkBqBYxylpIlN7jByVUC3m9QEM40DYga/view?usp=sharing)
- 百度网盘: [SuperPicky_Win_v4.0.4_CUDA](https://pan.baidu.com/s/1dM79au9DpnWZQoWdUISPQA?pwd=1usg) 提取码: 1usg
