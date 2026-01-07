# SuperPicky - 慧眼选鸟 🦅

[![Version](https://img.shields.io/badge/version-3.8.0-blue.svg)](https://github.com/jamesphotography/SuperPicky)
[![Platform](https://img.shields.io/badge/platform-macOS%20|%20Windows-lightgrey.svg)](https://github.com/jamesphotography/SuperPicky/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**智能鸟类照片筛选工具 - 让AI帮你挑选最美的鸟类照片**

拍片一时爽，选片照样爽！一款专门为鸟类摄影师设计的智能照片筛选软件，使用多模型AI技术自动识别、评分和筛选鸟类照片，大幅提升后期整理效率。

---

## 🌟 核心功能

### 🤖 多模型协作
- **YOLO11 检测**: 精准识别照片中的鸟类位置和分割掩码
- **SuperEyes 鸟眼**: 检测鸟眼位置和可见度，计算头部区域锐度
- **SuperFlier 飞鸟**: 识别飞行姿态，给予飞版照片额外加分
- **TOPIQ 美学**: 评估整体画面美感、构图和光影

### ⭐ 智能评分系统 (0-3星)
| 星级 | 条件 | 含义 |
|------|------|------|
| ⭐⭐⭐ | 锐度达标 + 美学达标 | 优选照片，值得后期处理 |
| ⭐⭐ | 锐度达标 或 美学达标 | 良好照片，可考虑保留 |
| ⭐ | 有鸟但都未达标 | 普通照片，通常可删除 |
| 0 | 无鸟/质量太差 | 建议删除 |

### 🏷️ 特殊标记
- **Pick 精选**: 3星照片中锐度+美学双排名前25%的交集
- **Flying 飞鸟**: AI检测到飞行姿态，额外加分并标记绿色
- **Exposure 曝光** (可选): 检测过曝/欠曝问题，降一星处理

### 📂 自动整理
- **按星级分类**: 自动移动到 0星/1星/2星/3星 文件夹
- **EXIF写入**: 评分、旗标、锐度/美学值写入RAW文件元数据
- **Lightroom兼容**: 导入即可按评分排序和筛选
- **可撤销**: 一键重置恢复原始状态

---

## 📋 系统要求

- **macOS**: macOS 10.15+ · Apple Silicon (M1/M2/M3/M4) · 1.5GB空间
- **Windows**: Windows 10+ · NVIDIA GPU (建议) · 2GB空间

---

## 📥 下载安装

### macOS
- [GitHub 下载](https://github.com/jamesphotography/SuperPicky/releases/download/%E6%85%A7%E7%9C%BC%E9%80%89%E9%B8%9F/SuperPicky_v3.8.0.dmg) | [Google Drive](https://drive.google.com/file/d/1rn_VctgLMW8SOAAfm3I1tGX7W_RCcBgm/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1ezKPFsviTYfU_jFbmfAhxw?pwd=kdbx) 提取码: kdbx

1. 下载 `SuperPicky_v3.8.0.dmg`
2. 双击 DMG 文件，将应用拖入 Applications
3. 首次打开：右键点击应用选择"打开"

### Windows
- [GitHub 下载](https://github.com/jamesphotography/SuperPicky/releases/download/%E6%85%A7%E7%9C%BC%E9%80%89%E9%B8%9F/SuperPicky_v3.8_Win.zip) | [Google Drive](https://drive.google.com/file/d/19cmFJxdyzeUoGZBEidQILWRQsP-ChYHj/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/14PUWtAZiwSiCo1k_jRAaIw?pwd=a8xv) 提取码: a8xv

1. 下载并解压 `SuperPicky_v3.8_Win.zip`
2. 运行 `SuperPicky.exe`

### 从源码运行

```bash
git clone https://github.com/jamesphotography/SuperPicky.git
cd SuperPicky
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## 🚀 快速开始

1. **选择文件夹**: 拖入或浏览选择包含鸟类照片的文件夹
2. **调整阈值** (可选): 锐度阈值 (200-600)、美学阈值 (4.0-7.0)
3. **开关功能** (可选): 飞鸟检测、曝光检测
4. **开始处理**: 点击按钮等待AI处理完成
5. **查看结果**: 照片自动分类，导入Lightroom即可使用

---

## 📝 更新日志

### v3.8.0 (2026-01-02)
- ✨ **新增曝光检测**: 检测鸟区域过曝/欠曝，可选功能默认关闭
  - 过曝判定：亮度 ≥235 的像素超过 10%
  - 欠曝判定：亮度 ≤15 的像素超过 10%
  - 有曝光问题的照片评分降一星
- 📊 新增曝光问题统计和日志标签 【曝光】
- 🎚️ 曝光阈值可在高级设置中调整 (5%-20%)

### v3.7.0 (2026-01-01)
- ✨ 重构评分逻辑，使用 TOPIQ 替代 NIMA
- 🦅 飞鸟检测加成：锐度+100，美学+0.5
- 👁️ 眼睛可见度封顶逻辑优化
- 🔧 UI 优化和 Bug 修复

### v3.6.0 (2025-12-30)
- ✨ 飞鸟照片绿色标签
- 📊 飞鸟统计计数
- 🔄 纯JPEG文件支持

---

## 👨‍💻 开发团队

| 角色 | 成员 | 贡献 |
|------|------|------|
| 开发者 | [James Yu (詹姆斯·于震)](https://github.com/jamesphotography) | 核心开发 |
| 模型训练 | [Jordan Yu (于若君)](https://github.com/jordan-yrj) | SuperEyes · SuperFlier |
| Windows版 | [小平](https://github.com/thp2024) | Windows移植 |

---

## 🙏 致谢

- [YOLO11](https://github.com/ultralytics/ultralytics) - Ultralytics 目标检测模型
- [TOPIQ](https://github.com/chaofengc/IQA-PyTorch) - Chaofeng Chen 等人的图像质量评估模型
- [ExifTool](https://exiftool.org/) - Phil Harvey 的 EXIF 处理工具

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

**让SuperPicky成为你鸟类摄影的得力助手！** 🦅📸
